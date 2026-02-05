import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import tenseal as ts
from sklearn.datasets import fetch_openml
import time

# --- PyTorch Models ---

# class SimpleCNN(nn.Module):
#     """A simple CNN for MNIST/FashionMNIST"""
#
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         # Input: 28x28
#         self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)  # Output: (28+6-8)/2 + 1 = 14
#         self.pool = nn.MaxPool2d(2, 1)  # Output: 14-2+1 = 13 (Stride 1 reduces by kernel-1)
#
#         # Conv1: 28x28 -> 14x14 (k=3, s=2, p=1)
#         # Conv2: 14x14 -> 7x7 (k=3, s=2, p=1)
#
#         self.conv1 = nn.Conv2d(1, 16, 3, 2, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, 2, padding=1)
#
#         # Flatten: 32 * 7 * 7 = 1568
#         self.fc1 = nn.Linear(32 * 7 * 7, 32)
#         self.fc2 = nn.Linear(32, 10)
#
#     def forward(self, x):
#         # x: [B, 1, 28, 28]
#         x = torch.tanh(self.conv1(x))  # [B, 16, 14, 14]
#         x = torch.tanh(self.conv2(x))  # [B, 32, 7, 7]
#         x = x.view(x.size(0), -1)  # [B, 1568]
#         x = torch.tanh(self.fc1(x))
#         x = self.fc2(x)
#         return x


class HEFriendlyCNN(nn.Module):
    """
    CNN with Square activation for HE compatibility.
    """
    def __init__(self):
        super(HEFriendlyCNN, self).__init__()
        # Conv1: 7x7 kernel, stride 3. Output: 8x8. Channels: 4.
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3, padding=0)
        self.fc1 = nn.Linear(4 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x # Square
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x * x # Square
        x = self.fc2(x)
        return x


class LogisticRegressionTorch(nn.Module):
    """Linear model for simpler datasets"""

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([HEFriendlyCNN, LogisticRegressionTorch])


# --- Differential Privacy (Opacus) ---

class PrivacyMLModel:
    def __init__(self, model_type='cnn', input_dim=None):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if model_type == 'cnn':
        #     self.model = SimpleCNN().to(self.device)
        if model_type == 'he_cnn':
            self.model = HEFriendlyCNN().to(self.device)
        elif model_type == 'logistic':
            if input_dim is None: raise ValueError("input_dim required for logistic")
            self.model = LogisticRegressionTorch(input_dim, 10).to(self.device)

        # Monkey-patch torch.load
        original_load = torch.load
        try:
            torch.load = lambda f, map_location=None, weights_only=False, **kwargs: original_load(
                f,map_location=map_location,weights_only=False,**kwargs)
            self.model = ModuleValidator.fix(self.model)
            ModuleValidator.validate(self.model, strict=False)
        finally:
            torch.load = original_load

    def train_with_privacy(self, X_train, y_train, epsilon, epochs=5, batch_size=64):
        if self.model_type in ['he_cnn']:
            X_tensor = torch.FloatTensor(X_train).view(-1, 1, 28, 28)
        else:
            X_tensor = torch.FloatTensor(X_train)

        y_tensor = torch.LongTensor(y_train)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine()
        self.model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )

        self.model.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.model

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            if self.model_type in ['he_cnn']:
                X_tensor = torch.FloatTensor(X_test).view(-1, 1, 28, 28).to(self.device)
            else:
                X_tensor = torch.FloatTensor(X_test).to(self.device)

            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()


# --- Homomorphic Encryption (TenSEAL) ---

class EncFullyConnectedNet:
    """
    Encrypted Evaluation Wrapper for Hybrid Inference.
    Handles the Fully Connected layers encrypted.
    """

    def __init__(self, torch_model):
        # We assume input to this is already flattened features after Conv
        # FC1: [64, 256] -> we need T -> [256, 64]
        self.fc1_weight = torch_model.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_model.fc1.bias.data.tolist()

        # FC2: [10, 64] -> T -> [64, 10]
        self.fc2_weight = torch_model.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_model.fc2.bias.data.tolist()

    def forward(self, enc_x):
        # FC1
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias

        # Square activation
        enc_x.square_()

        # FC2
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def run_encrypted_inference(model, X_test, y_test, progress_callback=None):
    """
    Run encrypted inference (Hybrid: Plain Conv -> Encrypted FC).
    This avoids the instability of TenSEAL's conv2d_im2col on varying environments.
    """
    # Parameters
    bits_scale = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = 2 ** bits_scale
    context.generate_galois_keys()

    # Unwrap Opacus model
    if hasattr(model.model, '_module'):
        torch_inner = model.model._module
    else:
        torch_inner = model.model

    # Prepare Encrypted Part (FC Layers)
    enc_model = EncFullyConnectedNet(torch_inner)

    correct = 0
    times = []

    n_samples = len(X_test)
    print(f"Running Encrypted Inference (Hybrid) on {n_samples} samples...")

    # Move model to CPU
    torch_inner.cpu()

    for i in range(n_samples):
        start = time.time()

        # 1. Plain image
        image = torch.FloatTensor(X_test[i]).view(1, 1, 28, 28)
        target = y_test[i]

        # 2. Client-Side: Plaintext Feature Extraction (Conv + Square + Flatten)
        # We execute the first part of the network in plaintext
        with torch.no_grad():
            x = torch_inner.conv1(image)
            x = x * x  # Square
            x = x.view(x.size(0), -1)  # Flatten -> [1, 256]
            features = x.flatten().tolist()

        # 3. Encrypt Features
        enc_x = ts.ckks_vector(context, features)

        # 4. Server-Side: Encrypted Inference (FC1 -> Square -> FC2)
        enc_output = enc_model(enc_x)

        # 5. Decrypt Result
        output = enc_output.decrypt()
        pred = np.argmax(output)

        if pred == target:
            correct += 1

        elapsed = time.time() - start
        times.append(elapsed)

        if progress_callback:
            progress_callback(i + 1, n_samples, elapsed)

    return {
        'accuracy': correct / n_samples,
        'avg_time': np.mean(times),
        'total_time': np.sum(times)
    }


def apply_tenseal_encryption(X):
    # Simulation logic
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()

    sample_size = min(50, len(X))
    noise_diffs = []

    for i in range(sample_size):
        vec = X[i].tolist()
        enc = ts.ckks_vector(ctx, vec)
        dec = enc.decrypt()
        diff = np.array(dec) - np.array(vec)
        noise_diffs.append(diff)

    avg_noise_std = np.std(noise_diffs)

    noise = np.random.normal(0, avg_noise_std, X.shape)
    return X + noise


# --- Helpers ---

def get_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255.0
    return X.to_numpy(), y.to_numpy()


def create_privacy_accuracy_plot(epsilon_values, accuracy_values, privacy_scores, he_accuracy_values=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=privacy_scores, y=accuracy_values,
        mode='markers+lines', name='Opacus DP (Plaintext Inference)',
        line=dict(color='blue'),
        marker=dict(size=8, color=epsilon_values, colorscale='Viridis', colorbar=dict(title="Epsilon"))
    ))

    # Trace 2: Real Encrypted
    if he_accuracy_values:
        fig.add_trace(go.Scatter(
            x=privacy_scores, y=he_accuracy_values,
            mode='markers+lines', name='Opacus DP + Real Encrypted Inference',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8, symbol='x')
        ))

    fig.update_layout(
        title="Privacy-Accuracy Tradeoff: DP vs DP+HE",
        xaxis_title="Privacy Score (1/1+eps)",
        yaxis_title="Accuracy",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig