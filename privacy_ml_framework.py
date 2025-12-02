import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DifferentialPrivacy:
    """Basic differential privacy implementation"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, data, sensitivity=1.0):
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

class PrivacyMLModel:
    """ML model with privacy protection"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.privacy_engine = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
    
    def train_with_privacy(self, X, y, epsilon):
        """Train model with differential privacy"""
        self.privacy_engine = DifferentialPrivacy(epsilon=epsilon)
        
        # Add noise to training data
        X_private = self.privacy_engine.add_noise(X)
        
        # Train model on noisy data
        self.model.fit(X_private, y)
        return X_private
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

def generate_sample_data(dataset_type='synthetic'):
    """Generate sample datasets"""

    if dataset_type == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif dataset_type == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
    
    return X, y, feature_names

def create_privacy_accuracy_plot(epsilon_values, accuracy_values, privacy_scores):
    """Create basic privacy-accuracy plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=privacy_scores,
        y=accuracy_values,
        mode='markers+lines',
        name='Privacy vs Accuracy',
        line=dict(color='blue'),
        marker=dict(size=8, color=epsilon_values, colorscale='Viridis', 
                   colorbar=dict(title="Epsilon"))
    ))
    
    fig.update_layout(
        title="Privacy-Accuracy Tradeoff",
        xaxis_title="Privacy Score",
        yaxis_title="Accuracy",
        height=500
    )
    
    return fig
