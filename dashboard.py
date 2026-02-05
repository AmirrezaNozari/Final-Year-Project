import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from privacy_ml_framework import (
    PrivacyMLModel, get_mnist_data, create_privacy_accuracy_plot, apply_tenseal_encryption,
    run_encrypted_inference
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import torch
import os

random.seed(42)
st.set_page_config(
    page_title="Privacy-Accuracy Tradeoff Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    return get_mnist_data()

def main():
    st.header('Privacy-Accuracy Tradeoff Dashboard')
    
    st.sidebar.markdown("## Control Panel")
    
    # # Dataset selection
    st.sidebar.info("Dataset: MNIST (Real World Handwritten Digits)")
    # dataset_type = st.sidebar.selectbox(
    #     "Choose Dataset",
    #     ["breast_cancer", "wine"],
    #     help="Select the dataset for privacy-accuracy experiments"
    # )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "ML Model",
        ["he_cnn", "logistic"],
        help="Opacus supports PyTorch Models (CNN)"
    )
    # model_type = st.sidebar.selectbox(
    #     "ML Model",
    #     ["random_forest", "logistic"],
    #     help="Choose the machine learning model"
    # )

    # HE Toggle
    # use_he = st.sidebar.checkbox(
    #     "Enable Homomorphic Encryption",
    #     value=False,
    #     help="Perform Homomorphic Encryption on input data before training"
    # )

    # Privacy parameters
    epsilon_range = st.sidebar.slider(
        "Epsilon Range",
        min_value=0.1,
        max_value=10.0,
        value=(0.5, 5.0),
        step=0.1,
        help="Range of epsilon values for differential privacy"
    )

    epochs = st.sidebar.number_input("Training Epochs", min_value=5, max_value=10, value=10)

    num_epsilon_points = st.sidebar.slider(
        "Number of Experiments",
        min_value=3,
        max_value=10,
        value=5
    )

    he_sample_mode = st.sidebar.radio(
        "Inference Scope (for HE Experiments)",
        ["Small Sample (Demo)", "Full Test Set (Slow)"],
        index=0
    )
    if he_sample_mode == "Small Sample (Demo)":
        he_samples_count = st.sidebar.slider("Number of Samples per Epsilon", 1, 50, 10)
    else:
        he_samples_count = -1  # Flag for full set
        st.sidebar.warning(
            f"Warning: You selected {num_epsilon_points} experiments. Running Full HE Inference for ALL of them will take significant time.")

    st.sidebar.markdown("---")
    
    if st.sidebar.button("Run Experiments"):
        run_experiments = True
    else:
        run_experiments = False
    # Generate epsilon values
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_epsilon_points)

    # Load Data
    with st.spinner("Loading MNIST dataset... (this may take a minute first time)"):
        X, y = load_data()

        # Subsample for Speed in Dashboard Demo
        # MNIST is 70k. Let's use 5k for responsiveness.
        indices = np.random.choice(len(X), 2000, replace=False)
        X = X[indices]
        y = y[indices]

    if run_experiments:
        st.markdown('Privacy-Accuracy Analysis', unsafe_allow_html=True)
        # if use_he:
        #     with st.spinner("Applying Homomorphic Encryption simulation (this may take a moment)..."):
        #         X = apply_he_to_data(X)

        st.warning("HE Simulation (Noise Injection) Enabled for Training Data.")
        X_sub = apply_tenseal_encryption(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_epsilon_points)
        accuracy_values = []
        he_accuracy_values = []
        privacy_scores = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        chart_placeholder = st.empty()

        for i, epsilon in enumerate(epsilon_values):
            status_text.text(f"Training with Epsilon = {epsilon:.2f}...")
            model = PrivacyMLModel(model_type)
            model.train_with_privacy(X_train, y_train, epsilon, epochs=epochs)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracy_values.append(acc)

            # Determine subset for this iteration

            if he_samples_count == -1:
                X_he_iter = X_test
                y_he_iter = y_test
            else:
                # Pick random subset or first N
                X_he_iter = X_test[:he_samples_count]
                y_he_iter = y_test[:he_samples_count]

            # Run HE
            he_res = run_encrypted_inference(model, X_he_iter, y_he_iter)
            he_accuracy_values.append(he_res['accuracy'])

            privacy_scores.append(1 / (1 + epsilon))
            progress_bar.progress((i + 1) / len(epsilon_values))

            he_acc_to_plot = he_accuracy_values if he_accuracy_values else None
            fig = create_privacy_accuracy_plot(epsilon_values[:i+1], accuracy_values, privacy_scores, he_acc_to_plot)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        status_text.text("Analysis complete!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy (Plain)", f"{max(accuracy_values):.3f}")
        with col2: st.metric("Best Accuracy (Real HE)", f"{max(he_accuracy_values):.3f}")
        with col3:
            st.metric("Privacy Cost", f"High: {min(epsilon_values)} - Low: {max(epsilon_values)}")

        st.markdown("### Results Table")

        data_dict = {'Target Epsilon': epsilon_values, 'Plaintext Accuracy': accuracy_values,
                     'Real HE Accuracy': he_accuracy_values}

        results_df = pd.DataFrame(data_dict)
        st.dataframe(results_df.round(4), use_container_width=True)
    else:
        st.info("Adjust parameters in the sidebar and click 'Run Experiments' to start.")

if __name__ == "__main__":
    main()
