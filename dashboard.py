import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from privacy_ml_framework import (
    DifferentialPrivacy, PrivacyMLModel, generate_sample_data, create_privacy_accuracy_plot
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

random.seed(42)
st.set_page_config(
    page_title="Privacy-Accuracy Tradeoff Dashboard",
    layout="wide"
)

def main():
    st.header('Privacy-Accuracy Tradeoff Dashboard')
    
    st.sidebar.markdown("## Control Panel")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Choose Dataset",
        ["breast_cancer", "wine"],
        help="Select the dataset for privacy-accuracy experiments"
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "ML Model",
        ["random_forest", "logistic"],
        help="Choose the machine learning model"
    )
    
    # Privacy parameters
    epsilon_range = st.sidebar.slider(
        "Epsilon Range",
        min_value=0.1,
        max_value=10.0,
        value=(0.5, 5.0),
        step=0.1,
        help="Range of epsilon values for differential privacy"
    )
    
    num_epsilon_points = st.sidebar.slider(
        "Number of Epsilon Points",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of epsilon values to test"
    )
    
    st.markdown('Privacy-Accuracy Analysis', unsafe_allow_html=True)
    
    # Generate epsilon values
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_epsilon_points)
    
    # Generate data and run experiments
    with st.spinner("Generating dataset and running experiments..."):
        X, y, feature_names = generate_sample_data(dataset_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Run experiments
        accuracy_values = []
        privacy_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, epsilon in enumerate(epsilon_values):
            status_text.text(f"Testing epsilon = {epsilon:.2f}")
            
            # Train model with privacy
            model = PrivacyMLModel(model_type)
            X_private = model.train_with_privacy(X_train, y_train, epsilon)
            
            # Test accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
            
            # Calculate privacy score (higher epsilon = lower privacy)
            privacy_score = 1 / (1 + epsilon)
            privacy_scores.append(privacy_score)
            
            progress_bar.progress((i + 1) / len(epsilon_values))
        
        status_text.text("Analysis complete!")
        progress_bar.empty()
        status_text.empty()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Accuracy", f"{max(accuracy_values):.3f}")

    with col2:
        st.metric("Worst Accuracy", f"{min(accuracy_values):.3f}")
    
    with col3:
        st.metric("Accuracy Range", f"{max(accuracy_values) - min(accuracy_values):.3f}")

    
    fig = create_privacy_accuracy_plot(epsilon_values, accuracy_values, privacy_scores)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""

       #### **Privacy Score Calculation:**
       ```
       Privacy Score = 1/(1+ε)
       ```
       - Higher epsilon = Lower privacy
       - Lower epsilon = Higher privacy
       """)

    st.markdown("Results")
    results_df = pd.DataFrame({
        'Epsilon': epsilon_values,
        'Privacy Score': privacy_scores,
        'Accuracy': accuracy_values
    })
    
    st.dataframe(results_df.round(4), use_container_width=True)



if __name__ == "__main__":
    main()
