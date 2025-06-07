import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import torch
from torch.utils.data import DataLoader

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from infer import EmailPredictor
from evaluate import ModelEvaluator
from dataset import load_datasets
from explain import EmailExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if 'predictor' not in st.session_state:
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA device for inference")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS device for inference")
    else:
        device = "cpu"
        logger.info("Using CPU device for inference")
    
    # Load the best model from the model directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'latest_model')
    st.session_state.predictor = EmailPredictor(model_dir=model_path, device=device)
    
    # Initialize explainer
    st.session_state.explainer = EmailExplainer(
        model=st.session_state.predictor.model,
        tokenizer=st.session_state.predictor.tokenizer,
        device=device
    )

# Initialize history for predictions
if 'history' not in st.session_state:
    st.session_state.history = []

def plot_confidence_bar(confidence: float, prediction: str) -> None:
    """
    Plot confidence bar.
    
    Args:
        confidence (float): Prediction confidence
        prediction (str): Prediction class
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create color gradient
    colors = ['#2ecc71', '#e74c3c']  # Green for ham, red for spam
    color = colors[1] if prediction == 'spam' else colors[0]
    
    # Plot horizontal bar
    ax.barh(
        [0],
        [confidence],
        color=color,
        alpha=0.6,
        height=0.5
    )
    
    # Set limits and remove y-axis
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    
    # Add text
    ax.text(
        0.5,
        0,
        f"{confidence:.1%}",
        ha='center',
        va='center',
        color='white',
        fontsize=20,
        fontweight='bold'
    )
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    st.pyplot(fig)
    plt.close()

def main():
    st.title("📧 Email Spam Detector")
    st.markdown("""
    This app uses a fine-tuned RoBERTa model to detect spam emails.
    You can either paste email text or upload an email file (.eml or .txt).
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the threshold for spam classification"
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Predict", "Explanation", "Model Performance"])
    
    with tab1:
        st.header("Make a Prediction")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload"]
        )
        
        if input_method == "Text Input":
            text = st.text_area(
                "Enter email text:",
                height=200,
                help="Paste the email content here"
            )
            
            if st.button("Predict"):
                if text:
                    with st.spinner("Analyzing..."):
                        progress = st.progress(0)
                        progress.progress(25)
                        result = st.session_state.predictor.predict(
                            text,
                            threshold=threshold
                        )
                        progress.progress(50)
                        
                        # Generate explanation
                        try:
                            feature_importance, explanation_fig = st.session_state.explainer.explain_prediction(
                                text,
                                result['prediction'],
                                result['confidence']
                            )
                            progress.progress(100)
                            
                            # Store explanation results
                            st.session_state.last_explanation = {
                                'feature_importance': feature_importance,
                                'explanation_fig': explanation_fig
                            }
                        except Exception as e:
                            logger.error(f"Error generating explanation: {str(e)}")
                            st.warning("Could not generate explanation for this text.")
                            progress.progress(100)
                        
                        st.session_state.history.append(result)
                        # Display result
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Prediction")
                            st.markdown(
                                f"### {'🚫 SPAM' if result['prediction'] == 'spam' else '✅ HAM'}"
                            )
                        with col2:
                            st.subheader("Confidence")
                            plot_confidence_bar(
                                result['confidence'],
                                result['prediction']
                            )
                else:
                    st.warning("Please enter some text to analyze.")
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload email file:",
                type=['eml', 'txt'],
                help="Upload an .eml or .txt file containing the email"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    with st.spinner("Analyzing..."):
                        progress = st.progress(0)
                        progress.progress(25)
                        result = st.session_state.predictor.predict(
                            temp_path,
                            threshold=threshold
                        )
                        progress.progress(50)
                        
                        # Generate explanation
                        try:
                            # Get email content for explanation
                            subject, body = st.session_state.predictor.extract_email_content(temp_path)
                            email_text = f"Subject: {subject}\n\nBody: {body}"
                            
                            feature_importance, explanation_fig = st.session_state.explainer.explain_prediction(
                                email_text,
                                result['prediction'],
                                result['confidence']
                            )
                            progress.progress(100)
                            
                            # Store explanation results
                            st.session_state.last_explanation = {
                                'feature_importance': feature_importance,
                                'explanation_fig': explanation_fig
                            }
                        except Exception as e:
                            logger.error(f"Error generating explanation: {str(e)}")
                            st.warning("Could not generate explanation for this email.")
                            progress.progress(100)
                        
                        st.session_state.history.append(result)
                        
                        # Display result
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Prediction")
                            st.markdown(
                                f"### {'🚫 SPAM' if result['prediction'] == 'spam' else '✅ HAM'}"
                            )
                        with col2:
                            st.subheader("Confidence")
                            plot_confidence_bar(
                                result['confidence'],
                                result['prediction']
                            )
                        
                        # Display email content
                        with st.expander("View Email Content"):
                            st.markdown(f"**Subject:** {subject}")
                            st.markdown("**Body:**")
                            st.text(body)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
        # Display prediction history
        if st.session_state.history:
            st.subheader("Prediction History")
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
    
    with tab2:
        st.header("Model Explanation")
        
        if 'last_explanation' in st.session_state:
            st.subheader("Feature Importance")
            
            # Display explanation plot
            st.pyplot(st.session_state.last_explanation['explanation_fig'])
            
            # Display feature importance table
            feature_importance = st.session_state.last_explanation['feature_importance']
            importance_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            st.dataframe(importance_df, use_container_width=True)
            
            st.markdown("""
            ### Understanding the Explanation
            
            The plot above shows the most important features (words or phrases) that influenced the model's prediction:
            
            - **Red bars** indicate features that contributed to a spam prediction
            - **Green bars** indicate features that contributed to a ham prediction
            - The length of each bar represents the strength of the feature's influence
            
            The table below shows the exact importance scores for each feature.
            """)
        else:
            st.info("Make a prediction first to see the explanation.")
    
    with tab3:
        st.header("Model Performance")
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating fine-tuned model..."):
                try:
                    # Load test data
                    _, _, test_dataset = load_datasets()
                    
                    # Create test loader
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=32,
                        shuffle=False
                    )
                    
                    # Get model path for fine-tuned model
                    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model')
                    if not os.path.exists(model_path):
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'latest_model')
                        if not os.path.exists(model_path):
                            st.error("No fine-tuned model found. Please train the model first.")
                            logger.error("No fine-tuned model found")
                            return
                    
                    try:
                        finetuned_evaluator = ModelEvaluator(
                            model_dir=model_path,  # Fine-tuned model
                            device=st.session_state.predictor.device
                        )
                        logger.info(f"Successfully loaded fine-tuned model from {model_path}")
                    except Exception as e:
                        st.error(f"Error loading fine-tuned model: {str(e)}")
                        logger.error(f"Error loading fine-tuned model: {str(e)}")
                        return
                    
                    # Evaluate fine-tuned model
                    try:
                        finetuned_metrics, finetuned_preds, finetuned_labels = finetuned_evaluator.evaluate(test_loader)
                    except Exception as e:
                        st.error(f"Error during model evaluation: {str(e)}")
                        logger.error(f"Error during model evaluation: {str(e)}")
                        return
                    
                    # Display metrics
                    st.subheader("Fine-tuned Model Performance")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        'Fine-tuned BERT': [
                            f"{finetuned_metrics['accuracy']:.1%}",
                            f"{finetuned_metrics['precision']:.1%}",
                            f"{finetuned_metrics['recall']:.1%}",
                            f"{finetuned_metrics['f1']:.1%}"
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Plot confusion matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    cm = pd.crosstab(
                        pd.Series(finetuned_labels, name='Actual'),
                        pd.Series(finetuned_preds, name='Predicted'),
                        normalize='index'
                    )
                    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues')
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")
                    logger.error(f"Error evaluating model: {str(e)}")

    

if __name__ == "__main__":
    main()