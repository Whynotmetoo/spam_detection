import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from infer import EmailPredictor
from evaluate import ModelEvaluator
from dataset import load_and_split_data

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
    st.session_state.predictor = EmailPredictor()

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
    This app uses a fine-tuned BERT model to detect spam emails.
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
    tab1, tab2 = st.tabs(["Predict", "Model Performance"])
    
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
                        progress.progress(100)
                        st.session_state.history.append(result)      # ← 新增
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
                            subject, body = st.session_state.predictor.extract_email_content(temp_path)
                            st.markdown(f"**Subject:** {subject}")
                            st.markdown("**Body:**")
                            st.text(body)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                # ✅ 新增：历史记录表
    if st.session_state.history:                 # ← 新增
        st.subheader("Prediction History")       # ← 新增
        hist_df = pd.DataFrame(st.session_state.history)   # ← 新增
        st.dataframe(hist_df, use_container_width=True)    # ← 新增
    
    with tab2:
        st.header("Model Performance")
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    # Load validation data
                    _, val_df = load_and_split_data('data/processed.csv')
                    
                    # Initialize evaluator
                    evaluator = ModelEvaluator()
                    
                    # Get predictions
                    probs = evaluator.predict_proba(val_df['text'].values)
                    preds = (probs[:, 1] >= threshold).astype(int)
                    labels = val_df['label'].values
                    
                    # Compute metrics
                    metrics, _, _ = evaluator.evaluate(
                        evaluator.trainer.model,
                        val_df['text'].values,
                        labels
                    )
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.1%}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.1%}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1']:.1%}")
                    
                    # Plot confusion matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = pd.crosstab(
                        pd.Series(labels, name='Actual'),
                        pd.Series(preds, name='Predicted'),
                        normalize='index'
                    )
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='.1%',
                        cmap='Blues',
                        xticklabels=['Ham', 'Spam'],
                        yticklabels=['Ham', 'Spam']
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    # Plot ROC curve
                    st.subheader("ROC Curve")
                    evaluator.plot_roc_curve(labels, probs[:, 1])
                    st.image('roc_curve.png')
                    
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")
                    logger.error(f"Error evaluating model: {str(e)}")

if __name__ == "__main__":
    main()