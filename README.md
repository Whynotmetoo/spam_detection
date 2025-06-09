# BERT-based Email Spam and Malware Detection

A machine learning project that uses BERT to detect spam and malicious emails. This project implements a fine-tuned BERT model for email classification with an interactive web interface.

## 🚀 Features

- BERT-based email spam detection with high accuracy
- Text preprocessing and cleaning for both email subjects and bodies
- Interactive web interface using Streamlit
- Model interpretability using SHAP/LIME
- Support for both text input and email file upload (.eml, .txt)
- Real-time prediction with confidence scores
- Detailed model performance metrics and visualizations
- Multi-language support (with English focus)
- Robust email parsing and content extraction

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.7.1+
- CUDA-capable GPU (optional but recommended for training)
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bert-spam-detector.git
cd bert-spam-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
bert-spam-detector/
├── app/
│   └── ui.py              # Streamlit web interface
├── data/
│   ├── raw/               # Raw email data
│   └── processed/         # Processed datasets
├── model/
│   ├── best_model/        # Best model checkpoints
│   └── latest_model/      # Latest model checkpoints
├── src/
│   ├── dataset.py         # Dataset construction
│   ├── evaluate.py        # Model evaluation
│   ├── explain.py         # Model interpretability
│   ├── infer.py           # Inference functions
│   ├── preprocess.py      # Text preprocessing
│   └── train.py           # BERT fine-tuning
├── datasets/              # Processed datasets
├── .streamlit/            # Streamlit configuration
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Usage

### 1. Data Preparation

First, prepare your datasets:

```bash
python src/preprocess.py
```

This will:
- Process raw email data
- Clean and normalize text
- Split into train/validation/test sets
- Save processed datasets

### 2. Model Training

Train the BERT model:

```bash
python src/train.py
```

Training features:
- Fine-tunes BERT on email classification
- Uses early stopping
- Saves best and latest models
- Tracks training metrics

### 3. Model Evaluation

Evaluate model performance:

```bash
python src/evaluate.py
```

This will generate:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC curve

### 4. Launch Web Interface

Start the interactive web app:

```bash
streamlit run app/ui.py
```

Features:
- Real-time spam detection
- File upload support
- Prediction confidence scores
- Model explanation using LIME
- Performance metrics visualization

## 📊 Model Performance

The model is evaluated using multiple metrics:
- Accuracy: Overall prediction accuracy
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve

## 🔍 Model Interpretability

The project includes model interpretability features:
- LIME-based feature importance analysis
- Visualization of key decision factors
- Confidence score explanation
- Interactive explanation interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Enron Email Dataset
- Spam Archive Dataset
- Streamlit for the web interface
- LIME for model interpretability

## 📧 Contact

For questions or feedback, please open an issue in the repository. 