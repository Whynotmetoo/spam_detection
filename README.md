# BERT-based Email Spam and Malware Detection

A machine learning project that uses BERT to detect spam and malicious emails. This project implements a fine-tuned BERT model for email classification with an interactive web interface.

## 🚀 Features

- BERT-based email spam detection
- Text preprocessing and cleaning
- Model training and evaluation
- Interactive web interface using Streamlit
- Model interpretability using SHAP/LIME
- Support for both text input and email file upload

## 📋 Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
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
├── data/
│   ├── raw/                # Raw email data (Enron + SpamAssassin)
│   └── processed.csv       # Processed dataset (text + labels)
├── src/
│   ├── preprocess.py       # Text preprocessing
│   ├── dataset.py          # Dataset construction
│   ├── train.py            # BERT fine-tuning
│   ├── evaluate.py         # Model evaluation
│   └── infer.py            # Inference functions
├── model/
│   └── best_model/         # Saved model checkpoints
├── app/
│   └── ui.py              # Streamlit web interface
├── requirements.txt
└── README.md
```

## 🚀 Usage

1. Data Preparation:
```bash
python src/preprocess.py
```

2. Model Training:
```bash
python src/train.py
```

3. Model Evaluation:
```bash
python src/evaluate.py
```

4. Launch Web Interface:
```bash
streamlit run app/ui.py
```

## 📊 Model Performance

The model is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC curve

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers
- Enron Dataset
- SpamAssassin Dataset 