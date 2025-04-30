# Facial Recognition System

A comprehensive facial recognition system built with Python, designed to train, evaluate, and deploy facial recognition models. The application features a web interface for real-time inference and is structured for scalability and ease of use.

## 📁 Project Structure

```
Facial-recognition/
├── app.py                 # Flask web application entry point
├── config.py              # Configuration settings
├── train.py               # Script to train the facial recognition model
├── evaluate.py            # Script to evaluate model performance
├── inference.py           # Script for running inference on new data
├── requirements.txt       # List of project dependencies
├── README.md              # Project documentation
├── qodana.sarif.json      # Static code analysis report
├── MODEL_NAME/            # Directory for storing model-related files
├── checkpoints/           # Directory for saving model checkpoints
├── data/                  # Dataset for training and testing
├── models/                # Model architecture definitions
├── static/                # Static assets (e.g., CSS, JavaScript)
├── templates/             # HTML templates for the Flask app
└── utils/                 # Utility functions and helpers
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8–3.11
- [Homebrew](https://brew.sh/) (for macOS users)
- pip (Python package installer)
- Virtual environment tool (e.g., `venv`)

### Installation

```bash
# Clone the repository
git clone https://github.com/revanthkolluri369/Facial-recognition.git
cd Facial-recognition

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If you're using an Apple Silicon (M1/M2) Mac, consider replacing `tensorflow` with `tensorflow-macos` in `requirements.txt` for compatibility.

### Running the Web Application

```bash
python app.py
```

Access the application by navigating to `http://localhost:5000` in your web browser.

## 🧠 Training the Model

To train the facial recognition model:

```bash
python train.py
```

Ensure your dataset is properly formatted and placed in the `data/` directory.

## 🔍 Running Inference

To perform inference using the trained model:

```bash
python inference.py
```

This will process new images and output recognition results.

## 📊 Evaluating Model Performance

To evaluate the model's performance on a test dataset:

```bash
python evaluate.py
```

Review the output metrics to assess accuracy and other performance indicators.

## 🧰 Utilities

The `utils/` directory contains helper functions and scripts to support data preprocessing, model evaluation, and other tasks.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

