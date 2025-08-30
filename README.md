# Student Performance Classification

This project implements a simple student performance classification pipeline (data generation, preprocessing, EDA, model training and evaluation) using Python.

Virtual environment: this repo assumes a virtual environment already exists at `.venv` (you mentioned it is already created). Activate it before running commands below.

Quick start (PowerShell):

```powershell
# Clone this repo by using the terminal
git clone https://github.com/naveedqadir/SmartEdu_Classifier.git

# Change to the cloned directory
cd SmartEdu_Classifier

# Create a venv
python3 -m venv .venv

# Activate the venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
python -m pip install -r requirements.txt

# Generate dataset
python src/generate_dataset.py

# Train and evaluate models
python src/train.py
```

What was added:
- `src/generate_dataset.py` - generates a synthetic dataset at `data/students.csv`.
- `src/train.py` - preprocessing, trains several classifiers, evaluates and saves best model to `models/best_model.joblib` and outputs plots to `outputs/`.
- `requirements.txt` - Python dependencies
- `notebooks/student_performance.ipynb` - exploratory notebook (Jupyter)

- `src/predict.py` - prediction CLI that loads `models/best_model.joblib` and predicts either a CSV batch or a single sample; saves batch predictions to `outputs/predictions.csv` by default.

Notes:
- Uses `.venv` in the repository root; ensure you activate it before running.
- The dataset is synthetic and created by `generate_dataset.py`.
