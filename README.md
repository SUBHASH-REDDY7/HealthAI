# HealthAI - Diabetes Risk Intelligence

An educational machine learning application that estimates diabetes risk using medical measurements and an SVM classifier.

## Features
- Diabetes risk estimation using eight medical features.
- Support Vector Machine (SVM) model.
- Configurable prediction threshold.
- Flask web application.
- Interactive patient assessment form.
- Prediction result with estimated model score.

## Technology Stack
- Python
- Flask
- Scikit-learn
- NumPy
- Joblib
- HTML and CSS

## Project Structure
- app.py
- requirements.txt
- models/model_config.json
- models/svm_diabetes_model.pkl
- templates/index.html

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    python app.py

Open http://127.0.0.1:5000 in your browser.

## Dataset
Pima Indians Diabetes Dataset.

## Medical Disclaimer
HealthAI is an academic demonstration and is not clinically validated. It must not be used to diagnose diabetes or make treatment decisions. Consult a qualified healthcare professional.

## Author
Subhash Reddy
