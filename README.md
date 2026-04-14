# Employee Attrition Predictor

Machine learning solution to predict employee turnover and identify key attrition drivers using IBM HR Analytics data.

## Features
- ML Pipeline: Data cleaning, preprocessing, and Random Forest training.
- Explainability: SHAP values to identify and rank turnover drivers.
- Imbalance Handling: SMOTE for improved minority class detection.
- Dashboard: Web-based visualization of model insights and risk assessment.

## Quick Start
1. Install dependencies:
   pip install -r requirements.txt
2. Run the pipeline:
   python employee_attrition_predictor.py
3. View the dashboard:
   Open 'employee-attrition-frontend/index.html' in a browser.

## Project Structure
- employee_attrition_predictor.py: Core ML script.
- employee-attrition-frontend/: Web dashboard.
- outputs/: Model files and visualizations.
- requirements.txt: Python dependencies.

## Key Insights
Top attrition drivers identified:
- Stock Option Level
- Job Satisfaction
- Job Involvement
- Environment Satisfaction
- Overtime Work

## Tech Stack
Python (Pandas, Scikit-learn, SHAP, Seaborn), HTML/CSS/JS.
