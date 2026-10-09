# AI-Churn-Intelligence | Customer Churn Prediction

## About the Project

I built this project to understand how machine learning can help businesses identify customers who may be likely to leave their service.

For this project, I worked with a Telco Customer Churn dataset. I explored the data, looked for patterns associated with customer churn, and trained two machine learning models: Logistic Regression and Random Forest.

I also built a Streamlit web app where users can enter customer details and get an estimated churn probability.

This project helped me practice data cleaning, exploratory data analysis, model evaluation, and building a simple application around a trained machine learning model.

## What I Worked On

* Cleaned the dataset and checked for duplicate records and missing values.
* Performed exploratory data analysis (EDA) to understand customer churn patterns.
* Studied how factors such as contract type, tenure, monthly charges, and payment method relate to churn.
* Prepared categorical features for machine learning.
* Trained and compared Logistic Regression and Random Forest.
* Evaluated the models using accuracy, precision, recall, F1-score, and ROC-AUC.
* Saved the trained model and used it in a Streamlit application.

## Tools and Technologies

* **Python** — main programming language
* **Pandas** — data cleaning and manipulation
* **Matplotlib and Seaborn** — data visualization
* **Scikit-learn** — model training and evaluation
* **Streamlit** — interactive web application
* **Joblib** — saving and loading the trained model

## Model Comparison

I compared Logistic Regression and Random Forest using 5-fold stratified cross-validation.

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |               80.3% |         77.3% |
| Precision |               65.3% |         56.4% |
| Recall    |               54.6% |         63.0% |
| F1-score  |               59.5% |         59.5% |
| ROC-AUC   |               84.6% |         82.7% |

### What I Learned from the Results

Logistic Regression achieved better accuracy and ROC-AUC in this comparison. Random Forest achieved higher recall, meaning it identified a larger proportion of customers who churned.

This showed me that choosing a model is not just about accuracy. In a churn prediction problem, missing a customer who is about to leave may be costly, so recall is also important.

I also explored a lower classification threshold of 0.40 to identify more potential churners. This increased recall but also produced more false positives. The threshold results are exploratory and need further independent validation before real-world use.

## Key Observations from EDA

* Customer contract type showed differences in churn patterns.
* Tenure was an important variable to investigate when comparing customers who churned with those who stayed.
* Monthly charges and internet service type also showed differences between churn groups.
* The Random Forest feature-importance analysis ranked TotalCharges, tenure, and MonthlyCharges among the top three features.

These are patterns in the dataset, not proof that any single factor causes a customer to churn.

## How to Run the Project

### 1. Clone the repository

Replace the placeholder with your actual GitHub repository URL after uploading the project.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Churn-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL that you can open in your browser.

## Files Used by the App

The application loads these saved files from the project root:

* `app.py` — Streamlit application
* `random_forest_churn_model.pkl` — trained Random Forest model
* `churn_model_features.pkl` — feature names used by the model
* `churn_threshold.json` — selected classification threshold
* `requirements.txt` — Python dependencies

The project also includes dataset and analysis files used during development.

## Limitations and Future Work

This is a learning and portfolio project, so there is room to improve it further.

Some improvements I would like to explore are:

* Building a reusable preprocessing and prediction pipeline.
* Tuning model parameters and selecting a threshold using validation data.
* Improving the app's visualizations and prediction explanations.
* Testing the model on genuinely independent data.
* Deploying the app so other people can try it online.

## Final Note

This project is part of my learning journey in data science and machine learning. My goal was to understand the complete workflow, from exploring a dataset to training a model and using it in an interactive application.
