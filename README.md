# capstone-project
customer response to insurance policy
# Insurance Customer Response Prediction

A machine-learning capstone project that predicts whether a customer will respond to an insurance policy offer. It includes exploratory data analysis, model comparison and tuning, plus a Streamlit web app and command-line prediction script.

## Project Contents

- `model/model_eda.ipynb` — data exploration, feature engineering, model training, evaluation, and hyperparameter tuning.
- `app.py` — interactive Streamlit prediction app.
- `predict.py` — command-line prediction script.
- `model/insurance_response_model.pkl` — saved tuned Random Forest pipeline used for predictions.
- `data (1).csv` — source dataset expected by the notebook.

## Analysis and Modeling

The notebook explores the dataset, checks its structure, missing values, duplicates, and target distribution, and visualizes customer and vehicle characteristics in relation to response.

It creates two additional features:
- `Vehicle_Age_Years`, a numeric representation of vehicle age.
- `Age_Group`, a group derived from customer age.

Logistic Regression, Decision Tree, and Random Forest models are trained and compared. Evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, classification reports, confusion matrices, and ROC curves. The notebook also examines Random Forest feature importance and tunes its hyperparameters using randomized search with stratified cross-validation and ROC-AUC scoring.

The final tuned Random Forest pipeline, including preprocessing, is saved as `model/insurance_response_model.pkl`.

## Setup

Use Python 3. Install the project dependencies:

```bash
py -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

Ensure `data (1).csv` is in the project root before running the notebook. Open `model/model_eda.ipynb` in Jupyter or VS Code and run its cells to repeat the analysis and training.

## Run Predictions

From the project root, start the web app:

```bash
py -m streamlit run app.py
```

Or use the interactive command-line script:

```bash
py predict.py
```

Both prediction tools load `model/insurance_response_model.pkl`, collect customer, vehicle, and policy details, and display the predicted response and response probability. Predictions use a classification threshold of `0.50`.

## Input Features

The model uses gender, age, driving-license status, region code, previous insurance status, vehicle age, vehicle damage, annual premium, policy sales channel, and customer vintage. Age group and numeric vehicle age are derived during feature engineering.

REQURIMENTS 

pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
jupyter