# Welcome to Med Predict, a simple interactive tool designed to demonstrate the application of machine learning in healthcare.

Med Predict is a simple Streamlit app designed to demonstrate machine learning in healthcare. Currently, it features a Logistic Regression model that predicts diabetes risk based on user health inputs like glucose, insulin, BMI, and age. The app takes user input, runs predictions, and visualizes results including probability, model metrics, and ROC curves.


**Note:  For Demonstration Purposes only**


# Features

  🧪 Prediction with probability

  📈 Accuracy, Precision, ROC-AUC

  🔍 Feature importance via Plotly


# Built With

  | Component      | Library/Tool                   |
| -------------- | ------------------------------ |
| User Interface | Streamlit                      |
| ML Modeling    | scikit-learn                   |
| Data Wrangling | pandas, NumPy                  |
| Visualizations | Plotly Express & Graph Objects |
| Evaluation     | Accuracy, Precision, ROC-AUC   |

# How it Works

 1) Loads and cleans a diabetes dataset from Plotly's GitHub.

 2) Trains a logistic regression model with key health parameters.

 3) Provides sliders for user input (like glucose, insulin, age, etc.).

 4) Predicts diabetes risk and displays probability.

 5) Shows feature importance and ROC curve for model interpretability.


## Live Demo

[🔮 **MedPredict: Explore Machine Learning in Healthcare**](https://medpredictv1.streamlit.app/)

## 🚀 Run Locally

Make sure you have Python installed.

```bash
git clone https://github.com/yourusername/med-predict.git
cd med-predict
pip install requirements.txt
streamlit run medpredict.py


