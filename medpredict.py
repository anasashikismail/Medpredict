import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Med Predict", page_icon="🧠")

# Title and intro
st.title("Med Predict: Exploring Machine Learning in Healthcare")
st.markdown("""
Welcome to **Med Predict**, a simple interactive tool designed to demonstrate the application of machine learning in healthcare.

### 🤖 What is Machine Learning?
Machine learning is a subset of artificial intelligence that gives computers the ability to learn from data and make predictions or decisions without being explicitly programmed.

### 📈 What is Logistic Regression?
Logistic Regression is a supervised machine learning algorithm used for binary classification problems. It estimates the probability that a given input point belongs to a certain class — in this case, whether someone is likely to have diabetes or not.

---
Click the link below to go to the Diabetes Prediction module.

➡️ [Go to Diabetes Prediction](#diabetes-prediction-app)
""")

# Title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics.
Adjust the sliders and click 'Predict' to see the result.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    # Replace 0s with NaN in critical columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # Fill missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return model, acc, prec, fpr, tpr, roc_auc

model, acc, prec, fpr, tpr, roc_auc = train_model()

# Input widgets in sidebar
st.sidebar.header("Patient Details")

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 2)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 200, 120)
blood_pressure = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 40, 120, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
insulin = st.sidebar.slider("Insulin Level (IU/mL)", 0, 846, 80)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.5, 0.5)
age = st.sidebar.slider("Age", 21, 100, 30)

# Prediction button
if st.sidebar.button("Predict Diabetes Risk"):
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High risk of diabetes")
    else:
        st.success("✅ Low risk of diabetes")

    st.write(f"Probability of diabetes: {prediction_proba[1]*100:.2f}%")

    # Show model metrics
    st.markdown("### 🔢 Model Performance Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")

    # ROC Curve
    st.subheader("ROC Curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='red', dash='dash')))
    fig_roc.update_layout(title=f'ROC Curve (AUC = {roc_auc:.2f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc)

    # Show feature importance (coefficients)
    st.subheader("Key Factors Influencing Prediction")
    coefficients = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)

    fig = px.bar(coefficients, x='Feature', y='Importance', title='Feature Importance', color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig)

# Show dataset info
if st.checkbox("Show raw data"):
    st.subheader("Diabetes Dataset")
    st.write(df)

if st.checkbox("Show statistics"):
    st.subheader("Data Statistics")
    st.write(df.describe())

st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 18px;
        margin-top: 50px;
        animation: fadeIn 2s ease-in-out;
    }

    .footer .heart {
        color: red;
        animation: beat 1s infinite;
    }

    .footer {
        color: #e60000;
        font-weight: bold;
    }

    @keyframes beat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>

    <div class="footer">
        Made with <span class="heart">❤️</span> by <span class="name">Anas Ashik Ismail</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Run instructions
st.sidebar.markdown("""
**How to use:**
1. Adjust the sliders
2. Click 'Predict Diabetes Risk'
3. View results
""")
