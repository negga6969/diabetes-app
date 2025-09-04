import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# -------------------------------
# Background video
# -------------------------------
def set_background_video(video_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            position: relative;
            overflow: visible;
            background: transparent;
        }}
        #video-background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -100; /* Ensure video stays in the background */
        }}
        .main .block-container {{
            background: rgba(0,0,0,0.6);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.6);
            color: white;
            position: relative;
            z-index: 10; /* Ensure content is above video */
            max-width: 800px;
            margin: auto;
        }}
        /* Title styling */
        .stTitle {{
            font-family: 'Arial', sans-serif; /* Changed from Algerian for compatibility */
            text-align: center;
            color: #E3C39D;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        /* Input boxes */
        .stNumberInput input {{
            border-radius: 8px;
            border: 1px solid #10538A;
            padding: 6px;
            background-color: rgba(0,0,0,0.6);
            color: white;
        }}
        /* Buttons */
        div.stButton > button:first-child {{
            background-color: rgba(0,0,0,0.7);
            color: white;
            border-radius: 12px;
            padding: 0.7em 1.4em;
            font-size: 1.1em;
            font-weight: bold;
            border: 1px solid white;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
            transition: all 0.3s ease-in-out;
        }}
        div.stButton > button:first-child:hover {{
            background-color: rgba(0,0,0,0.9);
            color: #E3C39D;
            border: 1px solid #E3C39D;
            transform: scale(1.05);
        }}
        /* Prediction result box */
        .result-box {{
            background: rgba(0,0,0,0.7);
            padding: 1rem;
            margin-top: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
            box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
        }}
        </style>
        <video id="video-background" autoplay muted loop>
            <source src="{video_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

set_background_video("https://raw.githubusercontent.com/negga6969/diabetes-app/main/dnafinal.mp4")

# -------------------------------
# Load dataset
# -------------------------------
data = None
data_path = "diabetes.csv"

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.warning("diabetes.csv not found. Please upload it.")
    uploaded = st.file_uploader("Upload your diabetes.csv", type="csv")
    if uploaded:
        data = pd.read_csv(uploaded)

if data is not None and "Outcome" in data.columns:
    # Train model
    X = data.drop('Outcome', axis=1)
    Y = data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Title
    st.markdown('<h1 class="stTitle">Diabetes Prediction App</h1>', unsafe_allow_html=True)
    st.write("Enter your details below:")

    # Session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = [0.0] * len(X.columns)

    # Input fields
    input_data = []
    for i, col in enumerate(X.columns):
        value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1, value=float(st.session_state.inputs[i]), key=f"input_{i}")
        input_data.append(value)

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict", key="predict"):
            st.session_state.inputs = input_data
            prediction = model.predict([input_data])[0]
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
            st.markdown(f"<div class='result-box'>Prediction: {result}</div>", unsafe_allow_html=True)
    with col2:
        if st.button("Clear", key="clear"):
            st.session_state.inputs = [0.0] * len(X.columns)
            st.rerun()  # Updated from st.experimental_rerun()

    # Model performance
    with st.expander("Model Performance"):
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report")
        st.text(classification_report(Y_test, Y_pred))
else:
    st.error("Dataset missing or 'Outcome' column not found in diabetes.csv")
