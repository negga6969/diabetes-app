import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import base64
import os

# Debugging: Print current working directory and file path
print(f"Current working directory: {os.getcwd()}")
print(f"Checking for file at: {os.path.abspath('diabetes.csv')}")

# Check if the dataset file exists
# If you are still getting an error after confirming the file is in the same directory,
# you can use an absolute path here for a guaranteed fix.
# For example: data_path = r"C:\Users\YourUser\Documents\diabetes.csv"
data_path = 'diabetes.csv'
if not os.path.exists(data_path):
    st.error(f"Error: '{data_path}' not found. Please make sure the file is in the same directory as the script.")
    st.stop()

# Load dataset
data = pd.read_csv(data_path)
X = data.drop('Outcome', axis=1)
Y = data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Function to set background video
def set_background_video(video_path):
    # Check if the video file exists before trying to display it
    if not os.path.exists(video_path):
        st.warning(f"Warning: '{video_path}' not found. The background will be black.")
        return

    st.markdown(
    f"""
    <style>
    .stApp {{
        position: relative;
    }}
    #video-background {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -1;
        background-size: cover;
    }}
    </style>
    <video id="video-background" autoplay muted loop>
        <source src="{video_path}" type="video/mp4">
    </video>
    """,
    unsafe_allow_html=True
    )

# Set your background video (replace with your filename)
set_background_video("dnafinal.mp4")

# Custom CSS for fonts, buttons, inputs
st.markdown("""
    <style>
    /* Add a semi-transparent overlay to the main app container for better text readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 0;
    }
    
    /* Title font and color */
    .stTitle {
        font-family: 'Algerian', serif;
        text-align: center;
        color: #E3C39D;
    }

    /* Text color for all standard markdown and text elements */
    .main .block-container {
        color: #E3C39D;
    }
    
    /* Input boxes */
    .stNumberInput input {
        border-radius: 8px;
        border: 1px solid #10538A;
        padding: 6px;
        background-color: rgba(0,0,0,0.7);
        color: #E3C39D;
    }

    /* Buttons */
    button[kind="secondary"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #E3C39D !important;
        border-radius: 10px !important;
        padding: 0.6em 1.2em !important;
        font-size: 1em !important;
        font-weight: bold !important;
        border: 1px solid #10538A !important;
    }
    button[kind="secondary"]:hover {
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid #10538A !important;
    }

    /* Success message */
    .stSuccess {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #E3C39D !important;
        border-radius: 10px !important;
        padding: 10px !important;
        font-size: 1.1em;
        font-weight: bold;
    }

    /* Expander styling */
    .stExpander > div:first-child {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: #E3C39D !important;
        border-radius: 6px;
        font-weight: bold;
    }
    .stExpander > div:first-child svg {
        fill: #E3C39D !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="stTitle">Diabetes Prediction App</h1>', unsafe_allow_html=True)
st.write("Enter the following details to predict the likelihood of diabetes:")

# User inputs
input_data = []
for col in X.columns:
    value = st.number_input(f"Enter {col}", min_value=0.0, step=0.1)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")

# Model performance in expander
with st.expander("Model Performance"):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text("Classification Report")
    st.text(classification_report(Y_test, Y_pred))
