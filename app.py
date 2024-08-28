import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the HAR70+ model
model = load_model('har70_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the feature names (replace with actual feature names)
feature_names = ['back_x', 'back_y', 'back_z', 
                 'thigh_x', 'thigh_y', 'thigh_z']  # Modify as per your dataset

# Streamlit app
st.title("Human Activity Recognition (HAR70+)")
st.write("Input the accelerometer data to classify activity:")

# Create input fields for user inputs
st.header("Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input data to DataFrame for processing
input_df = pd.DataFrame([input_data])

# Ensure the DataFrame has the correct columns and order
input_df = input_df[feature_names]

# Button to classify
if st.button("Classify"):
    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the activity class
    prediction = model.predict(input_scaled)
    predicted_class = prediction.argmax(axis=-1)[0]  # Adjust based on your model

    # Map predicted class to activity label (modify as per your classes)
    activity_labels = {
        0: 'Walking', 
        1: 'None', 
        2: 'shuffling',
        3: 'stairs(ascending)',
        4: 'stairs(descending)',
        5: 'standing', 
        6: 'sitting',
        7: 'lying',
        # Add all class labels accordingly
    }

    # Display the result
    st.subheader("Prediction")
    st.markdown(
        f"""
    <div style="background-color: #121212; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: #4CAF50; font-family: Arial, sans-serif;">Prediction Result</h3>
        <p style="font-size: 20px; color: #ffffff; font-family: Arial, sans-serif;">The predicted activity is:</p>
        <h1 style="color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">{activity_labels.get(predicted_class, 'Unknown')}</h1>
    </div>
    """,
        unsafe_allow_html=True
    )
