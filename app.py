import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model # Explicitly import load_model

# Set page config for better layout
st.set_page_config(layout="wide")

st.title("Customer Churn Prediction App")
st.write("### Predict if a customer will churn based on their characteristics.")

# Load the pre-trained model, scaler, and encoders
@st.cache_resource
def load_assets():
    try:
        model = load_model('model.keras')
        with open('/content/drive/MyDrive/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('/content/drive/MyDrive/onehot_encoder.geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        with open('/content/drive/MyDrive/label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        return model, scaler, onehot_encoder_geo, label_encoder_gender
    except Exception as e:
        st.error(f"Error loading necessary files: {e}")
        st.stop()

model, scaler, onehot_encoder_geo, label_encoder_gender = load_assets()

def preprocess_input(input_data, scaler, onehot_encoder_geo, label_encoder_gender):
    # Create a DataFrame from the input data
    input_df_temp = pd.DataFrame([input_data])

    # Label encode 'Gender'
    # Ensure gender is one of the classes seen during fit
    gender_options = list(label_encoder_gender.classes_)
    if input_df_temp['Gender'][0] not in gender_options:
        st.error(f"Invalid Gender: '{input_df_temp['Gender'][0]}'. Must be one of {gender_options}")
        st.stop()
    input_df_temp['Gender'] = label_encoder_gender.transform(input_df_temp['Gender'])

    # One-hot encode 'Geography'
    # Ensure geography is one of the categories seen during fit
    geo_options = list(onehot_encoder_geo.categories_[0])
    if input_df_temp['Geography'][0] not in geo_options:
        st.error(f"Invalid Geography: '{input_df_temp['Geography'][0]}'. Must be one of {geo_options}")
        st.stop()
    geo_encoded = onehot_encoder_geo.transform(input_df_temp[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Drop the original 'Geography' column and concatenate with one-hot encoded columns
    input_df_processed = input_df_temp.drop('Geography', axis=1)
    input_df = pd.concat([input_df_processed.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

    # Scale the input features
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Streamlit UI for input fields
with st.form("churn_prediction_form"):
    st.subheader("Customer Details:")
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 350, 850, 600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        balance = st.number_input("Balance", 0.0, 250000.0, 60000.0, format="%.2f")
    with col3:
        num_of_products = st.slider("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0, format="%.2f")

    submit_button = st.form_submit_button(label="Predict Churn")

if submit_button:
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    processed_input = preprocess_input(input_data, scaler, onehot_encoder_geo, label_encoder_gender)
    prediction = model.predict(processed_input)
    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result:")
    if prediction_proba > 0.5:
        st.error(f"Customer is likely to churn with a probability of {prediction_proba:.2f}")
    else:
        st.success(f"Customer is likely to stay with a probability of {prediction_proba:.2f}")

    st.write("
---
")
    st.write("**Note:** A probability above 0.5 indicates a higher likelihood of churning.")
