import streamlit as st
import pandas as pd
import pickle
import requests
import io

# 1. Load the trained model from Hugging Face
MODEL_URL = "https://huggingface.co/khalidv5/churn-model/resolve/main/model.pkl"

@st.cache_resource # This ensures the model downloads only once
def load_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status() # Check if the download was successful
        model_file = io.BytesIO(response.content)
        return pickle.load(model_file)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

model = load_model()

# Define the full feature list (Must match X_train.columns order)
all_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 
               'MonthlyCharges', 'TotalCharges', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
               'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 
               'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
               'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 
               'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
               'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 
               'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
               'PaymentMethod_Mailed check'] 

# Page Configuration
st.set_page_config(page_title="Churn Predictor", page_icon="📞")
st.title("📞 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# 2. User Inputs
if model is not None:
    st.subheader("Customer Demographics & Contract Details")
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
        monthly = st.number_input("Monthly Charges ($)", value=70.0)
        total = st.number_input("Total Charges ($)", value=70.0)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    # 3. Prediction Logic
    if st.button("Predict Churn Status", use_container_width=True):
        # Create an empty DataFrame with 0s
        input_df = pd.DataFrame(0, index=[0], columns=all_columns)
        
        # Fill Numerical Values
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = monthly
        input_df['TotalCharges'] = total
        
        # Map Categorical Inputs (One-Hot Encoding)
        # Contract
        if contract == "One year":
            input_df['Contract_One year'] = 1
        elif contract == "Two year":
            input_df['Contract_Two year'] = 1
            
        # Internet Service
        if internet == "Fiber optic":
            input_df['InternetService_Fiber optic'] = 1
        elif internet == "No":
            input_df['InternetService_No'] = 1
            
        # Tech Support
        if tech_support == "Yes":
            input_df['TechSupport_Yes'] = 1
        elif tech_support == "No internet service":
            input_df['TechSupport_No internet service'] = 1

        # Get Prediction and Probabilities
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        
        # 4. Display Results
        if prediction[0] == 1:
            st.error(f"⚠️ **Result: Likely to Churn** (Probability: {prediction_proba:.2%})")
            st.write("Recommendation: Offer a loyalty discount or a contract upgrade to retain the customer.")
        else:
            st.success(f"✅ **Result: Likely to Stay** (Probability: {1 - prediction_proba:.2%})")
            st.write("Customer is satisfied with current services.")
else:
    st.warning("Model could not be loaded. Please check the Hugging Face link and your internet connection.")