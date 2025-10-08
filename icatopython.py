import numpy as np
import joblib 
import streamlit as st
from PIL import Image

model=joblib.load('model.pkl')
le=joblib.load('le.pkl')
oe=joblib.load('oe.pkl')

st.set_page_config(page_title="Insurance Claim Approval", layout="wide")
col1,col2,col3=st.columns([1,3,1])
with col2:
    st.title("Insurance Claim Approval Prediction System")
    st.header("AI-Powered Insurance Claim Decision Support")
with col2:
    st.image("policy_approval/Insurance-claims.jpg",use_column_width=True)
col1,col2,col3=st.columns([1,4,1])
with col2:
    st.markdown(""" 'This project leverages machine learning to predict the likelihood of insurance 
            claim approval based on customer,policy, and claim-related attributes. By analyzing
            key factors such as fraud risk, documentation, claim vs. coverage amount, 
            and claim history, the system provides a transparent, data-driven decision framework. 
            The solution is designed with a realistic dataset, branded UI, and stakeholder-friendly reporting,
            making it both technically robust and business-ready.'""")
col1,col2,col3=st.columns([1,4,1])
with col2:
    age=st.number_input("Enter your age",min_value=10,max_value=99)
    annual_income=st.slider("Enter Your Annual Income",min_value=100,max_value=600000,step=100)
    poli_options=['Health', 'Auto', 'Life', 'Property']
    policy_type=st.selectbox("Enter policy Type",poli_options)
    coverage_amount=st.slider("Expected Coverage",min_value=100,max_value=600000,step=100)
    claim_amount=st.slider("Claim Amount",min_value=100,max_value=200000,step=100)
    past_claim=st.number_input("Past Claim",min_value=0,max_value=15,step=1)
    doc_sbmt=st.number_input("How Many Documment Submitted",min_value=0,max_value=15,step=1)
    fraud_option=['Low', 'Medium','High']
    fraud_risk=st.select_slider("Fraud Risk",fraud_option)
col1,col2,col3=st.columns([1,2,3])
with col2:
    if st.button("Click"):
        try:
            policy_type = le.transform([policy_type])[0]          # LabelEncoder
            fraud_risk = oe.transform([[fraud_risk]])[0][0]       # OrdinalEncoder
        
            features = [[age, annual_income, policy_type, coverage_amount,
                        claim_amount, past_claim, doc_sbmt, fraud_risk]]
        
            prediction = model.predict(features)
        
            if prediction[0] == 1:
                st.success("✅ Claim Approved: The submitted claim meets policy requirements.")
            else:
                st.error("❌ Claim Not Approved: Please review documentation and claim details.")
        except Exception as e:
            st.error(f'Error in Prediction: {e}')

        

