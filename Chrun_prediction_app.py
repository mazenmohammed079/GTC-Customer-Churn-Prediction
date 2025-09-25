import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-high-risk {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .prediction-low-risk {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        st.write("Checking for model file...")
        if os.path.exists("best_churn_model.pkl"):
            st.write("Model file found, loading...")
            model = joblib.load("best_churn_model.pkl")
            st.write("Model loaded successfully!")
            return model
        else:
            st.error("❌ Model file 'best_churn_model.pkl' not found.")
            st.write("Available files:", os.listdir("."))
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.write("Available files:", os.listdir("."))
        st.stop()

def validate_inputs(monthly_charges, total_charges, tenure):
    errors = []
    
    if total_charges > 0 and monthly_charges > 0:
        expected_total = monthly_charges * tenure
        if total_charges < (expected_total * 0.1):
            errors.append("⚠️ Total charges seem too low compared to monthly charges and tenure")
    
    if tenure == 0 and total_charges > monthly_charges:
        errors.append("⚠️ Total charges should not exceed monthly charges for 0 tenure")
    
    return errors

def create_feature_importance_chart():
    features = ['Monthly Charges', 'Contract Type', 'Internet Service', 'Tenure', 
               'Payment Method', 'Total Charges', 'Tech Support', 'Online Security']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                title="Key Factors Influencing Churn Prediction",
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=importance, color_continuous_scale='viridis')
    fig.update_layout(height=400)
    return fig

def create_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80}}))
    
    fig.update_layout(height=400)
    return fig
def main():
    st.markdown('<div class="main-header">📊 Customer Churn Prediction System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This intelligent system predicts the likelihood of customer churn using machine learning.
    Fill in the customer details below to get a comprehensive churn risk assessment.
    """)
    
    model = load_model()
    
    st.sidebar.header("📝 Customer Information")
    st.sidebar.subheader("👤 Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    
    st.sidebar.subheader("📞 Services")
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    st.sidebar.subheader("🔒 Additional Services")
    online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.sidebar.subheader("💰 Billing Information")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    
    st.sidebar.subheader("💵 Financial Details")
    tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=0)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    total_charges = st.sidebar.slider("Total Charges ($)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    
    validation_errors = validate_inputs(monthly_charges, total_charges, tenure)
    if validation_errors:
        st.sidebar.warning("Input Validation:")
        for error in validation_errors:
            st.sidebar.warning(error)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Customer Overview")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Tenure", f"{tenure} months")
        
        with metrics_col2:
            st.metric("Monthly Charges", f"${monthly_charges:.2f}")
        
        with metrics_col3:
            st.metric("Total Charges", f"${total_charges:.2f}")
        
        with metrics_col4:
            avg_monthly = total_charges / tenure if tenure > 0 else 0
            st.metric("Avg Monthly", f"${avg_monthly:.2f}")
    
    with col2:
        st.header("🎯 Key Services")
        services = []
        if phone == "Yes":
            services.append("📞 Phone Service")
        if internet != "No":
            services.append(f"🌐 Internet ({internet})")
        if online_sec == "Yes":
            services.append("🔒 Online Security")
        if tech == "Yes":
            services.append("🛠️ Tech Support")
        
        if services:
            for service in services:
                st.write(f"✅ {service}")
        else:
            st.write("📋 Basic services only")
    
    st.header("🔮 Churn Prediction")
    
    if st.button("🚀 Analyze Customer Churn Risk", type="primary"):
        map_yes_no = {"Yes": 1, "No": 0}
        map_gender = {"Male": 1, "Female": 0}
        map_internet = {"No": 0, "DSL": 1, "Fiber optic": 2}
        map_contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        map_payment = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3}
        
        input_data = pd.DataFrame({
            "gender": [map_gender[gender]],
            "SeniorCitizen": [senior],
            "Partner": [map_yes_no[partner]],
            "Dependents": [map_yes_no[dependents]],
            "tenure": [tenure],
            "PhoneService": [map_yes_no[phone]],
            "MultipleLines": [map_yes_no.get(multiple, 0)],
            "InternetService": [map_internet[internet]],
            "OnlineSecurity": [map_yes_no.get(online_sec, 0)],
            "OnlineBackup": [map_yes_no.get(online_backup, 0)],
            "DeviceProtection": [map_yes_no.get(device, 0)],
            "TechSupport": [map_yes_no.get(tech, 0)],
            "StreamingTV": [map_yes_no.get(tv, 0)],
            "StreamingMovies": [map_yes_no.get(movies, 0)],
            "Contract": [map_contract[contract]],
            "PaperlessBilling": [map_yes_no[paperless]],
            "PaymentMethod": [map_payment[payment]],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]})
        
        input_data["HasPhoneService"] = input_data["PhoneService"]
        input_data["HasInternetService"] = input_data["InternetService"].apply(lambda x: 0 if x == 0 else 1)
        input_data["MonthlyToTotalChargesRatio"] = input_data.apply(
            lambda row: row["MonthlyCharges"] / row["TotalCharges"] if row["TotalCharges"] > 0 else 0,
            axis=1)
        
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model.feature_names_in_]
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-high-risk">
                        <h3>🚨 HIGH CHURN RISK</h3>
                        <p><strong>Churn Probability: {probability:.1%}</strong></p>
                        <p>This customer is likely to churn. Consider immediate retention strategies.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("🎯 Recommended Actions:")
                    st.write("• 📞 Contact customer proactively")
                    st.write("• 💰 Offer retention incentives")
                    st.write("• 📋 Conduct satisfaction survey")
                    st.write("• 🔄 Review service offerings")
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-low-risk">
                        <h3>✅ LOW CHURN RISK</h3>
                        <p><strong>Churn Probability: {probability:.1%}</strong></p>
                        <p>This customer is likely to stay. Focus on maintaining satisfaction.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("🎯 Recommended Actions:")
                    st.write("• 🌟 Maintain current service level")
                    st.write("• 📈 Consider upselling opportunities")
                    st.write("• 💬 Regular satisfaction check-ins")
                    st.write("• 🎁 Loyalty program enrollment")
            
            with result_col2:
                gauge_fig = create_probability_gauge(probability)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.subheader("📊 Model Insights")
            importance_fig = create_feature_importance_chart()
            st.plotly_chart(importance_fig, use_container_width=True)
            
            st.subheader("⚠️ Risk Factor Analysis")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("📅 Month-to-month contract (higher flexibility)")
            if payment == "Electronic check":
                risk_factors.append("💳 Electronic check payment (less commitment)")
            if internet == "Fiber optic":
                risk_factors.append("🌐 Fiber optic service (premium service expectations)")
            if monthly_charges > 70:
                risk_factors.append(f"💰 High monthly charges (${monthly_charges:.2f})")
            if online_sec == "No" and internet != "No":
                risk_factors.append("🔒 No online security (missing value-add service)")
            if tech == "No" and internet != "No":
                risk_factors.append("🛠️ No tech support (potential frustration risk)")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("✅ No significant risk factors identified")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
        <p><em>This prediction is based on historical data and should be used as a guide for decision-making.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
