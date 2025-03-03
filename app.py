import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo previamente entrenado
modelo = joblib.load("XGBRFClassifier.pkl")

# Configuración de la aplicación
st.title("Predicción del Estado de Préstamos")
st.write("Ingrese los datos del cliente para predecir si el préstamo será pagado o incumplido.")

# Entradas del usuario
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
monthly_debt = st.number_input("Monthly Debt", min_value=0, value=1000)
years_credit_history = st.number_input("Years of Credit History", min_value=0, value=5)
open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=5)
credit_problems = st.number_input("Number of Credit Problems", min_value=0, value=0)
current_credit_balance = st.number_input("Current Credit Balance", min_value=0, value=10000)
max_open_credit = st.number_input("Maximum Open Credit", min_value=0, value=15000)
bankruptcies = st.number_input("Bankruptcies", min_value=0, value=0)
tax_liens = st.number_input("Tax Liens", min_value=0, value=0)

purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvements", "Other", "Personal"])
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own Home", "Home Mortgage"])
years_in_current_job = st.selectbox("Years in Current Job", ["0-1 year", "2-3 years", "4-6 years", "7-9 years", "10+ years"])

# Convertimos los datos en un DataFrame para que el modelo pueda procesarlo
if st.button("Predecir"):
  data = pd.DataFrame([[credit_score, annual_income, monthly_debt, years_credit_history, open_accounts, credit_problems, 
              current_credit_balance, max_open_credit, bankruptcies, tax_liens, purpose, home_ownership, years_in_current_job]], 
            columns=["Credit Score", "Annual Income", "Monthly Debt", "Years of Credit History", "Number of Open Accounts", 
                 "Number of Credit Problems", "Current Credit Balance", "Maximum Open Credit", "Bankruptcies", 
                 "Tax Liens", "Purpose", "Home Ownership", "Years in Current Job"])
  prediction = modelo.predict(data)
  result = "Fully Paid ✅" if prediction[0] == 1 else "Charged Off ❌"
  st.subheader(f"Resultado: {result}")
