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

# Convertimos los datos en un DataFrame para que el modelo pueda procesarlo
if st.button("Predecir"):
    data = pd.DataFrame([[credit_score, annual_income, monthly_debt, years_credit_history]], 
                        columns=["Credit Score", "Annual Income", "Monthly Debt", "Years of Credit History"])
    prediction = modelo.predict(data)
    result = "Fully Paid ✅" if prediction[0] == 1 else "Charged Off ❌"
    st.subheader(f"Resultado: {result}")
