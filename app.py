import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuración de la página y estilos
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="wide")

# Load the trained model
model = joblib.load('XGBRFClassifier.pkl')

# Load encoders and scaler
purpose_encoder = LabelEncoder()
home_ownership_encoder = LabelEncoder()
job_years_encoder = LabelEncoder()
term_encoder = LabelEncoder()
scaler = StandardScaler()

# Fit the encoders with appropriate data
purpose_encoder.fit(['Debt Consolidation', 'Home Improvements', 'Other', 'Personal'])
home_ownership_encoder.fit(['Rent', 'Own Home', 'Home Mortgage'])
job_years_encoder.fit(['0-1 year', '2-3 years', '4-6 years', '7-9 years', '10+ years'])
term_encoder.fit(['Short Term', 'Long Term'])

# Fit the scaler with appropriate data
# Assuming you have a dataset named 'train_data' for fitting the scaler
train_data = pd.read_csv('train_data_processed.csv')
scaler.fit(train_data[['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt',
                       'Years of Credit History', 'Number of Open Accounts', 'Current Credit Balance',
                       'Maximum Open Credit']])

# Define the columns to be scaled
scaling_columns = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt',
                   'Years of Credit History', 'Number of Open Accounts', 'Current Credit Balance',
                   'Maximum Open Credit']

# Predefined profiles
profiles = {
    "Cliente Bajo Riesgo": {
        'Current Loan Amount': 2000,                # Préstamo muy pequeño
        'Credit Score': 980,                        # Puntuación de crédito casi perfecta
        'Annual Income': 500000,                    # Ingreso extremadamente alto
        'Monthly Debt': 50,                         # Deuda mensual mínima
        'Years of Credit History': 30,              # Historial de crédito extenso
        'Number of Open Accounts': 2,               # Muy pocas cuentas abiertas
        'Current Credit Balance': 100,              # Balance de crédito muy bajo
        'Maximum Open Credit': 200000,              # Crédito disponible muy alto
        'Bankruptcies': 0,                          # Sin bancarrotas
        'Term': 'Short Term',                       # Término corto
        'Years in current job': '10+ years',        # Máxima estabilidad laboral
        'Home Ownership': 'Own Home',               # Casa propia
        'Purpose': 'Home Improvements',             # Propósito de bajo riesgo
        'Number of Credit Problems': 0,             # Sin problemas de crédito
        'Tax Liens': 0                              # Sin gravámenes fiscales
    },
    "Cliente Promedio": {
        'Current Loan Amount': 15000,
        'Credit Score': 700,
        'Annual Income': 60000,
        'Monthly Debt': 1500,
        'Years of Credit History': 10,
        'Number of Open Accounts': 5,
        'Current Credit Balance': 10000,
        'Maximum Open Credit': 20000,
        'Bankruptcies': 1,
        'Term': 'Long Term',
        'Years in current job': '4-6 years',
        'Home Ownership': 'Home Mortgage',
        'Purpose': 'Debt Consolidation',
        'Number of Credit Problems': 1,
        'Tax Liens': 0
    },
    "Cliente Alto Riesgo": {
        'Current Loan Amount': 100000,              # Préstamo extremadamente grande
        'Credit Score': 300,                        # Puntuación de crédito pésima
        'Annual Income': 5000,                      # Ingreso extremadamente bajo
        'Monthly Debt': 10000,                      # Deuda mensual muy alta
        'Years of Credit History': 1,               # Historial de crédito mínimo
        'Number of Open Accounts': 20,              # Demasiadas cuentas abiertas
        'Current Credit Balance': 95000,            # Balance de crédito muy alto
        'Maximum Open Credit': 100,                 # Crédito disponible mínimo
        'Bankruptcies': 2,                          # Múltiples bancarrotas
        'Term': 'Long Term',                        # Término largo
        'Years in current job': '0-1 year',         # Sin estabilidad laboral
        'Home Ownership': 'Rent',                   # Renta
        'Purpose': 'Other',                         # Propósito de alto riesgo
        'Number of Credit Problems': 10,            # Muchos problemas de crédito
        'Tax Liens': 5                              # Múltiples gravámenes fiscales
    }
}

# Añadir una función para calcular métricas financieras
def calculate_financial_metrics(loan_amount, annual_income, monthly_debt):
    debt_to_income = (monthly_debt * 12) / annual_income if annual_income > 0 else float('inf')
    loan_to_income = loan_amount / annual_income if annual_income > 0 else float('inf')
    monthly_loan_payment = loan_amount / 12  # Simplificado, podría ser más complejo
    debt_burden = (monthly_debt + monthly_loan_payment) / (annual_income / 12)
    
    return {
        "Debt to Income Ratio": debt_to_income,
        "Loan to Income Ratio": loan_to_income,
        "Monthly Loan Payment": monthly_loan_payment,
        "Total Debt Burden": debt_burden
    }

# Modificar la interfaz principal
st.title("🏦 Predictor de Estado de Préstamos")
st.markdown("""
Este sistema utiliza machine learning para evaluar la probabilidad de que un préstamo sea pagado completamente 
o entre en estado de incumplimiento.
""")

# Crear pestañas para diferentes secciones
tab1, tab2 = st.tabs(["📝 Entrada de Datos", "📊 Análisis de Riesgo"])

with tab1:
    def user_input_features():
        profile_descriptions = ["Personalizado", "Cliente Bajo Riesgo", "Cliente Promedio", "Cliente Alto Riesgo"]

        selected_profile = st.sidebar.selectbox(
            "Selecciona un Perfil",
            options=profile_descriptions,
            format_func=lambda x: x
        )

        if selected_profile == 'Personalizado':
            st.sidebar.markdown("### 💰 Información Financiera")
            current_loan_amount = st.sidebar.number_input('Monto del Préstamo', min_value=0, value=10000, format="%d")
            annual_income = st.sidebar.number_input('Ingreso Anual', min_value=0, value=50000, format="%d")
            monthly_debt = st.sidebar.number_input('Deuda Mensual', min_value=0, value=1000, format="%d")

            st.sidebar.markdown("### 📈 Historial Crediticio")
            credit_score = st.sidebar.slider('Puntaje Crediticio', 0, 1000, 500)
            years_of_credit_history = st.sidebar.slider('Años de Historial Crediticio', 0, 80, 10)
            
            st.sidebar.markdown("### 🏦 Información Bancaria")
            number_of_open_accounts = st.sidebar.number_input('Cuentas Abiertas', min_value=0, value=5)
            current_credit_balance = st.sidebar.number_input('Balance de Crédito Actual', min_value=0, value=10000)
            maximum_open_credit = st.sidebar.number_input('Crédito Máximo Disponible', min_value=0, value=15000)

            st.sidebar.markdown("### ⚠️ Factores de Riesgo")
            bankruptcies = st.sidebar.number_input('Bancarrotas', min_value=0, max_value=5, value=0)
            number_of_credit_problems = st.sidebar.number_input('Problemas de Crédito', min_value=0, max_value=10, value=0)
            tax_liens = st.sidebar.number_input('Gravámenes Fiscales', min_value=0, max_value=5, value=0)

            col1, col2 = st.sidebar.columns(2)
            with col1:
                term = st.selectbox('Plazo', ['Short Term', 'Long Term'])
            with col2:
                purpose = st.selectbox('Propósito', ['Debt Consolidation', 'Home Improvements', 'Other', 'Personal'])

            st.sidebar.markdown("### 👤 Información Personal")
            years_in_current_job = st.sidebar.selectbox('Años en Trabajo Actual', 
                ['0-1 year', '2-3 years', '4-6 years', '7-9 years', '10+ years'])
            home_ownership = st.sidebar.selectbox('Tipo de Vivienda', ['Rent', 'Own Home', 'Home Mortgage'])

            data = {
                'Current Loan Amount': current_loan_amount,
                'Credit Score': credit_score,
                'Annual Income': annual_income,
                'Monthly Debt': monthly_debt,
                'Years of Credit History': years_of_credit_history,
                'Number of Open Accounts': number_of_open_accounts,
                'Current Credit Balance': current_credit_balance,
                'Maximum Open Credit': maximum_open_credit,
                'Bankruptcies': bankruptcies,
                'Term': term,
                'Years in current job': years_in_current_job,
                'Home Ownership': home_ownership,
                'Purpose': purpose,
                'Number of Credit Problems': number_of_credit_problems,
                'Tax Liens': tax_liens
            }
        else:
            data = profiles[selected_profile]

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    
    # Mostrar los datos de entrada en un formato más atractivo
    st.subheader("Datos de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Monto del Préstamo", f"${input_df['Current Loan Amount'].iloc[0]:,.2f}")
      st.metric("Años de Historial Crediticio", int(input_df['Years of Credit History'].iloc[0]))
      st.metric("Cuentas Abiertas", int(input_df['Number of Open Accounts'].iloc[0]))
      st.metric("Balance de Crédito Actual", f"${input_df['Current Credit Balance'].iloc[0]:,.2f}")
      st.metric("Bancarrotas", int(input_df['Bankruptcies'].iloc[0]))
    with col2:
      st.metric("Puntaje Crediticio", int(input_df['Credit Score'].iloc[0]))
      st.metric("Deuda Mensual", f"${input_df['Monthly Debt'].iloc[0]:,.2f}")
      st.metric("Crédito Máximo Disponible", f"${input_df['Maximum Open Credit'].iloc[0]:,.2f}")
      st.metric("Problemas de Crédito", int(input_df['Number of Credit Problems'].iloc[0]))
      st.metric("Gravámenes Fiscales", int(input_df['Tax Liens'].iloc[0]))
    with col3:
      st.metric("Ingreso Anual", f"${input_df['Annual Income'].iloc[0]:,.2f}")
      st.metric("Plazo", "Corto" if input_df['Term'].iloc[0] == 0 else "Largo")
      st.metric("Años en Trabajo Actual", input_df['Years in current job'].iloc[0])
      st.metric("Tipo de Vivienda", input_df['Home Ownership'].iloc[0])
      st.metric("Propósito", input_df['Purpose'].iloc[0])

with tab2:
    # Preprocess the input data
    input_df['Purpose'] = purpose_encoder.transform(input_df['Purpose'])
    input_df['Home Ownership'] = home_ownership_encoder.transform(input_df['Home Ownership'])
    input_df['Years in current job'] = job_years_encoder.transform(input_df['Years in current job'])
    input_df['Term'] = term_encoder.transform(input_df['Term'])
    input_df[scaling_columns] = scaler.transform(input_df[scaling_columns])

    # Ensure the order of columns matches the model's expected order
    expected_columns = [
        'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
        'Years in current job', 'Home Ownership', 'Purpose', 'Monthly Debt',
        'Years of Credit History', 'Number of Open Accounts', 'Number of Credit Problems',
        'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'
    ]
    input_df = input_df[expected_columns]

    # Display user input
    st.subheader('Datos ingresados trasnfomados:')
    st.write(input_df)

    # Make prediction
    prediction = model.predict_proba(input_df)
    # Ajustamos el umbral para ser más estricto (0.7 en lugar del default 0.5)
    prediction_class = (prediction[:, 1] > 0.7).astype(int)
    
    # Mostrar predicción con mejor visualización
    col1, col2 = st.columns(2)
    
    with col1:
      st.subheader("Resultado de la Predicción")
      if prediction_class[0] == 1:
        st.success(" Préstamo Aprobado", icon="✅")
        st.balloons()
      else:
        st.error(" Préstamo No Recomendado", icon="❌")
    
    with col2:
      # Crear un gauge chart con Plotly
      fig = px.pie(values=[prediction[0][0], prediction[0][1]], 
            names=['Riesgo', 'Seguro'],
            hole=0.7,
            title="Probabilidad de Éxito")
      fig.update_traces(marker=dict(colors=['#ff4b4b', '#4CAF50']))
      fig.update_layout(showlegend=False, annotations=[
        dict(text=f"{prediction[0][1]*100:.2f}%", x=0.5, y=0.5, font_size=20, showarrow=False)
      ])
      st.plotly_chart(fig)