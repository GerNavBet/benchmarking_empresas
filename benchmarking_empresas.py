import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Título ---
st.title("Benchmarking de Empresas Similares")
st.write("Introduce los datos de tu empresa para compararte con otras de características similares.")

# --- Dataset sintético de empresas ---
def generar_empresas_sinteticas():
    np.random.seed(42)
    sectores = ["Comercio", "Servicios", "Industria", "Tecnología"]
    empresas = []
    for _ in range(50):
        emp = {
            "Sector": np.random.choice(sectores),
            "Empleados": np.random.randint(1, 50),
            "Ingresos": np.random.randint(50000, 500000),
            "Costes": np.random.randint(30000, 400000),
            "Digitalización": np.random.randint(0, 3)
        }
        emp["Rentabilidad"] = (emp["Ingresos"] - emp["Costes"]) / emp["Ingresos"]
        empresas.append(emp)
    return pd.DataFrame(empresas)

empresas_df = generar_empresas_sinteticas()

# --- Formulario de usuario ---
with st.form("formulario_usuario"):
    sector = st.selectbox("Sector", ["Comercio", "Servicios", "Industria", "Tecnología"])
    empleados = st.number_input("Número de empleados", min_value=1, value=10)
    ingresos = st.number_input("Ingresos anuales (€)", min_value=10000, value=120000)
    costes = st.number_input("Costes anuales (€)", min_value=5000, value=70000)
    digital = st.selectbox("Nivel de digitalización", ["Bajo", "Medio", "Alto"])
    submitted = st.form_submit_button("Comparar")

# --- Benchmarking ---
def codificar(df):
    sector_map = {"Comercio": 0, "Servicios": 1, "Industria": 2, "Tecnología": 3}
    df = df.copy()
    df["Sector"] = df["Sector"].map(sector_map)
    return df

if submitted:
    rentabilidad = (ingresos - costes) / ingresos if ingresos != 0 else 0
    nueva_empresa = pd.DataFrame([{
        "Sector": sector,
        "Empleados": empleados,
        "Ingresos": ingresos,
        "Costes": costes,
        "Digitalización": {"Bajo": 0, "Medio": 1, "Alto": 2}[digital],
        "Rentabilidad": rentabilidad
    }])

    df_full = pd.concat([empresas_df, nueva_empresa], ignore_index=True)
    df_codificado = codificar(df_full)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(df_codificado[["Sector", "Empleados", "Ingresos", "Costes", "Digitalización", "Rentabilidad"]])
    similitudes = cosine_similarity([features[-1]], features[:-1])[0]

    top_indices = np.argsort(similitudes)[-5:][::-1]
    comparables = empresas_df.iloc[top_indices].copy()
    comparables["Similitud"] = similitudes[top_indices]

    st.subheader("Empresas Similares")
    st.dataframe(comparables)

    st.info("Este benchmarking permite identificar empresas de características cercanas a la tuya. Ideal para evaluar objetivos y posicionamiento relativo.")
