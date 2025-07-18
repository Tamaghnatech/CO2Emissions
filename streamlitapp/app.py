# app.py

#%% 🚀 Imports
import streamlit as st
import pandas as pd
import pickle
import os
import json
import plotly.express as px
import altair as alt
from sklearn.inspection import permutation_importance

#%% 🌐 Language Translations
translations = {
    "English": {
        "title": "🚗 CO₂ Emissions Dashboard",
        "intro": "ML-powered engine built with 💻 + ☕ + 🔥 by",
        "input_header": "🛠 Enter Engine Specifications",
        "engine": "🛠 Engine Size (L)",
        "cylinders": "⚙️ Cylinders",
        "fuel": "⛽ Fuel Consumption (L/100 km)",
        "predict": "🎯 CO₂ Emissions Prediction",
        "estimated": "💨 Estimated Emissions",
        "download": "📤 Download Prediction History as CSV",
        "get_csv": "⬇️ Get CSV",
        "make_lookup": "🚘 Lookup Average CO₂ by Car Make/Model",
        "plotly_chart": "📊 Plotly: Engine Size vs CO₂ Emissions",
        "altair_chart": "📈 Altair: Fuel Consumption vs CO₂ Emissions",
        "importance": "🧠 Feature Importance (Permutation)",
        "model": "🎛 Choose Model",
        "best_model": "📌 Best Model",
        "footer": "🛠 App v2.0 | 👑 Built by Lord Nag | 🌐 GitHub | 📈 W&B Dashboard"
    },
    "Spanish": {
        "title": "🚗 Panel de Emisiones de CO₂",
        "intro": "Motor impulsado por ML creado con 💻 + ☕ + 🔥 por",
        "input_header": "🛠 Ingrese especificaciones del motor",
        "engine": "🛠 Tamaño del motor (L)",
        "cylinders": "⚙️ Cilindros",
        "fuel": "⛽ Consumo de combustible (L/100 km)",
        "predict": "🎯 Predicción de emisiones de CO₂",
        "estimated": "💨 Emisiones estimadas",
        "download": "📤 Descargar historial de predicciones como CSV",
        "get_csv": "⬇️ Obtener CSV",
        "make_lookup": "🚘 Buscar promedio de CO₂ por marca/modelo",
        "plotly_chart": "📊 Plotly: Tamaño del motor vs Emisiones CO₂",
        "altair_chart": "📈 Altair: Consumo vs Emisiones CO₂",
        "importance": "🧠 Importancia de características (Permutación)",
        "model": "🎛 Elegir modelo",
        "best_model": "📌 Mejor modelo",
        "footer": "🛠 App v2.0 | 👑 Creado por Lord Nag | 🌐 GitHub | 📈 W&B Dashboard"
    },
    "German": {
        "title": "🚗 CO₂-Emissions-Dashboard",
        "intro": "ML-angetriebener Motor gebaut mit 💻 + ☕ + 🔥 von",
        "input_header": "🛠 Geben Sie Motorspezifikationen ein",
        "engine": "🛠 Motorgröße (L)",
        "cylinders": "⚙️ Zylinder",
        "fuel": "⛽ Kraftstoffverbrauch (L/100 km)",
        "predict": "🎯 CO₂-Emissionsvorhersage",
        "estimated": "💨 Geschätzte Emissionen",
        "download": "📤 Prognoseverlauf als CSV herunterladen",
        "get_csv": "⬇️ CSV herunterladen",
        "make_lookup": "🚘 Durchschnittlicher CO₂-Ausstoß nach Marke/Modell",
        "plotly_chart": "📊 Plotly: Motorgröße vs CO₂-Emissionen",
        "altair_chart": "📈 Altair: Verbrauch vs CO₂-Emissionen",
        "importance": "🧠 Merkmalswichtigkeit (Permutation)",
        "model": "🎛 Modell wählen",
        "best_model": "📌 Bestes Modell",
        "footer": "🛠 App v2.0 | 👑 Erstellt von Lord Nag | 🌐 GitHub | 📈 W&B Dashboard"
    }
}

#%% 💾 Load Model and Metrics
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)

#%% 🌍 Load Dataset
data_path = "data/CO2_Emissions_cleaned.csv"
df = pd.read_csv(data_path) if os.path.exists(data_path) else None

#%% 🎨 Config
st.set_page_config(page_title="🚗 CO₂ App", layout="wide")

#%% 🌐 Language Selection
lang = st.sidebar.selectbox("🌐 Language", ["English", "Spanish", "German"])
T = translations[lang]

#%% 🚀 App Header
st.title(T["title"])
st.caption(f"{T['intro']} **Tamaghna Nag**")

#%% 🎛 Sidebar Model Toggle
st.sidebar.radio(T["model"], ["Polynomial", "Spline"])
st.sidebar.metric(T["best_model"], metrics['Best_Model'])
st.sidebar.metric("📉 RMSE (Poly)", metrics.get("Poly_deg_4_RMSE", 0))
st.sidebar.metric("📈 R² (Poly)", metrics.get("Poly_deg_4_R2", 0))
st.sidebar.metric("📉 RMSE (Spline)", metrics.get("Spline_RMSE", 0))

#%% 📥 Input Form
st.header(T["input_header"])
c1, c2, c3 = st.columns(3)
engine = c1.slider(T["engine"], 1.0, 7.0, 3.0, 0.1)
cyl = c2.slider(T["cylinders"], 2, 16, 4)
fuel = c3.slider(T["fuel"], 2.0, 25.0, 8.0, 0.5)

input_df = pd.DataFrame([[engine, cyl, fuel]], 
    columns=["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"])

#%% 🎯 Prediction Output
st.subheader(T["predict"])
try:
    pred = model.predict(input_df)[0]
    st.success(f"{T['estimated']}: **{pred:.2f} g/km**")
except Exception as e:
    st.error(f"❌ Prediction Failed: {e}")

#%% 💾 Prediction History
if "history" not in st.session_state:
    st.session_state["history"] = []

st.session_state["history"].append({
    "Engine Size": engine, "Cylinders": cyl, "Fuel": fuel, "Predicted CO₂": round(pred, 2)
})

if st.button(T["download"]):
    df_hist = pd.DataFrame(st.session_state["history"])
    st.download_button(T["get_csv"], df_hist.to_csv(index=False), file_name="prediction_history.csv")

#%% 🚘 Make/Model CO₂ Lookup
if df is not None:
    with st.expander(T["make_lookup"]):
        make = st.selectbox("Make", sorted(df["Make"].dropna().unique()))
        model_sel = st.selectbox("Model", sorted(df[df["Make"] == make]["Model"].dropna().unique()))
        match = df[(df["Make"] == make) & (df["Model"] == model_sel)]
        if not match.empty:
            avg = match["CO2 Emissions(g/km)"].mean()
            st.info(f"🔍 Avg CO₂ for **{make} {model_sel}**: {avg:.2f} g/km")

#%% 📊 Plotly Chart
if df is not None:
    st.subheader(T["plotly_chart"])
    fig = px.scatter(df, x="Engine Size(L)", y="CO2 Emissions(g/km)", color="Fuel Type",
                     title="Engine Size vs CO₂", height=400)
    st.plotly_chart(fig, use_container_width=True)

#%% 📈 Altair Chart
if df is not None:
    st.subheader(T["altair_chart"])
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x="Fuel Consumption Comb (L/100 km)",
        y="CO2 Emissions(g/km)",
        color="Fuel Type",
        tooltip=["Make", "Model", "Fuel Type"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

#%% 🧠 Feature Importance
if df is not None:
    st.subheader(T["importance"])
    try:
        result = permutation_importance(
            model,
            df[["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"]],
            df["CO2 Emissions(g/km)"],
            n_repeats=10, random_state=42
        )
        fi_df = pd.DataFrame({
            "Feature": ["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"],
            "Importance": result.importances_mean
        }).sort_values("Importance", ascending=False)

        bar = px.bar(fi_df, x="Feature", y="Importance", title="Permutation-Based Feature Importance")
        st.plotly_chart(bar, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Importance Calculation Failed: {e}")

#%% 📌 Footer
st.markdown("---")
st.caption(T["footer"])
