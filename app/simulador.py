# ============================================================
# simulador.py
# Simulador interactivo de inversión en VUT — Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Añadimos el path del proyecto
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE)

from src.simulation.motor_financiero import ParametrosInversion, calcular_tres_escenarios

# ── CONFIGURACIÓN PÁGINA ─────────────────────────────────────
st.set_page_config(
    page_title="Simulador VUT — Inversión Turística",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CARGA DE MODELOS ─────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    MODELS = os.path.join(BASE, 'src', 'models')
    PROCESSED = os.path.join(BASE, 'data', 'processed')

    with open(os.path.join(MODELS, 'xgboost_ocupacion.pkl'), 'rb') as f:
        modelo_xgb = pickle.load(f)
    with open(os.path.join(MODELS, 'encoders_ocupacion.pkl'), 'rb') as f:
        encoders_xgb = pickle.load(f)
    with open(os.path.join(MODELS, 'rf_precio.pkl'), 'rb') as f:
        modelo_rf = pickle.load(f)
    with open(os.path.join(MODELS, 'encoders_rf_precio.pkl'), 'rb') as f:
        encoders_rf = pickle.load(f)
    with open(os.path.join(MODELS, 'prophet_forecast_madrid.pkl'), 'rb') as f:
        forecast_madrid = pickle.load(f)
    with open(os.path.join(MODELS, 'prophet_forecast_barcelona.pkl'), 'rb') as f:
        forecast_barcelona = pickle.load(f)

    df = pd.read_csv(os.path.join(PROCESSED, 'listings_clean.csv'), low_memory=False)

    return modelo_xgb, encoders_xgb, modelo_rf, encoders_rf, forecast_madrid, forecast_barcelona, df

modelo_xgb, encoders_xgb, modelo_rf, encoders_rf, forecast_madrid, forecast_barcelona, df = cargar_modelos()

# ── FUNCIONES DE PREDICCIÓN ───────────────────────────────────

def predecir_ocupacion(ciudad, barrio, room_type, accommodates, bedrooms,
                        bathrooms, beds, price, minimum_nights, availability_365,
                        host_is_superhost, host_listings_count, instant_bookable,
                        review_scores_rating, review_scores_cleanliness,
                        review_scores_location, review_scores_value):

    le_ciudad = encoders_xgb['le_ciudad']
    le_barrio = encoders_xgb['le_barrio']
    le_room   = encoders_xgb['le_room']

    try:
        ciudad_enc = le_ciudad.transform([ciudad])[0]
    except:
        ciudad_enc = 0
    try:
        barrio_enc = le_barrio.transform([barrio])[0]
    except:
        barrio_enc = 0
    try:
        room_enc = le_room.transform([room_type])[0]
    except:
        room_enc = 0

    X = pd.DataFrame([{
        'ciudad_enc':               ciudad_enc,
        'barrio_enc':               barrio_enc,
        'room_enc':                 room_enc,
        'accommodates':             accommodates,
        'bedrooms':                 bedrooms,
        'bathrooms':                bathrooms,
        'beds':                     beds,
        'price':                    price,
        'minimum_nights':           minimum_nights,
        'availability_365':         availability_365,
        'host_is_superhost':        host_is_superhost,
        'host_listings_count':      host_listings_count,
        'instant_bookable':         instant_bookable,
        'review_scores_rating':     review_scores_rating,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_location':   review_scores_location,
        'review_scores_value':      review_scores_value,
    }])

    ocupacion = modelo_xgb.predict(X)[0]
    return float(np.clip(ocupacion, 0, 1))


def predecir_precio(ciudad, barrio, room_type, accommodates, bedrooms,
                     bathrooms, beds, minimum_nights, availability_365,
                     host_is_superhost, host_listings_count, instant_bookable,
                     review_scores_rating, review_scores_cleanliness,
                     review_scores_location, review_scores_value):

    le_ciudad = encoders_rf['le_ciudad']
    le_barrio = encoders_rf['le_barrio']
    le_room   = encoders_rf['le_room']

    try:
        ciudad_enc = le_ciudad.transform([ciudad])[0]
    except:
        ciudad_enc = 0
    try:
        barrio_enc = le_barrio.transform([barrio])[0]
    except:
        barrio_enc = 0
    try:
        room_enc = le_room.transform([room_type])[0]
    except:
        room_enc = 0

    X = pd.DataFrame([{
        'ciudad_enc':               ciudad_enc,
        'barrio_enc':               barrio_enc,
        'room_enc':                 room_enc,
        'accommodates':             accommodates,
        'bedrooms':                 bedrooms,
        'bathrooms':                bathrooms,
        'beds':                     beds,
        'minimum_nights':           minimum_nights,
        'availability_365':         availability_365,
        'host_is_superhost':        host_is_superhost,
        'host_listings_count':      host_listings_count,
        'instant_bookable':         instant_bookable,
        'review_scores_rating':     review_scores_rating,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_location':   review_scores_location,
        'review_scores_value':      review_scores_value,
    }])

    precio = modelo_rf.predict(X)[0]
    return float(np.clip(precio, 20, 2000))


# ── INTERFAZ ─────────────────────────────────────────────────

st.title("🏠 Simulador de Inversión en Viviendas de Uso Turístico")
st.caption("TFM — Máster en IA aplicada a Entornos Empresariales y Financieros | UNIA 2026")

st.markdown("---")

# ── SIDEBAR: PARÁMETROS DEL INMUEBLE ─────────────────────────
with st.sidebar:
    st.header("📋 Parámetros del inmueble")

    ciudad = st.selectbox("Ciudad", ["madrid", "barcelona"])

    barrios_disponibles = sorted(df[df['ciudad'] == ciudad]['neighbourhood_cleansed'].unique())
    barrio = st.selectbox("Barrio", barrios_disponibles)

    room_type = st.selectbox("Tipo de alojamiento", [
        "Entire home/apt", "Private room", "Shared room", "Hotel room"
    ])

    col1, col2 = st.columns(2)
    with col1:
        accommodates  = st.number_input("Huéspedes", 1, 16, 4)
        bedrooms      = st.number_input("Habitaciones", 0, 10, 2)
        bathrooms     = st.number_input("Baños", 0.0, 10.0, 1.0, step=0.5)
    with col2:
        beds              = st.number_input("Camas", 1, 16, 2)
        minimum_nights    = st.number_input("Noches mínimas", 1, 30, 2)
        availability_365  = st.number_input("Días disponibles/año", 30, 365, 300)

    st.subheader("⭐ Puntuaciones esperadas")
    review_scores_rating      = st.slider("Valoración general", 1.0, 5.0, 4.5, 0.1)
    review_scores_cleanliness = st.slider("Limpieza", 1.0, 5.0, 4.5, 0.1)
    review_scores_location    = st.slider("Ubicación", 1.0, 5.0, 4.5, 0.1)
    review_scores_value       = st.slider("Relación calidad-precio", 1.0, 5.0, 4.3, 0.1)

    st.subheader("👤 Perfil del anfitrión")
    host_is_superhost    = st.checkbox("Superhost", value=False)
    host_listings_count  = st.number_input("Nº anuncios del host", 1, 100, 1)
    instant_bookable     = st.checkbox("Reserva instantánea", value=True)

# ── PREDICCIONES IA ───────────────────────────────────────────
ocupacion_pred = predecir_ocupacion(
    ciudad, barrio, room_type, accommodates, bedrooms, bathrooms, beds,
    100, minimum_nights, availability_365, int(host_is_superhost),
    host_listings_count, int(instant_bookable),
    review_scores_rating, review_scores_cleanliness,
    review_scores_location, review_scores_value
)

precio_pred = predecir_precio(
    ciudad, barrio, room_type, accommodates, bedrooms, bathrooms, beds,
    minimum_nights, availability_365, int(host_is_superhost),
    host_listings_count, int(instant_bookable),
    review_scores_rating, review_scores_cleanliness,
    review_scores_location, review_scores_value
)

# ── SECCIÓN 1: PREDICCIONES ───────────────────────────────────
st.header("🤖 Predicciones de los modelos de IA")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ocupación estimada", f"{ocupacion_pred*100:.1f}%",
              help="Predicción XGBoost basada en características del inmueble")
with col2:
    st.metric("Precio óptimo/noche", f"{precio_pred:.0f}€",
              help="Predicción Random Forest basada en el mercado")
with col3:
    ingresos_estimados = precio_pred * ocupacion_pred * availability_365
    st.metric("Ingresos brutos anuales", f"{ingresos_estimados:,.0f}€",
              help="Estimación basada en ocupación y precio predichos")

# ── SECCIÓN 2: ESTACIONALIDAD PROPHET ────────────────────────
st.header("📅 Proyección de estacionalidad (Prophet)")

forecast = forecast_madrid if ciudad == 'madrid' else forecast_barcelona
forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast_futuro = forecast[forecast['ds'] >= pd.Timestamp.now()].head(365)

fig_prophet = go.Figure()
fig_prophet.add_trace(go.Scatter(
    x=forecast_futuro['ds'], y=forecast_futuro['yhat'],
    mode='lines', name='Ocupación predicha',
    line=dict(color='steelblue', width=2)
))
fig_prophet.add_trace(go.Scatter(
    x=pd.concat([forecast_futuro['ds'], forecast_futuro['ds'][::-1]]),
    y=pd.concat([forecast_futuro['yhat_upper'], forecast_futuro['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(70,130,180,0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Intervalo de confianza'
))
fig_prophet.update_layout(
    title=f"Proyección de ocupación próximos 12 meses — {ciudad.capitalize()}",
    xaxis_title="Fecha", yaxis_title="Ocupación",
    height=350, showlegend=True
)
st.plotly_chart(fig_prophet, use_container_width=True)

# ── SECCIÓN 3: PARÁMETROS FINANCIEROS ────────────────────────
st.header("💰 Parámetros de inversión")

with st.expander("📐 Configurar inversión y financiación", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Inversión inicial")
        precio_compra         = st.number_input("Precio compra (€)", 50000, 2000000, 200000, step=5000)
        gastos_compraventa_pct = st.slider("Gastos compraventa (%)", 5.0, 15.0, 10.0, 0.5) / 100
        reforma               = st.number_input("Reforma (€)", 0, 100000, 5000, step=1000)
        mobiliario            = st.number_input("Mobiliario (€)", 0, 50000, 4000, step=500)
        licencias             = st.number_input("Licencias (€)", 0, 10000, 800, step=100)

    with col2:
        st.subheader("Financiación")
        pct_financiado  = st.slider("% financiado", 0, 80, 0) / 100
        tipo_interes    = st.slider("Tipo interés (%)", 1.0, 8.0, 3.5, 0.1) / 100
        plazo_anos      = st.slider("Plazo hipoteca (años)", 5, 30, 20)

        st.subheader("Ingresos")
        usar_precio_ia  = st.checkbox("Usar precio predicho por IA", value=True)
        precio_noche    = precio_pred if usar_precio_ia else st.number_input("Precio/noche (€)", 20, 1000, int(precio_pred))
        usar_ocup_ia    = st.checkbox("Usar ocupación predicha por IA", value=True)
        ocupacion_input = ocupacion_pred if usar_ocup_ia else st.slider("Ocupación (%)", 5, 100, int(ocupacion_pred*100)) / 100
        noches_disp     = st.number_input("Noches disponibles/año", 100, 365, availability_365)
        crec_ingresos   = st.slider("Crecimiento ingresos anual (%)", 0.0, 5.0, 2.0, 0.5) / 100

    with col3:
        st.subheader("Costes operativos")
        coste_limpieza  = st.number_input("Limpieza por rotación (€)", 20, 200, 50)
        rotaciones      = st.number_input("Rotaciones al año", 10, 200, int(ocupacion_input * noches_disp / 3))
        mantenimiento   = st.number_input("Mantenimiento anual (€)", 0, 10000, 800)
        comunidad       = st.number_input("Comunidad anual (€)", 0, 5000, 1200)
        seguro          = st.number_input("Seguro anual (€)", 0, 3000, 350)
        ibi             = st.number_input("IBI anual (€)", 0, 5000, 400)
        suministros     = st.number_input("Suministros anual (€)", 0, 5000, 600)
        comision_plat   = st.slider("Comisión plataforma (%)", 0.0, 20.0, 12.0, 0.5) / 100
        gestion_ext     = st.slider("Gestión externa (%)", 0.0, 25.0, 0.0, 0.5) / 100

        st.subheader("Fiscalidad y análisis")
        tipo_irpf       = st.slider("Tipo IRPF (%)", 19, 47, 24) / 100
        tasa_descuento  = st.slider("Tasa descuento (%)", 3.0, 10.0, 6.0, 0.5) / 100
        horizonte       = st.slider("Horizonte análisis (años)", 5, 20, 10)

# ── CÁLCULO FINANCIERO ────────────────────────────────────────
params = ParametrosInversion(
    precio_compra=precio_compra,
    gastos_compraventa_pct=gastos_compraventa_pct,
    reforma=reforma,
    mobiliario=mobiliario,
    licencias=licencias,
    pct_financiado=pct_financiado,
    tipo_interes=tipo_interes,
    plazo_anos=plazo_anos,
    precio_noche=precio_noche,
    ocupacion=ocupacion_input,
    noches_disponibles=noches_disp,
    crecimiento_ingresos=crec_ingresos,
    coste_limpieza_rotacion=coste_limpieza,
    rotaciones_año=rotaciones,
    mantenimiento_anual=mantenimiento,
    comunidad_anual=comunidad,
    seguro_anual=seguro,
    ibi_anual=ibi,
    suministros_anual=suministros,
    comision_plataforma_pct=comision_plat,
    gestion_externa_pct=gestion_ext,
    tipo_irpf=tipo_irpf,
    tasa_descuento=tasa_descuento,
    horizonte_anos=horizonte,
)

resultados = calcular_tres_escenarios(params)

# ── SECCIÓN 4: RESULTADOS ─────────────────────────────────────
st.header("📊 Resultados financieros")

colores = {'optimista': '#2ecc71', 'neutro': '#3498db', 'pesimista': '#e74c3c'}

tabs = st.tabs(["📈 Resumen", "💸 Flujos de caja", "🔍 Análisis de sensibilidad"])

with tabs[0]:
    for escenario in ['optimista', 'neutro', 'pesimista']:
        res = resultados[escenario]
        color = colores[escenario]
        st.subheader(f"Escenario {escenario.capitalize()}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Inversión total", f"{res['inversion_total']:,.0f}€")
        c2.metric("Capital propio", f"{res['capital_propio']:,.0f}€")
        c3.metric("VAN", f"{res['van']:,.0f}€")
        c4.metric("TIR", f"{res['tir']*100:.1f}%" if res['tir'] else "N/A")
        c5.metric("Rent. neta", f"{res['rentabilidad_neta']*100:.1f}%")
        c6.metric("Punto muerto", f"{res['punto_muerto_ocupacion']*100:.1f}%" if res['punto_muerto_ocupacion'] else "N/A")
        st.markdown("---")

with tabs[1]:
    fig_fcn = go.Figure()
    for escenario in ['optimista', 'neutro', 'pesimista']:
        res = resultados[escenario]
        años  = [f['año'] for f in res['flujos_caja']]
        fcns  = [f['fcn'] for f in res['flujos_caja']]
        fig_fcn.add_trace(go.Bar(
            name=escenario.capitalize(),
            x=años, y=fcns,
            marker_color=colores[escenario],
            opacity=0.8
        ))
    fig_fcn.update_layout(
        title="Flujo de caja neto por año y escenario",
        xaxis_title="Año", yaxis_title="FCN (€)",
        barmode='group', height=400
    )
    st.plotly_chart(fig_fcn, use_container_width=True)

    st.subheader("Tabla de flujos de caja — Escenario neutro")
    df_fcn = pd.DataFrame(resultados['neutro']['flujos_caja'])
    df_fcn.columns = ['Año', 'Ingresos brutos', 'Costes operativos',
                       'Costes financieros', 'Impuestos', 'FCN']
    st.dataframe(df_fcn.set_index('Año').style.format("{:,.0f}€"), use_container_width=True)

with tabs[2]:
    st.subheader("Sensibilidad del VAN ante variaciones en ocupación y precio")

    ocupaciones  = np.arange(0.10, 0.80, 0.05)
    vans = []
    for oc in ocupaciones:
        p_tmp = ParametrosInversion(**{**params.__dict__, 'ocupacion': oc})
        r_tmp = calcular_tres_escenarios(p_tmp)
        vans.append(r_tmp['neutro']['van'])

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=ocupaciones * 100, y=vans,
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        name='VAN'
    ))
    fig_sens.add_hline(y=0, line_dash='dash', line_color='red',
                        annotation_text='VAN = 0')
    fig_sens.update_layout(
        title="Sensibilidad del VAN ante variaciones en tasa de ocupación",
        xaxis_title="Tasa de ocupación (%)",
        yaxis_title="VAN (€)",
        height=400
    )
    st.plotly_chart(fig_sens, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Este simulador tiene finalidad académica y orientativa. No constituye asesoramiento financiero.")