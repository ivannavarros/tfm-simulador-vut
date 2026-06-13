# ============================================================
# simulador.py
# Simulador interactivo de inversión en VUT — Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import os
import sys
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE)

from src.simulation.motor_financiero import ParametrosInversion, calcular_tres_escenarios, calcular_motor

st.set_page_config(
    page_title="Simulador VUT",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CARGA DE MODELOS ─────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    MODELS    = os.path.join(BASE, 'src', 'models')
    PROCESSED = os.path.join(BASE, 'data', 'processed')

    with open(os.path.join(MODELS, 'xgboost_ocupacion.pkl'), 'rb') as f:
        modelo_xgb = pickle.load(f)
    with open(os.path.join(MODELS, 'encoders_ocupacion.pkl'), 'rb') as f:
        encoders_xgb = pickle.load(f)

    with gzip.open(os.path.join(MODELS, 'rf_precio.pkl.gz'), 'rb') as f:
        modelo_rf = pickle.load(f)
    with open(os.path.join(MODELS, 'encoders_rf_precio.pkl'), 'rb') as f:
        encoders_rf = pickle.load(f)

    # Cargamos forecast Prophet para las 9 ciudades
    ciudades_prophet = [
        'barcelona', 'euskadi', 'girona', 'madrid', 'malaga',
        'mallorca', 'menorca', 'sevilla', 'valencia'
    ]
    forecasts = {}
    for ciudad in ciudades_prophet:
        ruta = os.path.join(MODELS, f'prophet_forecast_{ciudad}.pkl')
        with open(ruta, 'rb') as f:
            forecasts[ciudad] = pickle.load(f)

    with open(os.path.join(MODELS, 'shap_xgb.pkl'), 'rb') as f:
        shap_data = pickle.load(f)

    df = pd.read_csv(os.path.join(PROCESSED, 'listings_clean.csv'), low_memory=False)

    return modelo_xgb, encoders_xgb, modelo_rf, encoders_rf, forecasts, shap_data, df

modelo_xgb, encoders_xgb, modelo_rf, encoders_rf, forecasts, shap_data, df = cargar_modelos()

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
        'ciudad_enc': ciudad_enc, 'barrio_enc': barrio_enc, 'room_enc': room_enc,
        'accommodates': accommodates, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'beds': beds, 'price': price, 'minimum_nights': minimum_nights,
        'availability_365': availability_365,
        'host_is_superhost': host_is_superhost,
        'host_listings_count': host_listings_count,
        'instant_bookable': instant_bookable,
        'review_scores_rating': review_scores_rating,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_location': review_scores_location,
        'review_scores_value': review_scores_value,
    }])
    return float(np.clip(modelo_xgb.predict(X)[0], 0, 1)), X


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
        'ciudad_enc': ciudad_enc, 'barrio_enc': barrio_enc, 'room_enc': room_enc,
        'accommodates': accommodates, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'beds': beds, 'minimum_nights': minimum_nights,
        'availability_365': availability_365,
        'host_is_superhost': host_is_superhost,
        'host_listings_count': host_listings_count,
        'instant_bookable': instant_bookable,
        'review_scores_rating': review_scores_rating,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_location': review_scores_location,
        'review_scores_value': review_scores_value,
    }])
    return float(np.clip(modelo_rf.predict(X)[0], 20, 2000))


def generar_informe(ciudad, barrio, resultados, ocupacion_pred, precio_pred,
                    precio_compra, pct_financiado, horizonte):

    res_opt = resultados['optimista']
    res_neu = resultados['neutro']
    res_pes = resultados['pesimista']

    van_neutro = res_neu['van']
    tir_neutro = res_neu['tir']
    pm_neutro  = res_neu['punto_muerto_ocupacion']
    rent_neta  = res_neu['rentabilidad_neta']

    if van_neutro > 0:
        viabilidad = "VIABLE"
        color_viab = "success"
        texto_viab = "La inversión genera valor por encima del coste de oportunidad del capital en el escenario neutro."
    elif res_opt['van'] > 0:
        viabilidad = "CONDICIONALMENTE VIABLE"
        color_viab = "warning"
        texto_viab = "La inversión es viable en el escenario optimista pero no en el neutro. Requiere que las condiciones de mercado sean favorables."
    else:
        viabilidad = "NO VIABLE EN EL HORIZONTE ANALIZADO"
        color_viab = "error"
        texto_viab = "La inversión no recupera el capital invertido en ninguno de los escenarios analizados dentro del horizonte temporal definido."

    if ocupacion_pred >= 0.60:
        texto_ocup = f"La ocupación estimada por el modelo XGBoost es del {ocupacion_pred*100:.1f}%, un nivel alto que favorece la rentabilidad del activo."
    elif ocupacion_pred >= 0.40:
        texto_ocup = f"La ocupación estimada por el modelo XGBoost es del {ocupacion_pred*100:.1f}%, un nivel moderado coherente con la media del mercado de VUT en {ciudad.capitalize()}."
    else:
        texto_ocup = f"La ocupación estimada por el modelo XGBoost es del {ocupacion_pred*100:.1f}%, un nivel bajo que limita significativamente la generación de ingresos."

    if pm_neutro:
        margen = ocupacion_pred - pm_neutro
        if margen > 0.15:
            texto_pm = f"El punto muerto de ocupación se sitúa en el {pm_neutro*100:.1f}%, lo que supone un margen de seguridad de {margen*100:.1f} puntos porcentuales sobre la ocupación estimada. El proyecto presenta una resistencia sólida ante caídas de demanda."
        elif margen > 0:
            texto_pm = f"El punto muerto de ocupación se sitúa en el {pm_neutro*100:.1f}%, próxima a la ocupación estimada ({ocupacion_pred*100:.1f}%). El margen de seguridad es reducido y el proyecto es sensible a variaciones en la demanda."
        else:
            texto_pm = f"El punto muerto de ocupación ({pm_neutro*100:.1f}%) supera la ocupación estimada ({ocupacion_pred*100:.1f}%). El proyecto no cubre sus costes totales con el nivel de demanda previsto."
    else:
        texto_pm = ""

    if pct_financiado == 0:
        texto_fin = "La operación se financia íntegramente con capital propio, eliminando el riesgo de tipo de interés y mejorando la rentabilidad neta al no incurrir en costes financieros."
    elif pct_financiado <= 0.50:
        texto_fin = f"La operación se financia con un {pct_financiado*100:.0f}% de capital ajeno, lo que introduce un nivel moderado de apalancamiento financiero. La cuota hipotecaria mensual es de {res_neu['cuota_mensual']:,.0f} EUR."
    else:
        texto_fin = f"La operación presenta un nivel de apalancamiento elevado ({pct_financiado*100:.0f}% financiado). La cuota hipotecaria mensual de {res_neu['cuota_mensual']:,.0f} EUR supone una carga financiera significativa que condiciona el flujo de caja neto."

    texto_escenarios = f"El VAN oscila entre {res_pes['van']:,.0f} EUR (escenario pesimista) y {res_opt['van']:,.0f} EUR (escenario optimista), con un valor central de {res_neu['van']:,.0f} EUR en el escenario neutro. "
    if tir_neutro:
        texto_escenarios += f"La TIR del escenario neutro es del {tir_neutro*100:.1f}%."

    informe = f"""
**INFORME EJECUTIVO DE VIABILIDAD — {ciudad.upper()} / {barrio.upper()}**

**Precio de adquisición:** {precio_compra:,.0f} EUR | **Inversión total:** {res_neu['inversion_total']:,.0f} EUR | **Horizonte:** {horizonte} años

---

**VEREDICTO: {viabilidad}**

{texto_viab}

---

**Análisis de ocupación y precio**

{texto_ocup} El precio óptimo estimado por el modelo Random Forest para este inmueble en {barrio} es de {precio_pred:.0f} EUR por noche, posicionándolo en el segmento {'premium' if precio_pred > 150 else 'medio-alto' if precio_pred > 100 else 'económico'} del mercado local.

**Punto muerto de ocupación**

{texto_pm}

**Estructura de financiación**

{texto_fin}

**Comparativa de escenarios**

{texto_escenarios}

**Rentabilidad neta (escenario neutro):** {rent_neta*100:.1f}% | **ROE:** {res_neu['roe']*100:.1f}%

---

*Informe generado automáticamente por el Simulador VUT. Tiene carácter orientativo y no constituye asesoramiento financiero.*
"""
    return informe, color_viab


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Parámetros del inmueble")
    ciudad = st.selectbox("Ciudad", [
        'barcelona', 'euskadi', 'girona', 'madrid', 'malaga',
        'mallorca', 'menorca', 'sevilla', 'valencia'
    ], index=3, format_func=lambda x: x.capitalize())

    barrios_disponibles = sorted(df[df['ciudad'] == ciudad]['neighbourhood_cleansed'].unique())
    barrio = st.selectbox("Barrio", barrios_disponibles)

    room_type = st.selectbox("Tipo de alojamiento", [
        "Entire home/apt", "Private room", "Shared room", "Hotel room"
    ])

    col1, col2 = st.columns(2)
    with col1:
        accommodates = st.number_input("Huéspedes", 1, 16, 4)
        bedrooms     = st.number_input("Habitaciones", 0, 10, 2)
        bathrooms    = st.number_input("Baños", 0.0, 10.0, 1.0, step=0.5)
    with col2:
        beds             = st.number_input("Camas", 1, 16, 2)
        minimum_nights   = st.number_input("Noches mínimas", 1, 30, 2)
        availability_365 = st.number_input("Días disponibles al año", 30, 365, 300)

    st.subheader("⭐ Puntuaciones esperadas")
    review_scores_rating      = st.slider("Valoración general", 1.0, 5.0, 4.5, 0.1)
    review_scores_cleanliness = st.slider("Limpieza", 1.0, 5.0, 4.5, 0.1)
    review_scores_location    = st.slider("Ubicación", 1.0, 5.0, 4.5, 0.1)
    review_scores_value       = st.slider("Calidad-precio", 1.0, 5.0, 4.3, 0.1)

    st.subheader("👤 Perfil del anfitrión")
    host_is_superhost   = st.checkbox("Superhost", value=False)
    host_listings_count = st.number_input("Anuncios del host", 1, 100, 1)
    instant_bookable    = st.checkbox("Reserva instantánea", value=True)


# ── PREDICCIONES ─────────────────────────────────────────────
ocupacion_pred, X_usuario = predecir_ocupacion(
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

# ── TÍTULO ───────────────────────────────────────────────────
st.title("🏠 Simulador de Inversión en Viviendas de Uso Turístico")
st.caption("TFM — Máster en IA aplicada a Entornos Empresariales y Financieros | UNIA 2026")
st.markdown("---")

# ── SECCIÓN 1: PREDICCIONES IA ───────────────────────────────
st.header("🤖 Predicciones de los modelos de IA")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ocupación estimada", f"{ocupacion_pred*100:.1f}%",
              help="Predicción XGBoost basada en características del inmueble")
with col2:
    st.metric("Precio óptimo por noche", f"{precio_pred:.0f} EUR",
              help="Predicción Random Forest basada en el mercado")
with col3:
    ingresos_estimados = precio_pred * ocupacion_pred * availability_365
    st.metric("Ingresos brutos anuales", f"{ingresos_estimados:,.0f} EUR",
              help="Estimación basada en ocupación y precio predichos")

st.markdown("---")

# ── SECCIÓN 2: PROPHET ───────────────────────────────────────
st.header("📅 Proyección de estacionalidad (Prophet)")

forecast = forecasts[ciudad].copy()
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
    xaxis_title="Fecha",
    yaxis_title="Ocupación",
    yaxis=dict(range=[0, 1.0]),
    height=350
)
st.plotly_chart(fig_prophet, use_container_width=True)
st.markdown("---")

# ── SECCIÓN 3: PARÁMETROS FINANCIEROS ────────────────────────
st.header("💰 Parámetros de inversión")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Inversión inicial")
    precio_compra          = st.number_input("Precio compra (EUR)", 50000, 2000000, 200000, step=5000)
    gastos_compraventa_pct = st.slider("Gastos compraventa (%)", 5.0, 15.0, 10.0, 0.5) / 100
    reforma                = st.number_input("Reforma (EUR)", 0, 100000, 5000, step=1000)
    mobiliario             = st.number_input("Mobiliario (EUR)", 0, 50000, 4000, step=500)
    licencias              = st.number_input("Licencias (EUR)", 0, 10000, 800, step=100)

with col2:
    st.subheader("Financiación")
    pct_financiado = st.slider("Porcentaje financiado", 0, 80, 0) / 100
    tipo_interes   = st.slider("Tipo interés (%)", 1.0, 8.0, 3.5, 0.1) / 100
    plazo_anos     = st.slider("Plazo hipoteca (años)", 5, 30, 20)

    st.subheader("Ingresos")
    usar_precio_ia  = st.checkbox("Usar precio predicho por IA", value=True)
    precio_noche    = precio_pred if usar_precio_ia else st.number_input(
        "Precio noche (EUR)", 20, 1000, int(precio_pred))
    usar_ocup_ia    = st.checkbox("Usar ocupación predicha por IA", value=True)
    ocupacion_input = ocupacion_pred if usar_ocup_ia else st.slider(
        "Ocupación (%)", 5, 100, int(ocupacion_pred * 100)) / 100
    noches_disp     = st.number_input("Noches disponibles al año", 100, 365, availability_365)
    crec_ingresos   = st.slider("Crecimiento ingresos anual (%)", 0.0, 5.0, 2.0, 0.5) / 100

with col3:
    st.subheader("Costes operativos")
    coste_limpieza = st.number_input("Limpieza por rotación (EUR)", 20, 200, 50)
    rotaciones     = st.number_input("Rotaciones al año", 10, 200,
                                      max(10, int(ocupacion_pred * availability_365 / 3)))
    mantenimiento  = st.number_input("Mantenimiento anual (EUR)", 0, 10000, 800)
    comunidad      = st.number_input("Comunidad anual (EUR)", 0, 5000, 1200)
    seguro         = st.number_input("Seguro anual (EUR)", 0, 3000, 350)
    ibi            = st.number_input("IBI anual (EUR)", 0, 5000, 400)
    suministros    = st.number_input("Suministros anual (EUR)", 0, 5000, 600)
    comision_plat  = st.slider("Comisión plataforma (%)", 0.0, 20.0, 12.0, 0.5) / 100
    gestion_ext    = st.slider("Gestión externa (%)", 0.0, 25.0, 0.0, 0.5) / 100

    st.subheader("Fiscalidad y análisis")
    tipo_irpf      = st.slider("Tipo IRPF (%)", 19, 47, 24) / 100
    tasa_descuento = st.slider("Tasa descuento (%)", 3.0, 10.0, 6.0, 0.5) / 100
    horizonte      = st.slider("Horizonte análisis (años)", 5, 20, 10)

st.markdown("---")

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

for escenario in ['optimista', 'neutro', 'pesimista']:
    res = resultados[escenario]
    st.subheader(f"Escenario {escenario.capitalize()}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Inversión total", f"{res['inversion_total']:,.0f} EUR")
    c2.metric("Capital propio", f"{res['capital_propio']:,.0f} EUR")
    c3.metric("VAN (EUR)", f"{res['van']:,.0f}")
    c4.metric("TIR", f"{res['tir']*100:.1f}%" if res['tir'] else "N/A")
    c5.metric("Rentabilidad neta", f"{res['rentabilidad_neta']*100:.1f}%")
    c6.metric("Punto muerto ocupación",
              f"{res['punto_muerto_ocupacion']*100:.1f}%"
              if res['punto_muerto_ocupacion'] else "N/A")
    st.markdown("---")

# ── SECCIÓN 5: FLUJOS DE CAJA ─────────────────────────────────
st.header("💸 Flujos de caja proyectados")

fig_fcn = go.Figure()
for escenario in ['optimista', 'neutro', 'pesimista']:
    res  = resultados[escenario]
    años = [f['año'] for f in res['flujos_caja']]
    fcns = [f['fcn'] for f in res['flujos_caja']]
    fig_fcn.add_trace(go.Bar(
        name=escenario.capitalize(),
        x=años, y=fcns,
        marker_color=colores[escenario],
        opacity=0.8
    ))
fig_fcn.update_layout(
    title="Flujo de caja neto por año y escenario",
    xaxis_title="Año", yaxis_title="FCN (EUR)",
    barmode='group', height=400
)
st.plotly_chart(fig_fcn, use_container_width=True)

st.subheader("Tabla de flujos de caja — Escenario neutro")
df_fcn = pd.DataFrame(resultados['neutro']['flujos_caja'])
df_fcn.columns = ['Año', 'Ingresos brutos', 'Costes operativos',
                   'Costes financieros', 'Impuestos', 'FCN']
st.dataframe(
    df_fcn.set_index('Año').style.format("{:,.0f} €"),
    use_container_width=True
)
st.markdown("---")

# ── SECCIÓN 6: ANÁLISIS DE SENSIBILIDAD ──────────────────────
st.header("🔍 Análisis de sensibilidad del VAN")

ocupaciones = np.arange(0.10, 0.80, 0.05)
vans = []
for oc in ocupaciones:
    params_tmp = ParametrosInversion(
        precio_compra=precio_compra,
        gastos_compraventa_pct=gastos_compraventa_pct,
        reforma=reforma, mobiliario=mobiliario, licencias=licencias,
        pct_financiado=pct_financiado, tipo_interes=tipo_interes,
        plazo_anos=plazo_anos, precio_noche=precio_noche,
        ocupacion=oc, noches_disponibles=noches_disp,
        crecimiento_ingresos=crec_ingresos,
        coste_limpieza_rotacion=coste_limpieza, rotaciones_año=rotaciones,
        mantenimiento_anual=mantenimiento, comunidad_anual=comunidad,
        seguro_anual=seguro, ibi_anual=ibi, suministros_anual=suministros,
        comision_plataforma_pct=comision_plat, gestion_externa_pct=gestion_ext,
        tipo_irpf=tipo_irpf, tasa_descuento=tasa_descuento, horizonte_anos=horizonte,
    )
    r_tmp = calcular_motor(params_tmp, 'neutro')
    vans.append(r_tmp['van'])

fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=list(ocupaciones * 100), y=vans,
    mode='lines+markers',
    line=dict(color='steelblue', width=2),
    name='VAN'
))
fig_sens.add_hline(y=0, line_dash='dash', line_color='red',
                   annotation_text='VAN = 0')
fig_sens.update_layout(
    title="Sensibilidad del VAN ante variaciones en tasa de ocupación",
    xaxis_title="Tasa de ocupación (%)",
    yaxis_title="VAN (EUR)",
    height=400
)
st.plotly_chart(fig_sens, use_container_width=True)
st.markdown("---")

# ── SECCIÓN 7: SHAP ───────────────────────────────────────────
st.header("🔬 Interpretabilidad del modelo (SHAP)")
st.markdown("Factores que más influyen en la ocupación predicha para este inmueble concreto.")

feature_names_legibles = [
    'Ciudad', 'Barrio', 'Tipo alojamiento',
    'Huéspedes', 'Habitaciones', 'Baños', 'Camas',
    'Precio/noche', 'Noches mínimas', 'Disponibilidad anual',
    'Superhost', 'Anuncios host', 'Reserva instantánea',
    'Valoración general', 'Limpieza', 'Ubicación', 'Calidad-precio'
]

explainer_xgb = shap_data['explainer_xgb']
shap_values_usuario = explainer_xgb.shap_values(X_usuario)

shap_df = pd.DataFrame({
    'Variable': feature_names_legibles,
    'Contribución SHAP': shap_values_usuario[0],
    'Valor': X_usuario.values[0]
}).sort_values('Contribución SHAP', key=abs, ascending=False).head(10)

colores_shap = ['#2ecc71' if v > 0 else '#e74c3c'
                for v in shap_df['Contribución SHAP']]

fig_shap = go.Figure(go.Bar(
    x=shap_df['Contribución SHAP'],
    y=shap_df['Variable'],
    orientation='h',
    marker_color=colores_shap,
))
fig_shap.update_layout(
    title="Contribución de cada variable a la ocupación predicha (SHAP)",
    xaxis_title="Contribución SHAP (positivo = aumenta ocupación)",
    yaxis_title="",
    height=450,
    yaxis=dict(autorange='reversed')
)
st.plotly_chart(fig_shap, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Factores que aumentan la ocupación")
    positivos = shap_df[shap_df['Contribución SHAP'] > 0]
    for _, row in positivos.iterrows():
        st.success(f"**{row['Variable']}** +{row['Contribución SHAP']:.4f}")

with col2:
    st.subheader("Factores que reducen la ocupación")
    negativos = shap_df[shap_df['Contribución SHAP'] < 0]
    for _, row in negativos.iterrows():
        st.error(f"**{row['Variable']}** {row['Contribución SHAP']:.4f}")

st.markdown("---")

# ── SECCIÓN 8: INFORME AUTOMATIZADO ──────────────────────────
st.header("📄 Informe ejecutivo automatizado")
st.markdown("Pulse el botón para generar un informe ejecutivo personalizado.")

if st.button("Generar informe ejecutivo", type="primary"):
    with st.spinner("Generando informe..."):
        informe, color_viab = generar_informe(
            ciudad, barrio, resultados, ocupacion_pred, precio_pred,
            precio_compra, pct_financiado, horizonte
        )
    if color_viab == "success":
        st.success(informe)
    elif color_viab == "warning":
        st.warning(informe)
    else:
        st.error(informe)

st.markdown("---")
st.caption("Este simulador tiene finalidad académica y orientativa. No constituye asesoramiento financiero.")