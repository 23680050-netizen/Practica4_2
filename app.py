from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Regresión Lineal y Correlación", layout="wide")

# Paleta pastel para los mapas de color
PASTEL_DIV = sns.diverging_palette(220, 20, s=70, l=82, as_cmap=True)


# ---------------------------------------------------------------------------
# Utilidades de presentación (no son parte del análisis)
# ---------------------------------------------------------------------------
def codigo(nombre):
    """Lee este mismo archivo y devuelve el bloque marcado con '# >>> nombre'.
    Así el código que se muestra en la página es exactamente el que se ejecuta."""
    fuente = Path(__file__).read_text(encoding="utf-8")
    inicio, fin = f"# >>> {nombre}", f"# <<< {nombre}"
    return fuente.split(inicio)[1].split(fin)[0].strip("\n")


def ver_codigo(nombre):
    with st.expander("Ver el código usado en este paso"):
        st.code(codigo(nombre), language="python")


def tarjeta(titulo, valor, c1, c2):
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,{c1},{c2});padding:18px;
        border-radius:14px;color:#3F3A5A;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.12)">
        <div style="font-size:.95rem;opacity:.85">{titulo}</div>
        <div style="font-size:1.7rem;font-weight:700">{valor}</div></div>""",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .banner{background:linear-gradient(135deg,#CDB4F6,#F9C6E0,#FFE0B8);padding:28px 20px;
            border-radius:18px;text-align:center;box-shadow:0 3px 12px rgba(0,0,0,.12)}
    .banner h1{color:#4A3F6B;margin:0;font-size:2.2rem}
    .banner p{color:#5E5380;margin:6px 0 0 0;font-size:1.05rem}
    button[data-baseweb="tab"]{font-size:1.05rem;font-weight:600}
    button[data-baseweb="tab"][aria-selected="true"]{color:#7C6BB5}
    div[data-baseweb="tab-highlight"]{background-color:#B8A9E8}
    </style>
    <div class="banner">
        <h1>Regresión Lineal y Correlación</h1>
        <p>Práctica 4.2</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# PASO 1: SIMULACIÓN DE DATOS EN DUCKDB (REPRODUCIBLE)
# ---------------------------------------------------------------------------
# >>> paso1
con = duckdb.connect(database=':memory:')

# Fijar semilla determinista en DuckDB (setseed = 0.42)
con.execute("SELECT setseed(0.42);")

# Crear la tabla de telemetría de agua potable
con.execute("""
CREATE TABLE telemetria_agua AS
WITH base AS (
    SELECT
        residencia_id,
        ROUND(18 + (random() * 14), 2) AS temperatura_max,
        ROUND(50 + (random() * 40), 2) AS humedad_max,
        ROUND(3 + (random() * 4), 0) AS personas_familia,
        ROUND(0.2 + (random() * 1.8), 2) AS consumo_bidon_pc
    FROM generate_series(1, 100) AS t(residencia_id)
)
SELECT *,
    ROUND(94.0 + 2.5 * temperatura_max - 0.8 * humedad_max
               - 5.2 * personas_familia + 15.0 * consumo_bidon_pc
               + (random() * 6.0 - 3.0), 2) AS cpc_consumo
FROM base;
""")

# Extraer el dataset a un DataFrame de Pandas
df_agua = con.execute("SELECT * FROM telemetria_agua").df()
# <<< paso1

# ---------------------------------------------------------------------------
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON
# ---------------------------------------------------------------------------
# >>> paso2
matriz_correlacion = df_agua.drop(columns=['residencia_id']).corr(method='pearson')
corr_con_cpc = matriz_correlacion['cpc_consumo'].drop('cpc_consumo')
# <<< paso2

# ---------------------------------------------------------------------------
# PASO 3: REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ---------------------------------------------------------------------------
# >>> paso3
# Definir variables independientes (X) y dependiente (Y)
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']

# Agregar término constante (Beta_0) para el intercepto OLS
X_const = sm.add_constant(X)

# Ajustar el modelo OLS
modelo_ols = sm.OLS(Y, X_const).fit()
# <<< paso3

# ---------------------------------------------------------------------------
# PASO 4: VALIDACIÓN DE SUPUESTOS EN LOS RESIDUOS
# ---------------------------------------------------------------------------
# >>> paso4
residuos = modelo_ols.resid

# 1. Prueba de normalidad de Shapiro-Wilk sobre los residuos
stat_sw, p_val_sw = stats.shapiro(residuos)

# 2. Prueba de autocorrelación de Durbin-Watson (ideal cercano a 2.0)
dw_stat = durbin_watson(residuos)
# <<< paso4

# ---------------------------------------------------------------------------
# PASO 5: EVALUACIÓN PREDICTIVA CON SCIKIT-LEARN
# ---------------------------------------------------------------------------
# >>> paso5
model_sk = LinearRegression()
model_sk.fit(X, Y)

y_pred = model_sk.predict(X)

r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))
# <<< paso5

# ---------------------------------------------------------------------------
# PANEL DE GRÁFICAS
# ---------------------------------------------------------------------------
# >>> graficas
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)',
             fontsize=16, fontweight='bold', color='#6D5BA8')

# 1. Mapa de calor de correlación
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap=PASTEL_DIV, cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Dispersión: humedad vs consumo
sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='#93C5FD', ax=axes[0, 1],
            scatter_kws={'edgecolor': '#60A5FA'},
            line_kws={'color': '#F472B6', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

# 3. Valores reales vs predichos
axes[0, 2].scatter(y_pred, Y, alpha=0.8, color='#99F6E4', edgecolor='#2DD4BF')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], '--', color='#FB7185', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_sk * 100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

# 4. Residuos vs valores ajustados (homocedasticidad)
axes[1, 0].scatter(y_pred, residuos, color='#DDD6FE', edgecolor='#A78BFA', alpha=0.9)
axes[1, 0].axhline(y=0, color='#FB7185', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Gráfico Q-Q normal (normalidad de errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1],
          markerfacecolor='#FBCFE8', markeredgecolor='#F472B6')
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante
axes[1, 2].axis('off')

plt.tight_layout()
# <<< graficas

# ---------------------------------------------------------------------------
# PRESENTACIÓN POR PESTAÑAS
# ---------------------------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["1. Datos", "2. Correlación", "3. Modelo OLS", "4. Supuestos", "5. Métricas", "Gráficas"]
)

with t1:
    st.subheader("Datos simulados con DuckDB")
    st.write("Se generan **100 residencias** con semilla fija (`setseed(0.42)`), así el resultado es reproducible.")
    st.dataframe(
        df_agua.head().style.background_gradient(cmap="BuPu", subset=["cpc_consumo"], low=0, high=0.4),
        use_container_width=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta("Residencias", f"{len(df_agua)}", "#BFDBFE", "#A5F3FC")
    with c2:
        tarjeta("Variables predictoras", "4", "#DDD6FE", "#FBCFE8")
    with c3:
        tarjeta("CPC promedio", f"{df_agua['cpc_consumo'].mean():.1f} lt/hab-día", "#FED7AA", "#FECACA")
    st.write("")
    ver_codigo("paso1")

with t2:
    st.subheader("Matriz de correlación de Pearson (r)")
    st.dataframe(
        matriz_correlacion.round(4).style.background_gradient(cmap=PASTEL_DIV, vmin=-1, vmax=1),
        use_container_width=True,
    )
    st.write("**Correlación de cada variable con `cpc_consumo`:**")
    st.dataframe(corr_con_cpc.round(4).to_frame("r"), use_container_width=True)
    mas_fuerte = corr_con_cpc.abs().idxmax()
    st.info(f"La relación lineal más fuerte con el consumo es **{mas_fuerte}** (r = {corr_con_cpc[mas_fuerte]:.4f}).")
    ver_codigo("paso2")

with t3:
    st.subheader("Resumen del modelo OLS (statsmodels)")
    st.text(str(modelo_ols.summary()))
    st.write("**Coeficientes estimados:**")
    st.dataframe(modelo_ols.params.round(4).to_frame("coeficiente"), use_container_width=True)
    ver_codigo("paso3")

with t4:
    st.subheader("Diagnóstico de residuos y supuestos")
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta("Shapiro-Wilk (estadístico)", f"{stat_sw:.4f}", "#BBF7D0", "#A7F3D0")
    with c2:
        tarjeta("Valor p", f"{p_val_sw:.4f}", "#BAE6FD", "#C7D2FE")
    with c3:
        tarjeta("Durbin-Watson", f"{dw_stat:.4f}", "#F5D0FE", "#FECDD3")
    st.write("")
    if p_val_sw > 0.05:
        st.success("Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).")
    else:
        st.warning("Se rechaza la normalidad de los residuos (p <= 0.05).")
    if 1.5 <= dw_stat <= 2.5:
        st.success("Durbin-Watson entre 1.5 y 2.5: no hay autocorrelación en los residuos.")
    else:
        st.warning("Durbin-Watson fuera de 1.5 a 2.5: podría haber autocorrelación.")
    ver_codigo("paso4")

with t5:
    st.subheader("Métricas de rendimiento (scikit-learn)")
    c1, c2 = st.columns(2)
    with c1:
        tarjeta("Coeficiente de determinación (R²)", f"{r2_sk * 100:.2f}%", "#BBF7D0", "#D9F99D")
    with c2:
        tarjeta("RMSE", f"{rmse_sk:.4f} lt/hab-día", "#FED7AA", "#FEF08A")
    st.write("")
    st.write("**Ecuación del modelo ajustado:**")
    st.code(
        f"CPC = {model_sk.intercept_:.2f} + ({model_sk.coef_[0]:.2f} * Temp) + "
        f"({model_sk.coef_[1]:.2f} * Humedad) + ({model_sk.coef_[2]:.2f} * Personas) + "
        f"({model_sk.coef_[3]:.2f} * Bidon)"
    )
    ver_codigo("paso5")

with t6:
    st.subheader("Panel de diagnóstico visual")
    st.pyplot(fig)
    ver_codigo("graficas")
