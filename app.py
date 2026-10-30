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

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")
st.title("Práctica 4.2: Regresión Lineal y Correlación")
st.caption("Análisis y Exploración de Datos (INC-2505) - Predicción del Consumo Per Cápita de agua potable (CPC)")

# ---------------------------------------------------------------------------
# PASO 1: SIMULACIÓN DE DATOS EN DUCKDB (REPRODUCIBLE)
# ---------------------------------------------------------------------------
con = duckdb.connect(database=':memory:')
con.execute("SELECT setseed(0.42);")
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
df_agua = con.execute("SELECT * FROM telemetria_agua").df()

st.header("Paso 1: Primeros 5 registros obtenidos de DuckDB")
st.dataframe(df_agua.head())

# ---------------------------------------------------------------------------
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON
# ---------------------------------------------------------------------------
matriz_correlacion = df_agua.drop(columns=['residencia_id']).corr(method='pearson')

st.header("Paso 2: Matriz de correlación de Pearson (r)")
st.dataframe(matriz_correlacion.round(4))
st.write("Correlación lineal de cada variable con la respuesta `cpc_consumo`:")
st.dataframe(matriz_correlacion['cpc_consumo'].drop('cpc_consumo').round(4))

# ---------------------------------------------------------------------------
# PASO 3: REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ---------------------------------------------------------------------------
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

st.header("Paso 3: Resumen del modelo OLS (statsmodels)")
st.text(str(modelo_ols.summary()))

# ---------------------------------------------------------------------------
# PASO 4: VALIDACIÓN DE SUPUESTOS EN LOS RESIDUOS
# ---------------------------------------------------------------------------
residuos = modelo_ols.resid
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)

st.header("Paso 4: Diagnóstico de residuos y supuestos")
st.write(f"1. Prueba de normalidad (Shapiro-Wilk): estadístico = {stat_sw:.4f}, valor p = {p_val_sw:.4f}")
if p_val_sw > 0.05:
    st.success("Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).")
else:
    st.warning("Se rechaza la normalidad de los residuos (p <= 0.05).")
st.write(f"2. Durbin-Watson (independencia): {dw_stat:.4f} (valores entre 1.5 y 2.5 indican ausencia de autocorrelación).")

# ---------------------------------------------------------------------------
# PASO 5: EVALUACIÓN PREDICTIVA CON SCIKIT-LEARN
# ---------------------------------------------------------------------------
model_sk = LinearRegression()
model_sk.fit(X, Y)
y_pred = model_sk.predict(X)
r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))

st.header("Paso 5: Métricas de rendimiento (scikit-learn)")
col1, col2 = st.columns(2)
col1.metric("R²", f"{r2_sk * 100:.2f}%")
col2.metric("RMSE", f"{rmse_sk:.4f} lt/hab-día")
st.write("Ecuación del modelo ajustado:")
st.code(
    f"CPC = {model_sk.intercept_:.2f} + ({model_sk.coef_[0]:.2f} * Temp) + "
    f"({model_sk.coef_[1]:.2f} * Humedad) + ({model_sk.coef_[2]:.2f} * Personas) + "
    f"({model_sk.coef_[3]:.2f} * Bidon)"
)

# ---------------------------------------------------------------------------
# PANEL DE GRÁFICAS
# ---------------------------------------------------------------------------
st.header("Panel de diagnóstico visual")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=16, fontweight='bold')

sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1],
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_sk * 100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

axes[1, 2].axis('off')
plt.tight_layout()
st.pyplot(fig)
