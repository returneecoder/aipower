import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Grid Predictor",
    page_icon="⚡",
    layout="wide"
)

# ── Load model and data ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model('model/xgb_model.json')
    return model

@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.pkl') 
@st.cache_data
def load_data():
    return pd.read_csv('data/processed.csv')

model = load_model()
df    = load_data()

FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin',
    'is_weekend', 'is_peak', 'lag_1h', 'lag_24h', 'lag_168h',
    'roll_mean_6h', 'roll_mean_24h', 'roll_std_24h'
]

# ── Header ────────────────────────────────────────────────────────────────
st.title("⚡ AI Smart Grid Energy Predictor")
st.markdown("Predicting household energy consumption 24 hours ahead using XGBoost + LSTM.")

# ── Sidebar controls ──────────────────────────────────────────────────────
st.sidebar.header("Controls")

day_options = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
selected_day = st.sidebar.selectbox("Day of week", day_options)
selected_hour = st.sidebar.slider("Starting hour", 0, 23, 0)
selected_month = st.sidebar.slider("Month", 1, 12, 6)
is_weekend = 1 if selected_day in ['Saturday','Sunday'] else 0
temperature = st.sidebar.slider("Temperature (°C)", 15, 45, 32)
show_confidence = st.sidebar.checkbox("Show prediction range", value=True)

# ── Metric cards ──────────────────────────────────────────────────────────
split = int(len(df) * 0.8)
X_test = df[FEATURES].iloc[split:].values
y_test = df['usage_kw'].iloc[split:].values
test_preds = model.predict(X_test)
test_mae  = mean_absolute_error(y_test, test_preds)
test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model MAE",    f"{test_mae:.4f} kW")
col2.metric("MAPE",         f"{test_mape:.1f}%")
col3.metric("Training rows", f"{split:,}")
col4.metric("Test rows",    f"{len(X_test):,}")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 24-hr Forecast", "📊 Model performance", "🔍 Data explorer"])

# ── Tab 1 — 24-hr forecast ────────────────────────────────────────────────
with tab1:
    st.subheader("24-hour energy consumption forecast")

    hours = list(range(24))
    dow = day_options.index(selected_day)

    # Build feature rows for each of the next 24 hours
    rows = []
    for h in hours:
        hour_abs = (selected_hour + h) % 24
        row = {
            'hour_sin':      np.sin(2 * np.pi * hour_abs / 24),
            'hour_cos':      np.cos(2 * np.pi * hour_abs / 24),
            'dow_sin':       np.sin(2 * np.pi * dow / 7),
            'dow_cos':       np.cos(2 * np.pi * dow / 7),
            'month_sin':     np.sin(2 * np.pi * selected_month / 12),
            'is_weekend':    is_weekend,
            'is_peak':       int(17 <= hour_abs <= 21),
            'lag_1h':        df['usage_kw'].iloc[-1],
            'lag_24h':       df['usage_kw'].iloc[-24] if len(df) >= 24 else 1.0,
            'lag_168h':      df['usage_kw'].iloc[-168] if len(df) >= 168 else 1.0,
            'roll_mean_6h':  df['usage_kw'].iloc[-6:].mean(),
            'roll_mean_24h': df['usage_kw'].iloc[-24:].mean(),
            'roll_std_24h':  df['usage_kw'].iloc[-24:].std(),
        }
        rows.append(row)

    X_forecast = pd.DataFrame(rows)[FEATURES].values
    forecast   = model.predict(X_forecast)

    # Temperature effect
    temp_factor = 1 + (temperature - 20) * 0.008
    forecast = forecast * temp_factor * (0.78 if is_weekend else 1.0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    hour_labels = [(selected_hour + h) % 24 for h in hours]

    ax.plot(hours, forecast, color='#2a78d6', linewidth=2.5,
            marker='o', markersize=4, label='Predicted')

    if show_confidence:
        margin = forecast * 0.08
        ax.fill_between(hours, forecast - margin, forecast + margin,
                        alpha=0.15, color='#2a78d6', label='±8% range')

    # Shade peak hours
    peak_hours = [h for h in range(24)
                  if 17 <= (selected_hour + h) % 24 <= 21]
    if peak_hours:
        ax.axvspan(min(peak_hours), max(peak_hours),
                   alpha=0.08, color='#e34948', label='Peak hours')

    ax.set_xlabel('Hours from now')
    ax.set_ylabel('Predicted usage (kW)')
    ax.set_title(f'24-hr forecast — {selected_day}, starting {selected_hour:02d}:00')
    ax.set_xticks(hours[::2])
    ax.set_xticklabels([f'{l:02d}:00' for l in hour_labels[::2]], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary stats below chart
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak predicted",   f"{forecast.max():.3f} kW",
              f"at {hour_labels[forecast.argmax()]:02d}:00")
    c2.metric("Min predicted",    f"{forecast.min():.3f} kW",
              f"at {hour_labels[forecast.argmin()]:02d}:00")
    c3.metric("24-hr total",      f"{forecast.sum():.2f} kWh")

# ── Tab 2 — Model performance ─────────────────────────────────────────────
with tab2:
    st.subheader("Model performance on test data")

    n_show = st.slider("Hours to display", 24, 336, 168, step=24)

    actual = y_test[-n_show:]
    preds  = test_preds[-n_show:]

    fig2, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Actual vs predicted
    axes[0].plot(actual, color='#888780', linewidth=1.5,
                 linestyle='--', label='Actual')
    axes[0].plot(preds,  color='#2a78d6', linewidth=1.5,
                 label='XGBoost predicted')
    axes[0].set_title('Actual vs predicted energy usage')
    axes[0].set_ylabel('kW')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = actual - preds
    axes[1].hist(errors, bins=40, color='#2a78d6', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='#e34948', linewidth=1.5, linestyle='--',
                    label='Zero error')
    axes[1].axvline(errors.mean(), color='#1baf7a', linewidth=1.5,
                    label=f'Mean error: {errors.mean():.4f}')
    axes[1].set_title('Error distribution (actual − predicted)')
    axes[1].set_xlabel('Error (kW)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Feature importance
    st.subheader("Feature importance")
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    colors = ['#2a78d6' if v == fi.max() else '#b5d4f4' for v in fi]
    fi.plot(kind='barh', ax=ax3, color=colors)
    ax3.set_title('XGBoost feature importance')
    ax3.set_xlabel('Importance score')
    ax3.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Tab 3 — Data explorer ─────────────────────────────────────────────────
with tab3:
    st.subheader("Explore the dataset")

    col_a, col_b = st.columns(2)
    with col_a:
        x_col = st.selectbox("X axis", FEATURES, index=0)
    with col_b:
        y_col = st.selectbox("Y axis", ['usage_kw'] + FEATURES, index=0)

    sample = df.sample(min(500, len(df)), random_state=42)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.scatter(sample[x_col], sample[y_col],
                alpha=0.4, color='#2a78d6', s=15)
    ax4.set_xlabel(x_col)
    ax4.set_ylabel(y_col)
    ax4.set_title(f'{x_col} vs {y_col}')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.dataframe(df.describe().round(4), use_container_width=True)
