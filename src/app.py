import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet import Prophet

HOME_AIRPORTS = ('LGW', 'LIS', 'LYS', 'NTE')
PAIRED_AIRPORTS = ('FUE', 'AMS', 'ORY')

df = pd.read_parquet('data/traffic_10lines.parquet')

st.title('Traffic Forecaster')

# Variable de drapeau pour vérifier si le bouton "Forecast" a été cliqué
forecast_button_clicked = False

def calculate_performance(actual, predicted):
    mae = abs(actual - predicted).mean()
    rmse = ((actual - predicted) ** 2).mean() ** 0.5
    r_squared = 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    return mae, rmse, r_squared

# Créer une instance du modèle Prophet
model = Prophet()

def run_forecast(df_filtered, home_airport, paired_airport, forecast_date, nb_days):
    # Convertir forecast_date en datetime64[ns]
    forecast_date = pd.to_datetime(forecast_date)

    # Filtrer les données historiques jusqu'à la date de prévision
    df_filtered = df_filtered[df_filtered['date'] <= forecast_date]

    # Préparer les données pour Prophet
    df_prophet = df_filtered[['date', 'pax_total']].rename(columns={'date': 'ds', 'pax_total': 'y'})

    # Entraîner le modèle
    model.fit(df_prophet)

    # Générer les dates de prévision
    forecast_dates = pd.date_range(forecast_date, periods=nb_days)

    # Prédire le trafic pour les dates de prévision
    future = pd.DataFrame({'ds': forecast_dates})
    forecast = model.predict(future)

    actual_values = df_filtered['pax_total'].values[-nb_days:]
    predicted_values = forecast['yhat'].values
    mae, rmse, r_squared = calculate_performance(actual_values, predicted_values)

    # Retourner les performances du modèle
    return mae, rmse, r_squared


with st.sidebar:
    home_airport = st.selectbox('Home Airport', HOME_AIRPORTS)
    paired_airport = st.selectbox('Paired Airport', PAIRED_AIRPORTS)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast_button = st.button('Forecast')

# Affichage de la table
st.write('Home Airport selected:', home_airport)
st.write('Paired Airport selected:', paired_airport)
st.write('Days of forecast:', nb_days)
st.write('Forecast Start Date:', forecast_date)

# Filtrer les données pour le graphique initial
df_filtered_initial = df.query(
    f'home_airport == "{home_airport}" and paired_airport == "{paired_airport}"'
)
df_filtered_initial = df_filtered_initial.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()

# Créer un graphique avec les données initiales
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df_filtered_initial['date'], y=df_filtered_initial['pax_total'], fill='tozeroy',
                         name=f'{home_airport} → {paired_airport}'), row=1, col=1)

if run_forecast_button:
    forecast_button_clicked = True  # Mettre à jour le drapeau lorsque le bouton est cliqué

    df_filtered = df.query(
        f'home_airport == "{home_airport}" and paired_airport == "{paired_airport}"'
    )
    df_filtered = df_filtered.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()

    mae, rmse, r_squared = run_forecast(df_filtered, home_airport, paired_airport, forecast_date, nb_days)

    # Prédire le trafic pour les dates de prévision
    forecast_dates = pd.date_range(forecast_date, periods=nb_days)
    future = pd.DataFrame({'ds': forecast_dates})
    forecast = model.predict(future)

    # Ajouter les données prédictives au graphique
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dash', color='red'),
                             name='Forecast'), row=1, col=1)
st.plotly_chart(fig)

# Afficher les performances du modèle
if forecast_button_clicked:
    st.write('Performance Metrics:')
    st.write(f'- MAE: {mae:.2f}')
    st.write(f'- RMSE: {rmse:.2f}')
    st.write(f'- R²: {r_squared:.2f}')
