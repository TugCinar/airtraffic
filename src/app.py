
df = pd.read_parquet('src/data/traffic_10lines.parquet')
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet import Prophet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np 

image_path = 'dash.jpg'

# Afficher l'image
st.image(image_path, caption="Logo", use_column_width=True)

HOME_AIRPORTS = ('LGW', 'LIS', 'LYS', 'NTE')
PAIRED_AIRPORTS = ('FUE', 'AMS', 'ORY', 'BCN', 'OPO')

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
model_prophet = Prophet()

with st.sidebar:
    home_airport = st.selectbox('Home Airport', HOME_AIRPORTS)
    paired_airport = st.selectbox('Paired Airport', PAIRED_AIRPORTS)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    model_selection = st.selectbox('Model Selection', ['Prophet', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor'])
    run_forecast_button = st.button('Forecast')

# Affichage de la table
st.write('Home Airport selected:', home_airport)
st.write('Paired Airport selected:', paired_airport)
st.write('Days of forecast:', nb_days)
st.write('Forecast Start Date:', forecast_date)

df_filtered = df[(df['home_airport'] == home_airport) & (df['paired_airport'] == paired_airport)]

# Afficher le dataframe filtré
st.dataframe(data=df_filtered, width=600, height=300)

# Filtrer les données pour le graphique initial
df_filtered_initial = df.query(
    f'home_airport == "{home_airport}" and paired_airport == "{paired_airport}"'
)
df_filtered_initial = df_filtered_initial.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()

# Créer un graphique avec les données initiales
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df_filtered_initial['date'], y=df_filtered_initial['pax_total'], fill='tozeroy',
                         name=f'{home_airport} → {paired_airport}'), row=1, col=1)

# Créer l'étiquette du graphique initial
graph_label = f"Données de la route {home_airport} → {paired_airport}"

# Mettre à jour le titre du graphique
fig.update_layout(title=graph_label)

mae = 0.0
rmse = 0.0
r_squared = 0.0
if run_forecast_button:
    forecast_button_clicked = True  # Mettre à jour le drapeau lorsque le bouton est cliqué

    df_filtered = df.query(
        f'home_airport == "{home_airport}" and paired_airport == "{paired_airport}"'
    )
    df_filtered = df_filtered.groupby('date').agg(pax_total=('pax', 'sum')).reset_index()

    forecast_dates = pd.date_range(forecast_date, periods=nb_days)  # Définir les dates de prévision ici

    if model_selection == 'Prophet':
        # Convertir forecast_date en datetime64[ns]
        forecast_date = pd.to_datetime(forecast_date)

        # Filtrer les données historiques jusqu'à la date de prévision
        df_filtered = df_filtered[df_filtered['date'] <= forecast_date]

        # Préparer les données pour Prophet
        df_prophet = df_filtered[['date', 'pax_total']].rename(columns={'date': 'ds', 'pax_total': 'y'})

        # Entraîner le modèle
        model_prophet.fit(df_prophet)

        # Prédire le trafic pour les dates de prévision
        future = pd.DataFrame({'ds': forecast_dates})
        forecast = model_prophet.predict(future)

        actual_values = df_filtered['pax_total'].values[-nb_days:]
        predicted_values = forecast['yhat'].values
        mae, rmse, r_squared = calculate_performance(actual_values, predicted_values)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dash', color='red'),
                       name='Prophet'), row=1, col=1
        )

    elif model_selection == 'LGBMRegressor':
        # Préparer les ensembles de données pour LGBMRegressor
        X_train = df_filtered['date'].values.reshape(-1, 1)
        y_train = df_filtered['pax_total'].values
        X_forecast = forecast_dates.values.reshape(-1, 1)

        # Créer une instance du modèle LGBMRegressor
        lgb_model = LGBMRegressor()
        lgb_model.fit(X_train, y_train)
        # Faire des prédictions avec LGBMRegressor
        lgb_predictions = lgb_model.predict(X_forecast)
        actual_values = df_filtered['pax_total'].values[-nb_days:]
        mae, rmse, r_squared = calculate_performance(actual_values, lgb_predictions)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=lgb_predictions, mode='lines', line=dict(dash='dash', color='green'),
                       name='LGBMRegressor'), row=1, col=1
        )

    elif model_selection == 'XGBRegressor':
        # Préparer les ensembles de données pour XGBRegressor
        ref_date = np.min(df_filtered['date']).to_pydatetime()
        X_train_numeric = (df_filtered['date'] - ref_date).dt.days.values.reshape(-1, 1)
        X_forecast_numeric = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)

        y_train = df_filtered['pax_total'].values

        # Créer une instance du modèle XGBRegressor
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train_numeric, y_train)

        # Faire des prédictions avec XGBRegressor
        xgb_predictions = xgb_model.predict(X_forecast_numeric)

        # Créer un dataframe pour les prévisions
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': xgb_predictions})

        actual_values = df_filtered['pax_total'].values[-nb_days:]
        predicted_values = forecast_df['pax_total'].values
        mae, rmse, r_squared = calculate_performance(actual_values, predicted_values)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=xgb_predictions, mode='lines', line=dict(dash='dash', color='blue'),
                       name='XGBRegressor'), row=1, col=1
        )

    elif model_selection == 'RandomForestRegressor':
        # Préparer les ensembles de données pour RandomForestRegressor
        ref_date = np.min(df_filtered['date']).to_pydatetime()
        X_train_numeric = (df_filtered['date'] - ref_date).dt.days.values.reshape(-1, 1)
        X_forecast_numeric = (forecast_dates - ref_date).days.to_numpy().reshape(-1, 1)

        y_train = df_filtered['pax_total'].values

        # Créer une instance du modèle RandomForestRegressor
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train_numeric, y_train)

        # Faire des prédictions avec RandomForestRegressor
        rf_predictions = rf_model.predict(X_forecast_numeric)

        # Créer un dataframe pour les prévisions
        forecast_df = pd.DataFrame({'date': forecast_dates, 'pax_total': rf_predictions})

        actual_values = df_filtered['pax_total'].values[-nb_days:]
        predicted_values = forecast_df['pax_total'].values
        mae, rmse, r_squared = calculate_performance(actual_values, predicted_values)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=rf_predictions, mode='lines', line=dict(dash='dash', color='orange'),
                       name='RandomForestRegressor'), row=1, col=1
        )

    # Mettre à jour le titre du graphique avec les performances
    graph_label += f"<br>MAE: {mae:.2f} | RMSE: {rmse:.2f} | R^2: {r_squared:.2f}"
    fig.update_layout(title=graph_label)

# Afficher le graphique
st.plotly_chart(fig)


