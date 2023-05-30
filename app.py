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

# Variables pour stocker les informations de sélection
selected_model = ''
selected_airports = ''

# Mettre à jour les variables de sélection
selected_model = model_selection
selected_airports = f'{home_airport} → {paired_airport}'

# Créer un graphique avec les données initiales et le titre approprié
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=df_filtered_initial['date'], y=df_filtered_initial['pax_total'], fill='tozeroy',
                         name='Donnée de la route'), row=1, col=1)
fig.update_layout(title=f'Donnée de la route - {selected_airports}')

mae = 0.0
rmse = 0.0
r_squared = 0.0
if run_forecast_button:
    forecast_button_clicked = True

    if model_selection == 'Prophet':
        # Entraîner le modèle Prophet
        model_prophet.fit(df_filtered_initial.rename(columns={'date': 'ds', 'pax_total': 'y'}))

        # Générer les dates de prédiction
        future_dates = pd.date_range(start=forecast_date, periods=nb_days)

        # Effectuer la prédiction avec le modèle Prophet
        forecast = model_prophet.predict(pd.DataFrame({'ds': future_dates}))

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(dash='dash', color='red'),
                       name='Donnée prédite avec Prophet'), row=1, col=1
        )

        # Calculer les performances
        actual = df_filtered_initial['pax_total']
        predicted = forecast['yhat'][:len(actual)]
        mae, rmse, r_squared = calculate_performance(actual, predicted)

    elif model_selection == 'LGBMRegressor':
        # Préparation des données pour LGBMRegressor
        X = np.array(range(len(df_filtered_initial)))
        y = df_filtered_initial['pax_total'].values

        # Entraînement du modèle LGBMRegressor
        model_lgbm = LGBMRegressor()
        model_lgbm.fit(X.reshape(-1, 1), y)

        # Générer les dates de prédiction
        forecast_dates = pd.date_range(start=forecast_date, periods=nb_days)

        # Effectuer la prédiction avec le modèle LGBMRegressor
        lgb_predictions = model_lgbm.predict(np.array(range(len(df_filtered_initial), len(df_filtered_initial) + nb_days))).reshape(-1, 1)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=lgb_predictions, mode='lines', line=dict(dash='dash', color='green'),
                       name='Donnée prédite avec LGBMRegressor'), row=1, col=1
        )

        # Calculer les performances
        actual = df_filtered_initial['pax_total']
        predicted = np.concatenate([actual, lgb_predictions])
        mae, rmse, r_squared = calculate_performance(actual, predicted)

    elif model_selection == 'XGBRegressor':
        # Préparation des données pour XGBRegressor
        X = np.array(range(len(df_filtered_initial)))
        y = df_filtered_initial['pax_total'].values

        # Entraînement du modèle XGBRegressor
        model_xgb = XGBRegressor()
        model_xgb.fit(X.reshape(-1, 1), y)

        # Générer les dates de prédiction
        forecasted_dates = pd.date_range(start=forecast_date, periods=nb_days)

        # Effectuer la prédiction avec le modèle XGBRegressor
        forecasted_values = model_xgb.predict(np.array(range(len(df_filtered_initial), len(df_filtered_initial) + nb_days))).reshape(-1, 1)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=forecasted_dates, y=forecasted_values, mode='lines', line=dict(dash='dash', color='yellow'),
                       name='Donnée prédite avec XGBRegressor'), row=1, col=1
        )

        # Calculer les performances
        actual = df_filtered_initial['pax_total']
        predicted = np.concatenate([actual, forecasted_values])
        mae, rmse, r_squared = calculate_performance(actual, predicted)

    elif model_selection == 'RandomForestRegressor':
        # Préparation des données pour RandomForestRegressor
        X = np.array(range(len(df_filtered_initial)))
        y = df_filtered_initial['pax_total'].values

        # Entraînement du modèle RandomForestRegressor
        model_rf = RandomForestRegressor()
        model_rf.fit(X.reshape(-1, 1), y)

        # Générer les dates de prédiction
        prediction_dates = pd.date_range(start=forecast_date, periods=nb_days)

        # Effectuer la prédiction avec le modèle RandomForestRegressor
        predictions = model_rf.predict(np.array(range(len(df_filtered_initial), len(df_filtered_initial) + nb_days))).reshape(-1, 1)

        # Ajouter les données prédictives au graphique
        fig.add_trace(
            go.Scatter(x=prediction_dates, y=predictions, mode='lines', line=dict(dash='dash', color='blue'),
                       name='Donnée prédite avec RandomForestRegressor'), row=1, col=1
        )

        # Calculer les performances
        actual = df_filtered_initial['pax_total']
        predicted = np.concatenate([actual, predictions])
        mae, rmse, r_squared = calculate_performance(actual, predicted)

# Mettre à jour le titre du graphique avec les performances
fig.update_layout(title=f'Donnée de la route - {selected_airports} | MAE: {mae:.2f}, RMSE: {rmse:.2f}, R-Squared: {r_squared:.2f}')

# Afficher le graphique
st.plotly_chart(fig)
