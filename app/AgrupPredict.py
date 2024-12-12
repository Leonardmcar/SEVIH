import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import datetime
from conection.db import get_connection
import json

def load_data_from_postgres():
    connection = get_connection()
    query = """SELECT * FROM dataware."DataHGT";"""
    try:
        data = pd.read_sql(query, connection)
        print("Datos cargados correctamente")

        # Formato de fecha y manejo de nulos
        data['fecha_atencion'] = pd.to_datetime(data['fecha_atencion'], errors='coerce')
        data = data.dropna(subset=['fecha_atencion'])  # Eliminar fechas no válidas

        # Extraer año y mes
        data['mes'] = data['fecha_atencion'].dt.month
        data['año'] = data['fecha_atencion'].dt.year
        
        # Agrupar datos por año, mes y género
        grouped_data = data.groupby(['año', 'mes', 'sexo_des']).size().reset_index(name='total')


        print(grouped_data)

        # Guardar los datos agrupados en un archivo JSON
        grouped_data.to_json('./salida/datosAgrupados.json', orient='records')

        return grouped_data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
    finally:
        connection.close()


def forecast_next_year(data):
    # Determinar el año actual y el próximo año
    current_year = datetime.datetime.now().year
    next_year = current_year + 1
    print(f"Generando predicciones para el año {next_year}...")

    # Preparar los datos
    data = data.sort_values(by=['año', 'mes', 'sexo_des'])
    forecast_list = []

    for sexo in ['HOMBRE', 'MUJER']:
        data_sex = data[data['sexo_des'] == sexo]
        data_ts = data_sex.groupby(['año', 'mes'])['total'].sum().reset_index()

        # Seleccionar los últimos 3 a 5 años de datos
        available_years = data_ts['año'].unique()
        if len(available_years) >= 5:
            min_year = max(available_years) - 4
        else:
            min_year = max(available_years) - len(available_years) + 1
        data_ts = data_ts[data_ts['año'] >= min_year]

        # Validar y preparar las columnas para las fechas
        data_ts.rename(columns={'año': 'year', 'mes': 'month'}, inplace=True)
        data_ts['fecha'] = pd.to_datetime(data_ts[['year', 'month']].assign(day=1), errors='coerce')
        data_ts = data_ts.set_index('fecha')['total']

        # Entrenar el modelo ARIMA
        model = ARIMA(data_ts, order=(3, 1, 2))
        model_fit = model.fit()

        # Generar predicciones para los meses del próximo año
        forecast_dates = pd.date_range(start=f"{next_year}-01-01", end=f"{next_year}-12-31", freq='M')
        forecast = model_fit.forecast(steps=len(forecast_dates))

        # Formatear los resultados
        forecast_df = pd.DataFrame({
            'año': forecast_dates.year,
            'mes': forecast_dates.month,
            'sexo_des': sexo,
            'total': forecast.round().astype(int)
        })
        forecast_list.extend(forecast_df.to_dict(orient='records'))

    # Guardar las predicciones en un archivo JSON
    with open('./salida/predicciones.json', 'w') as f:
        json.dump(forecast_list, f, ensure_ascii=False, indent=4)

    print(f"Predicciones para el año {next_year} guardadas en predicciones.json")
    return forecast_list


def calculate_mape(y_true, y_pred):
    """Calcular el porcentaje de error absoluto medio (MAPE)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def evaluate_predictions(data, months_ahead=12):
    """Evaluar la efectividad de las predicciones para hombres, mujeres y en general."""
    data = data.sort_values(by=['año', 'mes', 'sexo_des'])
    effectiveness_results = {}
    overall_y_true = []
    overall_y_pred = []

    for sexo in ['HOMBRE', 'MUJER']:
        # Filtrar datos por género
        data_sex = data[data['sexo_des'] == sexo]
        data_ts = data_sex.groupby(['año', 'mes'])['total'].sum().reset_index()

        # Validar y preparar las columnas para las fechas
        data_ts.rename(columns={'año': 'year', 'mes': 'month'}, inplace=True)
        data_ts['fecha'] = pd.to_datetime(data_ts[['year', 'month']].assign(day=1), errors='coerce')
        data_ts = data_ts.set_index('fecha')['total']

        # Dividir datos en entrenamiento y prueba
        if len(data_ts) <= months_ahead:
            print(f"No hay suficientes datos para evaluar las predicciones de {sexo}.")
            effectiveness_results[sexo] = None
            continue

        train_data = data_ts[:-months_ahead]
        test_data = data_ts[-months_ahead:]

        # Ajustar modelo ARIMA
        model = ARIMA(train_data, order=(1, 1, 0))
        model_fit = model.fit()

        # Generar predicciones
        forecast = model_fit.forecast(steps=len(test_data))

        # Calcular MAPE y efectividad
        mape = calculate_mape(test_data, forecast)
        effectiveness = max(0, 100 - mape)
        print(f"MAPE para {sexo}: {mape:.2f}%, Efectividad: {effectiveness:.2f}%")

        # Guardar resultados
        effectiveness_results[sexo] = effectiveness

        # Acumular datos para la evaluación general
        overall_y_true.extend(test_data.values)
        overall_y_pred.extend(forecast)

    # Calcular efectividad general
    if overall_y_true and overall_y_pred:
        overall_mape = calculate_mape(overall_y_true, overall_y_pred)
        overall_effectiveness = max(0, 100 - overall_mape)
        effectiveness_results['general'] = overall_effectiveness
        print(f"MAPE general: {overall_mape:.2f}%, Efectividad general: {overall_effectiveness:.2f}%")
    else:
        effectiveness_results['general'] = None
        print("No se pudo calcular la efectividad general debido a la falta de datos.")

    return effectiveness_results

# Cargar datos agrupados desde la base de datos
grouped_data = load_data_from_postgres()

# Evaluar la efectividad de las predicciones
model_effectiveness = evaluate_predictions(grouped_data, months_ahead=14)
print("Evaluación de la efectividad del modelo:", model_effectiveness)

# Generar predicciones para el próximo año
forecast_data = forecast_next_year(grouped_data)
print("Predicciones generadas y guardadas en predicciones.json")