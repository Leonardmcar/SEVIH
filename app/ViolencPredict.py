import json
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import subprocess

def load_violence_json(file_path="salida/maestro.json"):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def predict_next_year(data, current_year):
    next_year = current_year + 1
    violencia_predict = {}

    for year, municipios in data["violencia_por_año"].items():
        year = int(year)
        for municipio, sexos in municipios.items():
            if municipio not in violencia_predict:
                violencia_predict[municipio] = {}
            for sexo, tipos_violencia in sexos.items():
                if sexo not in violencia_predict[municipio]:
                    violencia_predict[municipio][sexo] = {}

                for tipo_violencia, intencionalidades in tipos_violencia.items():
                    if tipo_violencia not in violencia_predict[municipio][sexo]:
                        violencia_predict[municipio][sexo][tipo_violencia] = {}

                    for intencionalidad, detalles in intencionalidades.items():
                        # Predicciones para agentes
                        agentes_pred = predict_component(detalles.get("agentes", {}))
                        
                        # Predicciones para Notificado al MP
                        notificado_pred = predict_component(detalles.get("Notificado al MP", {}))
                        
                        # Predicciones para Tipo de atención
                        tipo_atencion_pred = predict_component(detalles.get("Tipo de atención", {}))

                        # Guardar predicciones
                        violencia_predict[municipio][sexo][tipo_violencia][intencionalidad] = {
                            "agentes": agentes_pred,
                            "Notificado al MP": notificado_pred,
                            "Tipo de atención": tipo_atencion_pred
                        }

    return {str(next_year): violencia_predict}

def predict_component(component_data):
    predictions = {}
    for key, value in component_data.items():
        historical_values = [value] if isinstance(value, int) else value
        prediction = predict_time_series(historical_values)
        predictions[key] = prediction
    return predictions

def predict_time_series(data):
    if len(data) < 2:  # Si hay menos de 2 datos históricos, devolvemos el último valor
        return data[-1] if data else 0

    # Ajustar modelo ARIMA
    try:
        model = ARIMA(data, order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return int(forecast[0])
    except Exception as e:
        print(f"Error al predecir: {e}")
        return 0

def save_predictions(predictions, file_path="salida/maestroPredictViolencia.json"):
    with open(file_path, "w") as file:
        json.dump(predictions, file, indent=4, ensure_ascii=False)
    print(f"Predicciones guardadas en {file_path}")

# Ejecutar las predicciones
current_year = datetime.now().year
data = load_violence_json()
predictions = predict_next_year(data, current_year)
save_predictions(predictions)

subprocess.run(["python", "AgrupPredict.py"])