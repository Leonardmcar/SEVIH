import pandas as pd
import json
from conection.db import get_connection
import subprocess

def load_data_from_postgres():
    connection = get_connection()
    query = """SELECT * FROM dataware."DataHGT";"""
    try:
        data = pd.read_sql(query, connection)
        data = data.drop(columns=["_airbyte_generation_id", "_airbyte_raw_id", "_airbyte_extracted_at", "_airbyte_meta"], errors='ignore')
        print("Datos cargados correctamente")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
    finally:
        connection.close()
    return data

def analyze_and_save_to_json(data):
    # Normalización de texto para evitar problemas de formato
    data['sexo_des'] = data['sexo_des'].str.strip().str.upper()
    data['intencionalidad_des'] = data['intencionalidad_des'].str.strip().str.upper()
    data['municipio_ocurrencia_des'] = data['municipio_ocurrencia_des'].str.strip().str.upper()
    data['agente_lesion_des'] = data['agente_lesion_des'].str.strip().str.upper()

    # Convertir fechas a formato datetime y extraer año
    data['fecha_atencion'] = pd.to_datetime(data['fecha_atencion'], errors='coerce')
    data['año_atencion'] = data['fecha_atencion'].dt.year

    # Estructura de análisis por año, municipio y género
    lesiones_por_año = {}
    años = data['año_atencion'].dropna().unique()

    for año in años:
        año_data = data[data['año_atencion'] == año]
        lesiones_por_municipio = {}
        municipios = año_data['municipio_ocurrencia_des'].unique()

        for municipio in municipios:
            municipio_data = año_data[año_data['municipio_ocurrencia_des'] == municipio]
            municipio_entry = {}
            for sexo in ['HOMBRE', 'MUJER']:
                sexo_data = municipio_data[municipio_data['sexo_des'] == sexo]
                sexo_entry = {}

                # Agrupación por intencionalidad y agente de lesión
                for intencionalidad, group in sexo_data.groupby('intencionalidad_des'):
                    agentes_counts = group['agente_lesion_des'].value_counts().to_dict()
                    ministerio_publico_counts = group['ministerio_publico_des'].value_counts().to_dict()

                    # Conteo de tipos de atención específicos para cada intencionalidad
                    tipo_atencion_counts = {
                        "MEDICA": len(group[group['tipo_atencion_1_des'] == "MEDICA"]),
                        "PSICOLOGICA": len(group[group['tipo_atencion_2_des'] == "PSICOLOGICA"]),
                        "QUIRURGICA": len(group[group['tipo_atencion_3_des'] == "QUIRURGICA"]),
                        "PSIQUIATRICA": len(group[group['tipo_atencion_4_des'] == "PSIQUIATRICA"]),
                        "CONSEJERIA": len(group[group['tipo_atencion_5_des'] == "CONSEJERIA"]),
                    }

                    # Eliminar tipos de atención con conteo 0
                    tipo_atencion_counts = {key: value for key, value in tipo_atencion_counts.items() if value > 0}

                    # Crear entrada para la intencionalidad
                    intencionalidad_entry = {
                        "agentes": agentes_counts,
                        "Notificado al MP": ministerio_publico_counts,
                        "Tipo de atención": tipo_atencion_counts
                    }
                    sexo_entry[intencionalidad] = intencionalidad_entry

                if sexo_entry:  # Solo guardar si hay datos para este género
                    municipio_entry[sexo] = sexo_entry

            if municipio_entry:  # Solo guardar si hay datos para este municipio
                lesiones_por_municipio[municipio] = municipio_entry

        if lesiones_por_municipio:  # Solo guardar si hay datos para este año
            lesiones_por_año[int(año)] = lesiones_por_municipio

    # Estructura final del JSON
    result = {
        "lesiones_por_año": lesiones_por_año
    }

    # Guardar en un nuevo archivo JSON
    with open("salida/maestroLesiones.json", "w") as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

    print("Datos guardados en maestroLesiones.json")

# Ejecutar el análisis
data = load_data_from_postgres()
analyze_and_save_to_json(data)

subprocess.run(["python", "ViolencPredict.py"])