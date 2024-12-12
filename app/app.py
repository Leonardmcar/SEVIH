import pandas as pd
import json
from conection.db import get_connection
import subprocess

def load_data_from_postgres():
    connection = get_connection()
    query = """SELECT * FROM dataware."DataHGT";"""
    try:
        data = pd.read_sql(query, connection)
        data = data.drop(columns=["_airbyte_generation_id","_airbyte_raw_id","_airbyte_extracted_at","_airbyte_meta"], errors='ignore')
        print("Datos cargados correctamente")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
    finally:
        connection.close()
    return data

def analyze_and_save_to_json(data):
    # Normalización para evitar problemas de formato
    data['sexo_des'] = data['sexo_des'].str.strip().str.upper()
    data['intencionalidad_des'] = data['intencionalidad_des'].str.strip().str.upper()
    data['municipio_ocurrencia_des'] = data['municipio_ocurrencia_des'].str.strip().str.upper()
    data['agente_lesion_des'] = data['agente_lesion_des'].str.strip().str.upper()

    # Convertir fechas a formato datetime
    data['fecha_atencion'] = pd.to_datetime(data['fecha_atencion'], errors='coerce')
    data['año_atencion'] = data['fecha_atencion'].dt.year

    # Distribución de Intencionalidad por Año de Atención, municipio y Género
    violencia_por_año = {}
    años = data['año_atencion'].dropna().unique()

    for año in años:
        año_data = data[data['año_atencion'] == año]
        violencia_por_municipio = {}
        municipioes = año_data['municipio_ocurrencia_des'].unique()

        for municipio in municipioes:
            municipio_data = año_data[año_data['municipio_ocurrencia_des'] == municipio]
            municipio_entry = {}
            for sexo in ['HOMBRE', 'MUJER']:
                sexo_data = municipio_data[municipio_data['sexo_des'] == sexo]
                sexo_entry = {}

                # Para cada tipo de violencia, agrupamos por intencionalidad y agente de lesión
                tipos_violencia = {
                    "violencia fisica": "VIOLENCIA FISICA",
                    "violencia sexual": "VIOLENCIA SEXUAL",
                    "violencia psicologica": "VIOLENCIA PSICOLOGICA",
                    "violencia economica": "VIOLENCIA ECONOMICA/PATRIMONIAL",
                    "Abandono/Negligencia": "ABANDO/NEGLIGENCIA",
                }

                for i, (tipo_key, tipo_violencia) in enumerate(tipos_violencia.items(), start=1):
                    tipo_violencia_data = sexo_data[
                        (sexo_data[f'tipo_violencia_{i}_des'] == tipo_violencia) &
                        (~sexo_data['intencionalidad_des'].isin(['ACCIDENTE', 'AUTOINFLIGIDO']))
                    ]

                    if not tipo_violencia_data.empty:
                        intencionalidad_data = {}
                        for intencionalidad, group in tipo_violencia_data.groupby('intencionalidad_des'):
                            agentes_counts = group['agente_lesion_des'].value_counts().to_dict()
                            ministerio_publico_counts = group['ministerio_publico_des'].value_counts().to_dict()

                            # Conteo de tipos de atención específicos para este grupo
                            tipo_atencion_counts = {
                                "MEDICA": len(group[group['tipo_atencion_1_des'] == "MEDICA"]),
                                "PSICOLOGICA": len(group[group['tipo_atencion_2_des'] == "PSICOLOGICA"]),
                                "QUIRURGICA": len(group[group['tipo_atencion_3_des'] == "QUIRURGICA"]),
                                "PSIQUIATRICA": len(group[group['tipo_atencion_4_des'] == "PSIQUIATRICA"]),
                                "CONSEJERIA": len(group[group['tipo_atencion_5_des'] == "CONSEJERIA"]),
                                "OTRO": len(group[group['tipo_atencion_6_des'] == "OTRO"]),
                                "PILDORA ANT EMERGENCIA": len(group[group['tipo_atencion_7_des'] == "PILDORA ANT EMERGENCIA"]),
                                "PROFILAXIS VIH": len(group[group['tipo_atencion_8_des'] == "PROFILAXIS VIH"]),
                                "PROFILAXIS OTRAS ITS": len(group[group['tipo_atencion_9_des'] == "PROFILAXIS OTRAS ITS"]),
                            }

                            # Eliminar tipos de atención con valor 0
                            tipo_atencion_counts = {key: int(value) for key, value in tipo_atencion_counts.items() if value > 0}

                            intencionalidad_data[intencionalidad] = {
                                "agentes": agentes_counts,
                                "Notificado al MP": ministerio_publico_counts,
                                "Tipo de atención": tipo_atencion_counts
                            }

                        if intencionalidad_data:
                            sexo_entry[tipo_key] = intencionalidad_data

                if sexo_entry:  # Solo guardar si hay datos para este género
                    municipio_entry[sexo] = sexo_entry

            if municipio_entry:  # Solo guardar si hay datos para esta municipio
                violencia_por_municipio[municipio] = municipio_entry

        if violencia_por_municipio:  # Solo guardar si hay datos para este año
            violencia_por_año[int(año)] = violencia_por_municipio

    # Estructura del JSON
    result = {
        "violencia_por_año": violencia_por_año
    }

    # Guardar en JSON
    with open("maestro.json", "w") as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

    print("Datos guardados en maestro.json")

# Ejecutar el análisis
data = load_data_from_postgres()
analyze_and_save_to_json(data)

subprocess.run(["python", "app2.py"])
