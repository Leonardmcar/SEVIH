import psycopg2
from psycopg2 import DatabaseError
from decouple import config



def get_connection():
    try:
        return psycopg2.connect(
            host=config('POSTGRES_HOST'),
            database=config('POSTGRES_DB'),
            user=config('POSTGRES_USER'),
            password=config('POSTGRES_PASSWORD'),
            port=config('POSTGRES_PORT')
        )
    except DatabaseError as ex:
        raise ex