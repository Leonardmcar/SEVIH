FROM python:3.11.0

# Instalar dependencias necesarias para psycopg2, mysqlclient y cron
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    default-mysql-client \
    cron \
    python3-dev \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    procps \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libffi-dev

# Actualizar pip a la última versión
RUN pip install --upgrade pip

# Instalar numpy por separado y asegurarse de que la versión es compatible
RUN pip install numpy==1.23.5

# Instalar pandas antes de instalar el resto de los paquetes
RUN pip install pandas==2.0.3

RUN pip install python-decouple==3.6

RUN pip install prophet

RUN pip install seaborn

# Instalar psycopg2-binary y mysqlclient por separado
RUN pip install psycopg2-binary==2.9.5 mysqlclient==2.1.0

# Instalar statsmodels y matplotlib por separado
RUN pip install statsmodels==0.14.0 matplotlib==3.7.2

    # Copiar el archivo de requerimientos al contenedor
COPY app/requirements.txt /app/requirements.txt

# Instalar las dependencias desde requirements.txt
RUN pip install -r /app/requirements.txt

# Copiar el archivo cronjob y la aplicación al contenedor
COPY ./cronjob /etc/cron.d/my-cron-job
COPY app/. /app

# Dar permisos correctos para el archivo cronjob
RUN chmod 0644 /etc/cron.d/my-cron-job

# Aplicar el cronjob
RUN crontab /etc/cron.d/my-cron-job

# Crear el directorio de trabajo
WORKDIR /app

# Iniciar cron y luego ejecutar el contenedor de forma indefinida
CMD cron && tail -f /dev/null
