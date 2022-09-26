# Databricks notebook source
# MAGIC %md
# MAGIC # Sobre os dados
# MAGIC 
# MAGIC Como uma empresa orientada para a comunidade, apreciamos o uso de dados abertos em nossa análise. Portanto, para este
# MAGIC teste, queremos que você baixe e seja criativo em sua análise sobre Open Ocean Data do
# MAGIC Digital Ocean Institute da Irlanda.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DateType, TimestampType, StringType
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import pandas as pd

from datetime import date, datetime, timedelta

from pandas_profiling import ProfileReport

import plotly.express as px

# COMMAND ----------

# converte mes numerico pra string
def func(month):
    if month == 1: return 'JAN'
    if month == 2: return 'FEB'
    if month == 3: return 'MAR'
    if month == 4: return 'APR'
    if month == 5: return 'MAY'
    if month == 6: return 'JUN'
    if month == 7: return 'JUL'
    if month == 8: return 'AUG'
    if month == 9: return 'SEP'
    if month == 10: return 'OCT'
    if month == 11: return 'NOV'
    if month == 12: return 'DEC'
    return month

# funcção converte string para datetime
func_time_datetime =  udf (lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'), DateType())

# funcção converte string para datetime
func_time_stamp =  udf (lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'), TimestampType())
func_time_stamp_simples =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), TimestampType())

# funcção converte string para datetime
func_time_string =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1) Entendendo os dados
# MAGIC 
# MAGIC Uma análise sobre Open Ocean Data do Digital Ocean Institute da Irlanda. 
# MAGIC 
# MAGIC 
# MAGIC **Intervalo de tempo:** \
# MAGIC Análise feita num intervalo de 1 ano, do dia 21/09/2021 a 21/09/2022.

# COMMAND ----------

# MAGIC %md
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2) Coleta de dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tides - Marés
# MAGIC Medidas sobre Marés coletadas por várias bóias no mar da Irlanda
# MAGIC 
# MAGIC Time - UTC \
# MAGIC Station ID \
# MAGIC Latitude - degrees_north \
# MAGIC Longitude - degrees_east \
# MAGIC Water Level (LAT) - meters \
# MAGIC Water Level (OD Malin) - meters \
# MAGIC Quality Control Flag

# COMMAND ----------

# File location and type
file_location_td = "/FileStore/tables/tide/*.csv"

# The applied options are for CSV files. For other file types, these will be ignored.
df_tides = spark.read.format('csv') \
                .option("inferSchema", 'true') \
                .option("header", 'true') \
                .option("sep", ',') \
                .load(file_location_td)\
                .drop_duplicates()

# remove a primeira linha que possui dados discrepantes do resto da base
df_tides = df_tides.filter(~col("time").contains(df_tides.first()[0]))

# convert types
df_tides = df_tides.withColumn("latitude", df_tides["latitude"].cast(FloatType()))\
                   .withColumn("longitude", df_tides["longitude"].cast(FloatType()))\
                   .withColumn("Water_Level_LAT", df_tides["Water_Level_LAT"].cast(FloatType()))\
                   .withColumn("Water_Level_OD_Malin", df_tides["Water_Level_OD_Malin"].cast(FloatType()))\

display(df_tides)
df_tides.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Estações na base de dados

# COMMAND ----------

# df_tides.station_id.unique()
display(df_tides.select('station_id', 'latitude', 'longitude').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ### Waves - Ondas
# MAGIC Medidas sobre Ondas coletadas por várias bóias no mar da Irlanda
# MAGIC 
# MAGIC 
# MAGIC Time - UTC\
# MAGIC Station_id \
# MAGIC Latitude - degrees_north \
# MAGIC Longitude - degrees_east \
# MAGIC Peak Period - S\
# MAGIC Peak Direction - degrees_true\
# MAGIC Upcross Period - S\
# MAGIC Significant Wave Height - cm\
# MAGIC Maximum Wave Height - cm\
# MAGIC Sea Temperature - degree_C\
# MAGIC Current Speed - m/s\
# MAGIC Current Direction - degrees_true

# COMMAND ----------

# File location and type
file_location_wv = "/FileStore/tables/wave/*.csv"

# The applied options are for CSV files. For other file types, these will be ignored.
df_waves = spark.read.format('csv') \
                .option("inferSchema", 'true') \
                .option("header", 'true') \
                .option("sep", ',') \
                .load(file_location_wv) \
                .drop_duplicates()\
                .select('time', 'station_id', 'latitude', 'longitude', 'PeakPeriod', 'PeakDirection', 'UpcrossPeriod', 'SignificantWaveHeight', 'Hmax', 
                        'SeaTemperature', 'MeanCurSpeed', 'MeanCurDirTo')

# remove a primeira linha que possui dados discrepantes do resto da base
df_waves = df_waves.filter(~col("time").contains(df_waves.first()[0]))

# convert types
df_waves = df_waves.withColumn("latitude", df_waves["latitude"].cast(FloatType()))\
                   .withColumn("longitude", df_waves["longitude"].cast(FloatType()))\
                   .withColumn("PeakPeriod", df_waves["PeakPeriod"].cast(FloatType()))\
                   .withColumn("PeakDirection", df_waves["PeakDirection"].cast(FloatType()))\
                   .withColumn("UpcrossPeriod", df_waves["UpcrossPeriod"].cast(FloatType()))\
                   .withColumn("SignificantWaveHeight", df_waves["SignificantWaveHeight"].cast(FloatType()))\
                   .withColumn("Hmax", df_waves["Hmax"].cast(FloatType()))\
                   .withColumn("SeaTemperature", df_waves["SeaTemperature"].cast(FloatType()))\
                   .withColumn("MeanCurSpeed", df_waves["MeanCurSpeed"].cast(FloatType()))\
                   .withColumn("MeanCurDirTo", df_waves["MeanCurDirTo"].cast(FloatType()))\

display(df_waves)
df_waves.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Estações na base de dados
# MAGIC 
# MAGIC Observaçao: estacoes AMETS Berth B Wave Buoy e Westwave Wave Buoy estava com sua base de dados vazia

# COMMAND ----------

# df_waves.station_id.unique()
display(df_waves.select('station_id', 'latitude', 'longitude').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3) Limpeza dos dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tides

# COMMAND ----------

df_tides.count()

# COMMAND ----------

df_tides.select(*[(
                   F.count(F.when((F.isnan(c) | F.col(c).isNull()), c)) if t not in ("timestamp", "date")
                   else F.count(F.when(F.col(c).isNull(), c))
                  ).alias(c)
                  for c, t in df_tides.dtypes if c in df_tides.columns
                 ]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Waves

# COMMAND ----------

df_waves.count()

# COMMAND ----------

df_waves.select(*[(
                   F.count(F.when((F.isnan(c) | F.col(c).isNull()), c)) if t not in ("timestamp", "date")
                   else F.count(F.when(F.col(c).isNull(), c))
                  ).alias(c)
                  for c, t in df_waves.dtypes if c in df_waves.columns
                 ]).show()
