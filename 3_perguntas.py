# Databricks notebook source
# MAGIC %run ./2_explora_dados

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.5) Questions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 - What is the lowest temperature of each one of the Bouys?\

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resposta: 
# MAGIC 
# MAGIC A TABELA abaixo nos mostra as temperaturas mais baixas por boia (Bouys).

# COMMAND ----------

# classifica pelo maior valor agrupado por station
dtprox = Window.partitionBy("station_id").orderBy(F.asc("SeaTemperature"))

# filtra por 'Buoy', conforme pede o enunciado da pergunta
# cria coluna 'rn' classificando, por estação, as menores temperaturas. 
# Filtra pela menor termperatura rankeada (rn = 1), exclui o restante das outras temperaturas rankeadas
seatmp = df_waves.filter(col('station_id').contains("Buoy"))\
                 .withColumn("rn", F.row_number().over(dtprox))\
                 .filter("rn = 1")\
                 .drop('rn')\
                 .orderBy('station_id')\
                 .select('latitude', 'longitude', 'station_id', 'SeaTemperature', 'time')

# muda tipagem de string para timestamp
seatmp = seatmp.withColumn('timetmstp', func_time_stamp(col('time')))\
               .withColumn('month' , month("timetmstp"))\
               
func_udf = udf(func, StringType())

# convert o mes de numero para string
seatmp = seatmp.withColumn('month_name', func_udf(seatmp['month']))\
               .drop('timetmstp')
                                    
display(seatmp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Localização das Boias (Bouys) que registraram menores temperaturas 

# COMMAND ----------

pd_seatmp = seatmp.toPandas()

fig = px.density_mapbox(pd_seatmp, 
                        lat=pd_seatmp.latitude, 
                        lon=pd_seatmp.longitude, 
                        z=pd_seatmp.SeaTemperature, 
                        radius=15, 
                        center=dict(lat=54.2753, lon=-10.29737), 
                        zoom=6, 
                        mapbox_style="stamen-terrain")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which usually month it occurs?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resposta: 
# MAGIC 
# MAGIC De GRÁFICO DE BARRAS abaixo, é o mês de Março, com 66,7% (2) ocorrencias, enquanto o mês de Janeiro tem 33,3% (1) ocorrencia.

# COMMAND ----------

import plotly.express as px

df = px.data.tips()
fig = px.pie(seatmp.toPandas(), names='month_name')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Where (lat/long) do we have the biggest water level?\

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resposta: 
# MAGIC 
# MAGIC A TABELA e o gráfico abaixo nos mostra que o nivel mais alto de agua é das coordenadas latitude 55.3717, longitude -7.3344

# COMMAND ----------

# seleciona colunas, 
df_watlvl = df_tides.select('latitude', 'longitude', 'station_id', 'Water_Level_LAT', 'time')\
                    .na.drop(subset='Water_Level_LAT')\
                    .orderBy(desc('Water_Level_LAT'))

# muda tipagem de string para timestamp
pd_watlvl = df_watlvl.withColumn('timetmstp', func_time_stamp(col('time')))\
                     .withColumn('month' , month("timetmstp"))\
                     .drop('timetmstp')\
                     .toPandas()

biggest_wtr_lvl = pd_watlvl.iloc[0:1]

display(biggest_wtr_lvl)

# COMMAND ----------

fig = px.density_mapbox(biggest_wtr_lvl, 
                        lat=biggest_wtr_lvl.latitude, 
                        lon=biggest_wtr_lvl.longitude, 
                        z=biggest_wtr_lvl.Water_Level_LAT, 
                        radius=20, 
                        center=dict(lat=55.3717, lon=-7.3344), 
                        zoom=7, 
                        mapbox_style="stamen-terrain")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which usually month it occurs?

# COMMAND ----------

biggest_wtr_lvl.month

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resposta: 
# MAGIC 
# MAGIC De acordo com o codigo acima, o mês que usualmente ocorre é o mes de setembro

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 - How the Wave Lenghts correlates with Sea Temperature?\
# MAGIC It is possible to predict with accuracy the Wave Lenght, based on the Sea Temperature and the Bouy location?

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

pd_waves = df_waves.toPandas()

corr = pd_waves[['latitude', 'longitude', 'Hmax', 'SeaTemperature']].corr()

fig, ax = plt.subplots(figsize=(12,8))
display(sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Resposta:
# MAGIC 
# MAGIC **Wave Lenghts (Hmax)** com **Sea Temperature** possuem correlação negativa de -0,29 indica que uma leve influencia entre as features (valor 0,29, que vai de 0 a 1), e pelo fato de ser negativa, indica que uma feature pode ter influencia contraria sobre a outra.

# COMMAND ----------

# MAGIC %md
# MAGIC ### It is possible to predict with accuracy the Wave Lenght, based on the Sea Temperature and the Bouy location?

# COMMAND ----------

# MAGIC %md
# MAGIC Com relação a **latitude** e **longitude**, possuem correlação em modulo de -0,033 e 0,08 respectivamente, bem proximas de 0, o que indica que não se sabe exatamente o que ocorre com uma feature quando a outra varia e portanto, pode indicar uma potencial complementariedade entre si, sendo potenciais candidatas à combinação de features
