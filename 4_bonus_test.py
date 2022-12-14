# Databricks notebook source
# MAGIC %md
# MAGIC # Bonus items
# MAGIC 
# MAGIC 
# MAGIC Build a Time Series model that can predict the sea temperature throughout the year.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agrupando os dados por 'time' e tire a média se houver vários valores 'SeaTemperature' em um mesmo dia da feature 'time'

# COMMAND ----------

# MAGIC %md
# MAGIC Waves

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

# STEP 1: RUN THIS CELL TO INSTALL BAMBOOLIB

# You can also install bamboolib on the cluster. Just talk to your cluster admin for that
# %pip install bamboolib  

# Heads up: this will restart your python kernel, so you may need to re-execute some of your other code cells.

# COMMAND ----------

# STEP 2: RUN THIS CELL TO IMPORT AND USE BAMBOOLIB

# import bamboolib as bam

# This opens a UI from which you can import your data
# bam  

# Already have a pandas data frame? Just display it!
# Here's an example
# import pandas as pd
# df_test = pd.DataFrame(dict(a=[1,2]))
# df_test  # <- You will see a green button above the data set if you display it

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/df_top_norte-2.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_waves = spark.read.format(file_type) \
                     .option("inferSchema", infer_schema) \
                     .option("header", first_row_is_header) \
                     .option("sep", delimiter) \
                     .load(file_location)\
                     .drop('ds1')\
                     .withColumnRenamed('ds0', 'time')\
                     .withColumnRenamed('y', 'MeanSeaTemperature')

display(df_waves)

# COMMAND ----------

# funcção converte string para datetime
# func_time_datetime =  udf (lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'), DateType())
func_time_datetime =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), DateType())

func_time_stamp_simples =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), TimestampType())

# COMMAND ----------

# muda tipagem de time de string para datetime (por causa do formato %y%m%d)
df_waves_timser = df_waves.withColumn('timedt', func_time_datetime(col('time')))
# display(df_waves_timser)

# # Agrupe os dados por 'time' e tire a média se houver vários valores 'SeaTemperature' no mesmo dia
# df_waves_timser = df_waves_timser.groupBy('timedt')\
#                           .agg(F.mean("SeaTemperature").alias("MeanSeaTemperature"))

# # com o time agrupado, voltar para string
df_waves_timser = df_waves_timser.withColumn('timestr', date_format('timedt',"yyyy-MM-dd"))
# display(df_waves_timser)

# # muda tipagem de string para timestamp
df_waves_timser = df_waves_timser.withColumn('time', func_time_stamp_simples(col('timestr')))
# display(df_waves_timser)

# # apaga colunas desnecessarias
df_waves_timser = df_waves_timser.drop('timedt', 'timestr')
# display(df_waves_timser)

# # organiza as colunas e ordena as datas por ordem crescente
df_waves_timser = df_waves_timser.select('time', 'MeanSeaTemperature').orderBy('time')
# display(df_waves_timser)

df_waves_timser = df_waves_timser.withColumnRenamed('time', 'ds0')\
                                 .withColumnRenamed('MeanSeaTemperature', 'y')

# # converte para dataframe pandas
pd_waves_timser = df_waves_timser.toPandas()
display(pd_waves_timser)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise da feature de tempo

# COMMAND ----------

target_col = "MeanSeaTemperature"
time_col = "time"

df_time_range = pd_waves_timser[time_col].agg(["min", "max"])
df_time_range

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise de target

# COMMAND ----------

# MAGIC %md
# MAGIC Status do target da série temporal

# COMMAND ----------

target_stats_df = pd_waves_timser[target_col].describe()
display(target_stats_df.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Verifique o número de valores ausentes na coluna de target

# COMMAND ----------

def num_nulls(x):
    num_nulls = x.isnull().sum()
    return pd.Series(num_nulls)

null_stats_df = pd_waves_timser.apply(num_nulls)[target_col]
null_stats_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize os dados

# COMMAND ----------

df_sub = pd_waves_timser

df_sub = df_sub.filter(items=[time_col, target_col])
df_sub.set_index(time_col, inplace=True)
df_sub[target_col] = df_sub[target_col].astype("float")

fig = df_sub.plot()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sazonalidade
# MAGIC 
# MAGIC Algo que ocorre em determinados periodos de tempo, que influenciam determinado fato
# MAGIC 
# MAGIC No caso, estações do ano podem influenciar a temperatura do lago

# COMMAND ----------

from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

temperature = pd_waves_timser.set_index("time").to_period("D")

temperature

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parcelas sazonais ao longo de uma semana e mais de um ano

# COMMAND ----------

X = temperature.copy()

# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 7))
seasonal_plot(X, y="MeanSeaTemperature", period="week", freq="day", ax=ax0)
display(seasonal_plot(X, y="MeanSeaTemperature", period="year", freq="dayofyear", ax=ax1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Periodograma

# COMMAND ----------

display(plot_periodogram(temperature.MeanSeaTemperature))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deterministic Process
# MAGIC 
# MAGIC Quando os valores futuros de amostras podem ser previstos a partir de dados historicos
# MAGIC 
# MAGIC Usado para criar features de tendência.

# COMMAND ----------

# MAGIC %md
# MAGIC Tanto o gráfico sazonal quanto o periodograma sugerem uma forte sazonalidade ANUAL.
# MAGIC 
# MAGIC De fato, de acordo com as estacoes do ano, indicam uma queda de temperatura no inverno (primero e ultimos meses do ano), e alta de temperatua (meio do ano) – uma possível motivo para sazonalidade.

# COMMAND ----------

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=12)  # 12 sin/cos pairs for "A"nnual seasonality

# Para usar períodos sazonais, precisaremos instanciar um deles como um "additional_terms". 
dp = DeterministicProcess(
    index=temperature.index,     # dados do index, data
    order=1,                     # trend (ordem polinominal 1 linear, 2 para quadrático, 3 para cúbico)
    seasonal=True,               # seasonality (indicators) - se false, ele sobrepoe ao peiodo anterior
    additional_terms=[fourier]   # annual seasonality (fourier) 
)

X = dp.in_sample()  # dados das amostras de treinamento, transformados na serie de fourier, das amostras reais

# COMMAND ----------

# X

# COMMAND ----------

# display(X[['sin(1,freq=A-DEC)','cos(1,freq=A-DEC)', 
#            'sin(2,freq=A-DEC)','cos(2,freq=A-DEC)',
#            'sin(3,freq=A-DEC)','cos(3,freq=A-DEC)',
#            'sin(4,freq=A-DEC)','cos(4,freq=A-DEC)',
#            'sin(5,freq=A-DEC)','cos(5,freq=A-DEC)',
#            'sin(6,freq=A-DEC)','cos(6,freq=A-DEC)',
#            'sin(7,freq=A-DEC)','cos(7,freq=A-DEC)',
#            'sin(8,freq=A-DEC)','cos(8,freq=A-DEC)',
#            'sin(9,freq=A-DEC)','cos(9,freq=A-DEC)',
#            'sin(10,freq=A-DEC)','cos(10,freq=A-DEC)',
#          ]].query('index < "2021-12-30"').plot())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Previsões
# MAGIC 
# MAGIC Com nosso conjunto de recursos criado, estamos prontos para ajustar o modelo e fazer previsões. Adicionaremos uma previsão de 360 dias para ver como nosso modelo extrapola além dos dados de treinamento

# COMMAND ----------

import xgboost as xgb

# COMMAND ----------

# valores de target
y = temperature["MeanSeaTemperature"]                         # dados reais de temperatura, target de treinamento

# modelo e fit
model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

# previsao
y_pred = pd.Series(model.predict(X), index=y.index)           # dados de target previstos usando dados historicos, mesmo linha do tempo de y

# forecast
X_fore = dp.out_of_sample(steps=360)                          # dados fora da amostra, transformados na serie de fourier, linha de forecast
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index) # dados de target usando dados gerados pela serie de fourier, forecast

# grafico
ax = y.plot(color='0.25', style='.', title="Sea Temperature - Seasonal Forecast") # configura pontos dos dados de target reais grafico
# ax = y_fore.plot(color='0.25', style='.', title="Sea Temperature - Seasonal Forecast") # configura pontos dos dados de target reais grafico
ax = y_pred.plot(ax=ax, label="Seasonal")                     # plota grafico de target previsto dentro da linha do tempo dos dados historicos
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')# plota grafico de target previsto apos os dados historicos, forecast
_ = ax.legend()
display(ax)
