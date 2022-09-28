# Databricks notebook source
# MAGIC %run ./3_perguntas

# COMMAND ----------

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

# muda tipagem de time de string para datetime (por causa do formato %y%m%d)
df_waves_timser = df_waves.withColumn('timedt', func_time_datetime(col('time')))

# Agrupe os dados por 'time' e tire a média se houver vários valores 'SeaTemperature' no mesmo dia
df_waves_timser = df_waves_timser.groupBy('timedt')\
                          .agg(F.mean("SeaTemperature").alias("MeanSeaTemperature"))

# com o time agrupado, voltar para string
df_waves_timser = df_waves_timser.withColumn('timestr', date_format('timedt',"yyyy-MM-dd"))

# muda tipagem de string para timestamp
df_waves_timser = df_waves_timser.withColumn('time', func_time_stamp_simples(col('timestr')))

# apaga colunas desnecessarias
df_waves_timser = df_waves_timser.drop('timedt', 'timestr')

# organiza as colunas e ordena as datas por ordem crescente
df_waves_timser = df_waves_timser.select('time', 'MeanSeaTemperature').orderBy('time')

# converte para dataframe pandas
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
# MAGIC Verifique o número de valores ausentes na coluna de destino

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
# MAGIC Usar esta função nos ajudará a evitar alguns casos de falha complicados que podem surgir com séries temporais e regressão linear. O argumento de ordem refere-se à ordem polinomial: 1 para linear, 2 para quadrático, 3 para cúbico e assim por diante.
# MAGIC 
# MAGIC DeterministicProcess, usado para criar recursos de tendência. Para usar dois períodos sazonais (semanal e anual), precisaremos instanciar um deles como um "termo adicional":
# MAGIC 
# MAGIC Um processo determinístico, a propósito, é um termo técnico para uma série temporal que não é aleatória ou completamente determinada, como as séries const e tendência. Os recursos derivados do índice de tempo geralmente serão determinísticos.

# COMMAND ----------

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=temperature.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in temperature.index

# COMMAND ----------

# MAGIC %md
# MAGIC ### Previsões
# MAGIC 
# MAGIC Com nosso conjunto de recursos criado, estamos prontos para ajustar o modelo e fazer previsões. Adicionaremos uma previsão de 360 dias para ver como nosso modelo extrapola além dos dados de treinamento

# COMMAND ----------

y = temperature["MeanSeaTemperature"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=360)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="Sea Temperature - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()

# COMMAND ----------

display(ax)

# COMMAND ----------


