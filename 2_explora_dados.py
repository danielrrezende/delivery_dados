# Databricks notebook source
# MAGIC %run ./1_coleta_limpa_dados

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4) Exploração de dados

# COMMAND ----------

# MAGIC %md
# MAGIC Tides

# COMMAND ----------

df_profile_td = ProfileReport(df_tides.toPandas(), title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_td_html = df_profile_td.to_html()

displayHTML(profile_td_html)

# COMMAND ----------

# MAGIC %md
# MAGIC Waves

# COMMAND ----------

df_profile_wv = ProfileReport(df_waves.toPandas(), title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_wv_html = df_profile_wv.to_html()

displayHTML(profile_wv_html)
