# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/FornecedoresArea.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df_FornecedoresArea = spark.read.format(file_type) \
               .option("inferSchema", infer_schema) \
               .option("header", first_row_is_header) \
               .option("sep", delimiter) \
               .load(file_location)\
               .drop('NOME_UNICO')

display(df_FornecedoresArea)
df_FornecedoresArea.count()

# COMMAND ----------

# File location and type
file_location1 = "/FileStore/tables/Comissao.csv"
file_type1 = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df_comissao = spark.read.format(file_type1) \
               .option("inferSchema", infer_schema) \
               .option("header", first_row_is_header) \
               .option("sep", delimiter) \
               .load(file_location1)

display(df_comissao)
df_comissao.count()

# COMMAND ----------

# File location and type
file_location2 = "/FileStore/tables/Faturamento.csv"
file_type2 = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df_faturamento = spark.read.format(file_type2) \
                           .option("inferSchema", infer_schema) \
                           .option("header", first_row_is_header) \
                           .option("sep", delimiter) \
                           .load(file_location2)

display(df_faturamento)
df_faturamento.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Faturamento x FornecedoresArea

# COMMAND ----------

fat_for = df_faturamento.join(
                              df_FornecedoresArea,
                              how='inner',
                              on=['RAIZ_CNPJ']
                             )\
                             .drop_duplicates()

display(fat_for)
fat_for.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Comissao x FornecedoresArea

# COMMAND ----------

com_for = df_comissao.join(
                            df_FornecedoresArea,
                            how='inner',
                            on=['RAIZ_CNPJ']
                          )\
                          .drop_duplicates()

display(com_for)
com_for.count()
