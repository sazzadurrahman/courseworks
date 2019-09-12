
# coding: utf-8


# File location and type in Databricks on Aaamazon S3
file_location = "/FileStore/tables/climate_csv.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[2]:


from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, when, lit, concat, round, sum


# In[3]:


df.count()


# In[4]:


df.registerTempTable("climate")


# In[6]:


get_ipython().run_line_magic('sql', '')
SELECT  country_name, y2004, y2014 FROM climate B WHERE B.indicator_name='Renewable energy consumption (% of total final energy consumption)' and y2014<y2004  order by y2004 desc Limit 10


# In[7]:


get_ipython().run_line_magic('sql', '')
SELECT  country_name, y2014 FROM climate B WHERE B.indicator_name='CO2 emissions (metric tons per capita)' order by y2014 desc Limit 10


# In[8]:


get_ipython().run_line_magic('sql', '')
SELECT  country_name, y2004, y2014, round(y2014-y2004,2) as difference FROM climate  WHERE indicator_name='CO2 emissions (metric tons per capita)' and y2014-y2004>0 order by difference desc Limit 10


# In[9]:


get_ipython().run_line_magic('sql', '')
SELECT  country_name, y2000, y2014 FROM climate  WHERE indicator_name='Electricity production from nuclear sources (% of total)' order by y2014-y2004 desc Limit 10


# In[10]:


df1=spark.sql("select country_name, round(y2000+y2014,2) as new2014 from climate where indicator_name='Electricity production from nuclear sources (% of total)' order by new2014 desc").limit(10)


# In[11]:


display(df1)


# In[12]:


get_ipython().run_line_magic('sql', '')
SELECT  country_name, y2000, y2014 FROM climate  WHERE indicator_name='Electricity production from nuclear sources (% of total)' order by y2014-y2004 desc Limit 10

