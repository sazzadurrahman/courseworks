
# coding: utf-8

# ### Web Scraping from Glasgow Open Data (Housing Price Data)

import os
import pandas as pd
#download the page the supplied URL using curl
webpage = os.popen('curl -k -X GET https://data.glasgow.gov.uk/dataset/glasgow-house-sales-1991-2013').read()
#print(webpage)
print('Done')


# In[49]:


#download the webpage and filter out any lines which do not include
#the words  analytics or grep Download or grep href using the unix grep command
filtered_webpage = os.popen('curl -k -X GET https://data.glasgow.gov.uk/dataset/glasgow-house-sales-1991-2013 | grep analytics |grep href').read()
#print(filtered_webpage)
print('Done')


# In[48]:


#split the filtered string using double quotes as a separator
#the result is an array 
strArray=filtered_webpage.split('"')


# In[4]:


#use a for loop to create a an array with only the URLs
urls = []
for s in strArray:
    if s.startswith('http'):
        urls.append(s)


# In[ ]:


#take the first url from the array
firt_url = urls[0]

#construct a curl command string using string 
curlString = 'curl -k -X GET ' + firt_url
#issue a curl command using the first url to download the the first set of house price data
firstHousePrice = os.popen(curlString).read()
print(firstHousePrice)


# In[13]:


text_file = firstHousePrice
lines = text_file.split('\n')


# In[31]:


attributes = firstHousePrice.split('\r\n')[0].split(',')


# In[28]:


#compose create statement
start_of_create = "CREATE TABLE IF NOT EXISTS glasgow_housing ("
for c in attributes:
    start_of_create += str(c) 
    start_of_create += ' VARCHAR(50), '
start_of_create = start_of_create[:-2]
start_of_create += ');'
start_of_create += '\n'
print(start_of_create)


# In[33]:


def insertRow(row):
  temp =''
  for c in row:
    temp += "'" + c + "'"
    temp += ', '
  temp += ''
  return temp  


# In[55]:


firstHousePrice.split('\r\n')[1]


# In[ ]:


insert = "Insert Into glasgow_housing Values "
values = firstHousePrice.split('\r\n')[1:]
for line in values:
  #create += insert
  insert +="(" + insertRow(line.split(',')) + ")"
  insert += ",  "
insert = insert[:-4]
insert += ');'
insert += '\n'
print(insert)


# In[ ]:



column_names= lines[0].split(',')
create = "CREATE TABLE IF NOT EXISTS US_KPI ("
for c in column_names:
  #c += '\t'
  create += c 
  create += ' VARCHAR(50), '
create = create[:-2]
create += ');'
create += '\n'


# #### Store to MySQL Database

# In[ ]:


#!/Python27/python
import MySQLdb
db=MySQLdb.connect(host = '127.0.0.1', 
    user = 'root', 
    passwd = '', 
    db = 'timeline', 
    port = 3306)
cursor = db.cursor()
def insertRow(row):
  temp =''
  for c in row:
    temp += "'" + c + "'"
    temp += ', '
  temp = temp[:-2]
  #print '-----'
  #print type(temp)
  temp += ''
  return temp    
text_file = open("DS_KPI_Top_10_Rows.csv", "r")
lines = text_file.read().split('\n')
column_names= lines[0].split(',')
create = "CREATE TABLE IF NOT EXISTS US_KPI ("
for c in column_names:
  #c += '\t'
  create += c 
  create += ' VARCHAR(50), '
create = create[:-2]
create += ');'
create += '\n'
#cursor.execute(create)
#db.commit()
#values_1 = lines[1].split(',')
insert = "Insert Into US_KPI Values "
#for c in values_1:
#  create += c
#  create += ', '
#create = create[:-2]
#create += ')'
values = lines[1:]
for line in values:
  #create += insert
  insert +="(" + insertRow(line.split(',')) + ")"
  insert += ",  "
insert = insert[:-4]
insert += ');'
insert += '\n'
cursor.execute(create)
cursor.execute(insert)
db.commit()
#create += insert
#create += insertRow(lines[2].split(','))
cursor.close()
#!/Python27/python
import MySQLdb
db=MySQLdb.connect(host = '127.0.0.1', 
    user = 'root', 
    passwd = '', 
    db = 'timeline', 
    port = 3306)
cursor = db.cursor()
def insertRow(row):
  temp =''
  for c in row:
    temp += "'" + c + "'"
    temp += ', '
  temp = temp[:-2]
  #print '-----'
  #print type(temp)
  temp += ''
  return temp    
text_file = open("DS_KPI_Top_10_Rows.csv", "r")
lines = text_file.read().split('\n')
column_names= lines[0].split(',')
create = "CREATE TABLE IF NOT EXISTS US_KPI ("
for c in column_names:
  #c += '\t'
  create += c 
  create += ' VARCHAR(50), '
create = create[:-2]
create += ');'
create += '\n'
#cursor.execute(create)
#db.commit()
#values_1 = lines[1].split(',')
insert = "Insert Into US_KPI Values "
#for c in values_1:
#  create += c
#  create += ', '
#create = create[:-2]
#create += ')'
values = lines[1:]
for line in values:
  #create += insert
  insert +="(" + insertRow(line.split(',')) + ")"
  insert += ",  "
insert = insert[:-4]
insert += ');'
insert += '\n'
cursor.execute(create)
cursor.execute(insert)
db.commit()
#create += insert
#create += insertRow(lines[2].split(','))
cursor.close()
db.close
#print create
#print insert
print ("Done")
text_file.close()

