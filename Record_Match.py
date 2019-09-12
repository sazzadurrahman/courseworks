
# coding: utf-8

# ### Record Matching

import os
import recordlinkage as rl
import pandas as pd
from recordlinkage.base import BaseIndex, BaseCompare
from recordlinkage.index import Full, Block, SortedNeighbourhood, Random
from recordlinkage.compare import Exact, String, Numeric, Geographic, Date


# In[15]:


df_a = pd.read_csv('listings.csv')
df_b = pd.read_csv('listings.csv')


# In[18]:


df_a.head()


# In[21]:


df_a_clean=df_a.drop(['neighbourhood_group', 'latitude', 'longitude', 'number_of_reviews','last_review','reviews_per_month','calculated_host_listings_count','availability_365'], axis=1)
df_b_clean=df_b.drop(['neighbourhood_group', 'latitude', 'longitude', 'number_of_reviews','last_review','reviews_per_month','calculated_host_listings_count','availability_365'], axis=1)


# In[20]:


df_a_clean


# In[27]:


df_a_clean.shape


# In[22]:


df_a_clean.head()


# In[62]:


df_a_clean.dtypes


# In[16]:


indexer = rl.Index()


# In[23]:


indexer.block('id')
candidate_links = indexer.index(df_a_clean, df_b_clean)


# In[61]:


compare = recordlinkage.Compare()

compare.string('name', 'name', method='jarowinkler', threshold=0.85)
compare.exact('host_id', 'host_id')
compare.string('host_name', 'host_name')
compare.string('neighbourhood', 'neighbourhood', method='damerau_levenshtein', threshold=0.7)
compare.string('room_type', 'room_type', method='damerau_levenshtein', threshold=0.7)
compare.exact('price', 'price')
compare.exact('minimum_nights', 'minimum_nights')


# In[25]:


compare_vectors = compare.compute(candidate_links, df_a_clean, df_b_clean)


# In[ ]:


compare_vectors


# In[29]:


ecm = rl.BernoulliEMClassifier()
ecm.fit_predict(compare_vectors)


# In[38]:


from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3, load_febrl4


# In[45]:


df1=load_febrl1()
df2=load_febrl2()
df3=load_febrl3()
df4=load_febrl4()


# In[32]:


df1.head()


# In[34]:


df2.head()


# In[ ]:


df2.shape


# In[40]:


df3


# In[43]:


df1.isnull().sum()


# In[52]:


df1['given_name'].fillna('missing', inplace=True)
df1['surname'].fillna('missing', inplace=True)
df1['street_number'].fillna(0, inplace=True)
df1['address_1'].fillna('missing', inplace=True)
df1['address_2'].fillna('missing', inplace=True)
df1['suburb'].fillna('missing', inplace=True)
df1['postcode'].fillna(0, inplace=True)
df1['state'].fillna('missing', inplace=True)
df1['date_of_birth'].fillna(0, inplace=True)


# In[53]:


df1.isnull().sum()


# In[54]:


df2.isnull().sum()


# In[55]:


df2['given_name'].fillna('missing', inplace=True)
df2['surname'].fillna('missing', inplace=True)
df2['street_number'].fillna(0, inplace=True)
df2['address_1'].fillna('missing', inplace=True)
df2['address_2'].fillna('missing', inplace=True)
df2['suburb'].fillna('missing', inplace=True)
df2['state'].fillna('missing', inplace=True)
df2['date_of_birth'].fillna(0, inplace=True)


# In[56]:


df2.isnull().sum()


# In[57]:


indexer = rl.Index()


# In[59]:


indexer.block('street_number')
candidate_links = indexer.index(df1, df2)


# In[63]:


compare = recordlinkage.Compare()

compare.string('given_name', 'given_name', method='jarowinkler', threshold=0.85)
compare.string('surname', 'surname',  method='jarowinkler', threshold=0.85)
compare.exact('street_number', 'street_number')
compare.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.7)
compare.string('address_2', 'address_2', method='damerau_levenshtein', threshold=0.7)
compare.string('suburb', 'suburb', method='damerau_levenshtein', threshold=0.7)
compare.exact('postcode', 'postcode')
compare.string('state', 'state', method='damerau_levenshtein', threshold=0.7)
compare.exact('date_of_birth', 'date_of_birth')
compare.exact('soc_sec_id', 'soc_sec_id')


# In[64]:


compare_vectors = compare.compute(candidate_links, df1, df2)


# In[69]:


compare_vectors.describe().T


# In[71]:


compare_vectors.shape


# In[89]:


compare_vectors.dtypes


# In[99]:


tuples = [tuple(x) for x in compare_vectors.values]


# In[81]:


compare_vectors_train=compare_vectors[0:14000]


# In[83]:


compare_vectors_train.shape


# In[82]:


compare_vectors_train_index=compare_vectors_train.index


# In[84]:


compare_vectors_train_index.shape


# In[ ]:


compare_vectors_train_index


# In[105]:


compare_vector_matches_index = compare_vectors_train_index &  tuples


# In[ ]:


svm = rl.SVMClassifier()
svm.fit(tuples, compare_vector_matches_index)


# In[109]:


df1_1=load_febrl1()
df2_2=load_febrl2()
df3_3=load_febrl3()
df4_4=load_febrl4()


# In[111]:


df1_1.isnull().sum()


# In[122]:


df1_1.shape


# In[112]:


df1=df1_1.dropna()


# In[116]:


df1.shape


# In[118]:


df2_2.isnull().sum()


# In[121]:


df2_2.shape


# In[119]:


df2=df2_2.dropna()


# In[120]:


df2.shape


# In[123]:


df3_3.isnull().sum()


# In[124]:


df3_3.shape


# In[125]:


df3=df3_3.dropna()


# In[126]:


df3.shape


# In[129]:


indexer.block('street_number')
candidate_links = indexer.index(df1, df2)


# In[130]:


compare = recordlinkage.Compare()

compare.string('given_name', 'given_name', method='jarowinkler', threshold=0.85)
compare.string('surname', 'surname',  method='jarowinkler', threshold=0.85)
compare.exact('street_number', 'street_number')
compare.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.7)
compare.string('address_2', 'address_2', method='damerau_levenshtein', threshold=0.7)
compare.string('suburb', 'suburb', method='damerau_levenshtein', threshold=0.7)
compare.exact('postcode', 'postcode')
compare.string('state', 'state', method='damerau_levenshtein', threshold=0.7)
compare.exact('date_of_birth', 'date_of_birth')
compare.exact('soc_sec_id', 'soc_sec_id')


# In[131]:


compare_vectors = compare.compute(candidate_links, df1, df2)


# In[132]:


compare_vectors.describe().T


# In[133]:


compare_vectors.describe()


# In[135]:


compare_vectors.mean()


# In[136]:


df3.head()


# In[137]:


df1.head()


# In[138]:


frames=[df1,df2]


# In[139]:


newdf=pd.concat(frames)


# In[145]:


newdf.head()


# In[165]:


indexer = rl.Index()
indexer.add(SortedNeighbourhood())


# In[185]:


indexer = rl.index.SortedNeighbourhood('street_number','soc_sec_id', window=3)


# In[186]:


df_index=newdf.set_index('street_number','soc_sec_id')


# In[188]:


df_index = newdf.set_index(newdf.groupby(level=0).cumcount(), append=True)


# In[189]:


df_index.index.is_unique


# In[206]:


df_index.loc[df_index['soc_sec_id'] == '7364009']


# In[208]:


len(newdf)


# In[209]:


len(df_index)


# In[210]:


newdf.loc[newdf['soc_sec_id'] == '7364009']


# In[213]:


df1.loc[df1['soc_sec_id'] == '7364009']


# In[214]:


df2.loc[df2['soc_sec_id'] == '7364009']


# In[181]:


indexer = rl.Index()


# In[215]:


candidate_links = indexer.index(df1, df_index)


# In[157]:


newdf.reset_index()


# In[242]:


compare = recordlinkage.Compare()

compare.string('given_name', 'given_name', method='jarowinkler', threshold=0.85, label='given_name')
compare.string('surname', 'surname',  method='jarowinkler', threshold=0.85, label='surname')
compare.exact('street_number', 'street_number', label='street_number')
compare.string('address_1', 'address_1', method='damerau_levenshtein', threshold=0.7, label='address_1')
compare.string('address_2', 'address_2', method='damerau_levenshtein', threshold=0.7, label='address_2')
compare.string('suburb', 'suburb', method='damerau_levenshtein', threshold=0.7, label='suburb')
compare.exact('postcode', 'postcode', label='postcode')
compare.string('state', 'state', method='damerau_levenshtein', threshold=0.7, label='state')
compare.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare.exact('soc_sec_id', 'soc_sec_id', label='soc_sec_id')


# In[244]:


compare_vectors = compare.compute(candidate_links, df1, df_index)


# In[245]:


compare_vectors


# In[246]:


#Sum the comparison results
compare_vectors.sum(axis=1).value_counts().sort_index(ascending=False)


# In[247]:


len(compare_vectors[compare_vectors.sum(axis=1) ==1])


# In[239]:


newdf

