#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries:
    
import pandas as pd
import numpy as np

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# To Scale our data
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# Length = Age of the customer -> ('Account:Created Date' MINUS today's date)
# Frequency = Number of Order ID's for an Account ID (per time dimension)
# Recency = Latest 'Close Date' for a particular Account ID
# Monetary = 'Total price (converted)' for an Account ID

# data = pd.read_csv("customersegmentation_data1.csv", encoding = 'ISO-8859-1')
# df = pd.read_csv('C:\\Users\\leyla.dong\\Desktop\\CP 2021.csv',encoding= 'unicode_escape')
#data.head()
#data.info()
#data.isnull.sum()
#data.nunique()

# data = df[df.Stage.isin(["Closed Won","Closed/Won-Channel Inventory"]) & df['Shipping Country'].isin(['US','United States'])]


# In[2]:


data['Close Date'] = pd.to_datetime(data['Close Date']) # Converting to date-time
#type(data['Close Date'][0])
data['Account: Created Date'] = pd.to_datetime(data['Account: Created Date']) # Converting to date-time
#type(data['Account: Created Date'][0])
#data.info()
data_new = data

curr_time = pd.to_datetime("now")


# In[3]:


# Monetary = Revenue (Total price (converted)) for an Account ID

df_monetary = pd.DataFrame()
df_monetary['Monetary'] = data_new.groupby(['Account ID'])['Total Price (converted)'].sum()
df_monetary = df_monetary.reset_index()

#Checking:
#df_monetary
#df_monetary[df_monetary.index.str.startswith('0014000000iwyyJ')]
#data[data['Account ID'] == '0014000000iwyyJ'] # This adds up to $9779 for this ID


# In[4]:


# Recency = Time since latest 'Close Date' for a particular Account ID

df_recency = pd.DataFrame()
#data.sort_values('Account ID')
df_recency['Recency'] = curr_time - data_new.groupby(['Account ID']).agg({'Close Date':np.max})
df_recency = df_recency.reset_index()
# Checking:
#df_recency
#data[data['Account ID'] == '0011W00001rJ1zQ']


# In[5]:


# Length: Age of customer

df_length_1 = pd.DataFrame()


data_new['Length'] = (curr_time - data_new['Account: Created Date'])
df_length_1 = data_new.groupby(['Account ID','Account Name','Length']).count()
df_length_1 = df_length_1.reset_index()
df_length_1[['Account ID','Length']]
#sfdc_length=df5[['Account ID','Account Name','Length']]

df_length = df_length_1[['Account ID', 'Length']]

#Checking:
#df_length


# In[6]:


df_length


# In[7]:


# Frequency: Number of Order IDs for an Account ID (per time dimension)

df_frequency_1 = data_new.groupby(['Account ID','Account Name']).agg(['nunique'])
df_frequency_1 = df_frequency_1.reset_index()
df_frequency_1.info()
df_frequency = df_frequency_1[['Account ID','Order Id']]
df_frequency.columns = ['Account ID', 'Frequency per FY']

type(df_frequency)
df_frequency.sort_values('Frequency per FY', ascending = False)

#Checking:
df_frequency


# In[8]:


# Merging DFs for analysis:
# Create DF with unique account IDs
main_df = pd.DataFrame()
main_df['Account ID'] = df_length['Account ID']
main_df['Length'] = df_length['Length']
main_df['Recency'] = df_recency['Recency']
main_df['Frequency'] = df_frequency['Frequency per FY']
main_df['Monetary'] = df_monetary['Monetary']
main_df
main_df.info()


# In[9]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[10]:


# outlier treatment for Amount
plt.boxplot(main_df.Length)
Q1 = main_df.Length.quantile(0.25)
Q3 = main_df.Length.quantile(0.75)
IQR = Q3 - Q1
#main_df = main_df[(main_df.Length >= (Q1 - 1.5*IQR)) & (main_df.Length <= (Q3 + 1.5*IQR))]
# plt.boxplot(main_df.Length)


# In[11]:


plt.boxplot(main_df.Recency)
# Q1 = main_df.Recency.quantile(0.25)
# Q3 = main_df.Recency.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Recency >= (Q1 - 1.5*IQR)) & (main_df.Recency <= (Q3 + 1.5*IQR))]


# In[12]:


plt.boxplot(main_df.Frequency)
# Q1 = main_df.Frequency.quantile(0.25)
# Q3 = main_df.Frequency.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Frequency >= (Q1 - 1.5*IQR)) & (main_df.Frequency <= (Q3 + 1.5*IQR))]


# In[13]:


plt.boxplot(main_df.Monetary)
# Q1 = main_df.Monetary.quantile(0.25)
# Q3 = main_df.Monetary.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Monetary >= (Q1 - 1.5*IQR)) & (main_df.Monetary <= (Q3 + 1.5*IQR))]

# plt.boxplot(main_df.Monetary)
# main_df


# In[14]:


# standardise all parameters

main_df_norm1 = main_df.drop(["Account ID"], axis=1)
main_df_norm1.Length = main_df_norm1.Length.dt.days
main_df_norm1.Recency = main_df_norm1.Recency.dt.days

standard_scaler = StandardScaler()
main_df_norm1 = standard_scaler.fit_transform(main_df_norm1)

main_df_norm1 = pd.DataFrame(main_df_norm1)
main_df_norm1.columns = ['Length', 'Recency', 'Frequency', 'Monetary']
main_df_norm1


# In[15]:


# Hopkins statistic

hopkins(main_df_norm1)


# In[16]:


# Kmeans with K=5
model_clus5 = KMeans(n_clusters = 5, max_iter = 50)

model_clus5.fit(main_df_norm1)


# In[17]:


# Silhouette score

# Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.
# 1: Means clusters are well apart from each other and clearly distinguished.
# 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
# -1: Means clusters are assigned in the wrong way.

from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(main_df_norm1)
    sse_.append([k, silhouette_score(main_df_norm1, kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# In[18]:


# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(main_df_norm1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)


# In[19]:


main_df.index = pd.RangeIndex(len(main_df.index))
LRFM_km = pd.concat([main_df, pd.Series(model_clus5.labels_)], axis=1)
LRFM_km.columns = ['Account ID', 'Length', 'Recency', 'Frequency', 'Monetary', 'ClusterID']

LRFM_km.Recency = LRFM_km.Recency.dt.days
LRFM_km.Recency = LRFM_km.Length.dt.days
km_clusters_length = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Length.mean())
km_clusters_recency = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Recency.mean())
km_clusters_frequency = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_monetary = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Monetary.mean())


# In[20]:


km_clusters_count = pd.DataFrame(LRFM_km.groupby(['ClusterID'])['Account ID'].count())
km_clusters_count


# In[21]:


LRFM_mean = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_length, km_clusters_recency, km_clusters_frequency, km_clusters_monetary], axis=1)
LRFM_mean.columns = ["ClusterID", "Length_mean", "Recency_mean", "Frequency_mean", "Monetary_mean"]
LRFM_mean.info()
LRFM_mean


# In[22]:


sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Length_mean/ np.timedelta64(1, 'D'))


# In[23]:


sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Recency_mean)


# In[24]:


sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Frequency_mean)


# In[25]:


sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Monetary_mean)


# In[ ]:





# In[ ]:




