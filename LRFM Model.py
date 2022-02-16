
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
#data.head()
#data.info()
#data.isnull.sum()
#data.nunique()

# Filter Data
data = df[df.Stage.isin(["Closed Won","Closed/Won-Channel Inventory"]) & df['Shipping Country'].isin(['US','United States'])]
data['Close Date'] = pd.to_datetime(data['Close Date']) # Converting to date-time
#type(data['Close Date'][0])
data['Account: Created Date'] = pd.to_datetime(data['Account: Created Date']) # Converting to date-time
#type(data['Account: Created Date'][0])
#data.info()
data_new = data
curr_time = pd.to_datetime("now")


# Monetary = Revenue (Total price (converted)) for an Account ID

df_monetary = pd.DataFrame()
df_monetary['Monetary'] = data_new.groupby(['Account ID'])['Total Price (converted)'].sum()
df_monetary = df_monetary.reset_index()

#Checking:
#df_monetary


# Recency = Time since latest 'Close Date' for a particular Account ID

df_recency = pd.DataFrame()
#data.sort_values('Account ID')
df_recency['Recency'] = curr_time - data_new.groupby(['Account ID']).agg({'Close Date':np.max})
df_recency = df_recency.reset_index()
# Checking:
#df_recency


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


# Frequency: Number of Order IDs for an Account ID (per time dimension)

df_frequency_1 = data_new.groupby(['Account ID','Account Name']).agg(['nunique'])
df_frequency_1 = df_frequency_1.reset_index()
df_frequency_1.info()
df_frequency = df_frequency_1[['Account ID','Order Id']]
df_frequency.columns = ['Account ID', 'Frequency per FY']

type(df_frequency)
df_frequency.sort_values('Frequency per FY', ascending = False)

#Checking:
#df_frequency


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




# outlier treatment if needed

plt.boxplot(main_df.Length)
# Q1 = main_df.Length.quantile(0.25)
# Q3 = main_df.Length.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Length >= (Q1 - 1.5*IQR)) & (main_df.Length <= (Q3 + 1.5*IQR))]
# plt.boxplot(main_df.Length)


plt.boxplot(main_df.Recency)
# Q1 = main_df.Recency.quantile(0.25)
# Q3 = main_df.Recency.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Recency >= (Q1 - 1.5*IQR)) & (main_df.Recency <= (Q3 + 1.5*IQR))]
# plt.boxplot(main_df.Recency)

# In[12]:


plt.boxplot(main_df.Frequency)
# Q1 = main_df.Frequency.quantile(0.25)
# Q3 = main_df.Frequency.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Frequency >= (Q1 - 1.5*IQR)) & (main_df.Frequency <= (Q3 + 1.5*IQR))]
# plt.boxplot(main_df.Frequency)

plt.boxplot(main_df.Monetary)
# Q1 = main_df.Monetary.quantile(0.25)
# Q3 = main_df.Monetary.quantile(0.75)
# IQR = Q3 - Q1
# main_df = main_df[(main_df.Monetary >= (Q1 - 1.5*IQR)) & (main_df.Monetary <= (Q3 + 1.5*IQR))]
# plt.boxplot(main_df.Monetary)


# main_df.head()








