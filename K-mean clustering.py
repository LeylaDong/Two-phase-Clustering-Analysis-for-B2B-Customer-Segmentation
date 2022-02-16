from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
# Hopkins Analysis

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


# standardise all parameters

main_df_norm1 = main_df.drop(["Account ID"], axis=1)
main_df_norm1.Length = main_df_norm1.Length.dt.days
main_df_norm1.Recency = main_df_norm1.Recency.dt.days

standard_scaler = StandardScaler()
main_df_norm1 = standard_scaler.fit_transform(main_df_norm1)

main_df_norm1 = pd.DataFrame(main_df_norm1)
main_df_norm1.columns = ['Length', 'Recency', 'Frequency', 'Monetary']
main_df_norm1



# Hopkins statistic

hopkins(main_df_norm1)

# Kmeans with K=5
model_clus5 = KMeans(n_clusters = 5, max_iter = 50)

model_clus5.fit(main_df_norm1)


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



# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(main_df_norm1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)




main_df.index = pd.RangeIndex(len(main_df.index))
LRFM_km = pd.concat([main_df, pd.Series(model_clus5.labels_)], axis=1)
LRFM_km.columns = ['Account ID', 'Length', 'Recency', 'Frequency', 'Monetary', 'ClusterID']

LRFM_km.Recency = LRFM_km.Recency.dt.days
LRFM_km.Recency = LRFM_km.Length.dt.days
km_clusters_length = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Length.mean())
km_clusters_recency = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Recency.mean())
km_clusters_frequency = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_monetary = pd.DataFrame(LRFM_km.groupby(["ClusterID"]).Monetary.mean())




km_clusters_count = pd.DataFrame(LRFM_km.groupby(['ClusterID'])['Account ID'].count())
km_clusters_count



LRFM_mean = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_length, km_clusters_recency, km_clusters_frequency, km_clusters_monetary], axis=1)
LRFM_mean.columns = ["ClusterID", "Length_mean", "Recency_mean", "Frequency_mean", "Monetary_mean"]
LRFM_mean.info()
LRFM_mean


# Plot every cluster regarding 4 attibutes

sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Length_mean/ np.timedelta64(1, 'D'))

sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Recency_mean)

sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Frequency_mean)

sns.barplot(x=LRFM_mean.ClusterID, y=LRFM_mean.Monetary_mean)
