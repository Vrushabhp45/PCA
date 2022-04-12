import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
wine= pd.read_csv("wine.csv")
wine.describe()
wine['Type'].value_counts()
#drop Type column
Wine=wine.drop(['Type'], axis = 1)
Wine.shape
Wine.info()
# Normalizing the numerical data
from sklearn.preprocessing import scale
wine_norm = scale(Wine)
#95% of variance
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(wine_norm)
reduced = pca.transform(wine_norm)

# PCA Components matrix or convariance Matrix
pca.components_
reduced[:,0:3]
# The amount of variance that each PCA has
var = pca.explained_variance_ratio_
# Cummulative variance of each PCA
Var = np.cumsum(np.round(var,decimals= 4)*100)
plt.plot(Var,color="red");
# Final Dataframe
finalDf = pd.concat([pd.DataFrame(reduced[:,0:3],columns=['pc1','pc2','pc3']), wine[['Type']]], axis = 1)
# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=finalDf);
sns.scatterplot(data=finalDf, x='pc1', y='pc2', hue='Type');
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
#create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))
# Create Clusters
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters
y=pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()
# Adding clusters to dataset
wine2=wine.copy()
wine2['clustersid']=hclusters.labels_

from sklearn.cluster import KMeans   #Importing KMeans from sklearn and
k_inertia = []                       # Creating a list to store the kmeans.inertia_
for k in range(2,10):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(wine_norm)
    k_inertia.append(kmeans.inertia_)
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(2,10),k_inertia)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');
# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(wine_norm)
KMeans(n_clusters=3, random_state=30)
clusters3.labels_

# Assign clusters to the data set
wine3=wine.copy()
wine3['clusters3id']=clusters3.labels_
wine3['clusters3id'].value_counts()
