#KNN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df_knn = pd.read_csv('./iris_knn.csv')
print('Data loaded successfully')

df_knn.head()

feature_cols = ['sepal_length', 'sepal_width','petal_width']
x= df_knn[feature_cols]
y= df_knn.species

x_train,x_test,y_train,ytest = train_test_split(x,y,test_size=0.2 , random_state=42)

scaler =StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal_length',y='sepal_width',hue='species', data=df_knn,palette='bright')
plt.title('KNN Decision Boundaries')
plt.show()

#K-Means
from sklearn.cluster import KMeans

df_kmeans = pd.read_csv('./iris_kmeans.csv')
print('Data loaded successfully')

df_kmeans.head()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_kmeans)

df_kmeans['cluster']=kmeans.labels_

plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal_length',y='sepal_width',hue = 'cluster', data=df_kmeans,palette='bright')

plt.title('KMeans Clustring')
plt.show()


#actual comparision starts from here
import matplotlib.pyplot as plt
import seaborn as sns

df_knn['cluster'] = df_kmeans['cluster']
fig,axes = plt.subplots(1,2,figsize=(20,8))

sns.scatterplot(x='sepal_length',y='sepal_width',hue='species', data=df_knn, palette='bright',ax=axes[0])
axes[0].set_title('KNN Classification')

sns.scatterplot(x='sepal_length',y='sepal_width', hue='cluster',data=df_knn,palette='bright',ax=axes[1])
axes[1].set_title('KMeans Clustering')

for i,row in df_knn.iterrows():
    if row['species']!= row['cluster']:
        axes[1].annotate('X',(row['sepal_length'],row['sepal_width']),color='red',fontsize=12,weight='bold')
plt.show()

df_knn.head()
df_kmeans.tail()
