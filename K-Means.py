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
