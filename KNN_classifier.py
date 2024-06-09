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
