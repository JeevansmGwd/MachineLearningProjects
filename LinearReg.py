import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./house_price.csv")

print(df.head())

x = df[['area','rooms']]
y = df['price']


x_train,x_test,y_train,Y_test = train_test_split(x,y, test_size = 0.2 , random_state = 1)

model =LinearRegression()

model.fit(x_train,y_train)

import joblib
joblib.dump(model,'house_price_model.pkl')

print('model saved as house_price_model.pkl')
import joblib

model = joblib.load('house_price_model.pkl')

def predict_price(area,rooms):
    prediction = model.predict([[area, rooms]])
    return prediction[0]

area = float(input ( 'Enter the area of the house:'))

rooms = int(input('Enter the number of rooms:'))


predicted_price = predict_price(area,rooms)
print('Predected Price for area',area,'and rooms',rooms,':',predicted_price)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

plt.figure(figsize=(10,6))

sns.scatterplot(x='area', y='price',data=df,hue='rooms',palette='bright',s=100)

x=df['area']. values.reshape(-1,1)
y=df['price'].values
line=LinearRegression()
line.fit(x,y)
plt.plot(x,line.predict(x),color='red',label = 'linear Regression')

plt.title('Relationship between Area and Price,colored by number of Rooms')
plt.xlabel('Area(sq ft)')
plt.ylabel('Price')
plt.legend(title='Number of Rooms')
plt.show()
