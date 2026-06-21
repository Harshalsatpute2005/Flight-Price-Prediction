import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("data3.csv")
data.rename(columns={'Unnamed: 0' : 'id'}, inplace=True)
data['id'] = data['id'].fillna(4)
print(data.head())
print(data.describe())



print(data.isnull().sum())


data.drop(['id', 'flight'], axis=1, inplace=True)

X = data[['airline',
          'source_city',
          'destination_city',
          'stops',
          'class']]

X = pd.get_dummies(X, drop_first=True)

y = data['price']


airline=str(input("Enter the Airline name : "))
source = str(input("Enter the source city name :"))
dest = str(input("Enter the Destination city name :"))
stop = int(input("Enter the number of stops :"))
clas = str(input("Enter the class (Economy / Business) :"))

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state = 42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

input_dict = {
    'airline': airline,
    'source_city': source,
    'destination_city': dest,
    'stops': stop,
    'class': clas
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(input_df)

print("Predicted price is:", predicted_price[0])