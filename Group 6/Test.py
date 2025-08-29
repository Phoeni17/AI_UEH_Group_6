from statistics import correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("global air pollution dataset.csv")

del data['Country']
del data['City']
del data['AQI Category']
del data['CO AQI Category']
del data['Ozone AQI Category']
del data['NO2 AQI Category']

data.head()

#correlation = data.corr()
#print(correlation)

sns.heatmap(correlation, annot=True)

#sns.countplot(x=data["Result"])
plt.title("Target Distribution")
plt.show()
