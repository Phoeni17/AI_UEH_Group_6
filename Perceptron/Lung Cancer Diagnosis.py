from statistics import correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("lung_cancer_examples.csv")

del data["Name"]
del data["Surname"]
data.head()

correlation = data.corr()
print(correlation)

#sns.heatmap(correlation, annot=True)

sns.countplot(x=data["Result"])
plt.title("Target Distribution")
plt.show()
