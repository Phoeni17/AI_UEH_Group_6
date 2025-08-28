from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv("lung_cancer_examples.csv")

X = data.loc[:, ['Age', 'Smokes', 'AreaQ', 'Alkhol']]
Y = data.loc[:, ['Result']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model = Perceptron(max_iter=1000, eta0=0.05, random_state=0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(f"Do chinh xac ma hinh chua scale: {accuracy_score(Y_test, Y_pred)}")

model1 = Perceptron(max_iter=1000, eta0=0.05, random_state=0)
model1.fit(X_train_std, Y_train)
Y_pred_std = model1.predict(X_test_std)
print(f"Do chinh xac da scale: {accuracy_score(Y_test, Y_pred_std)}")
