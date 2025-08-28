from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
Y = data.target
#print(X.shape)
#print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=42)

#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

model = Perceptron(max_iter=1000, eta0=0.1)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred)

conf_matrix = confusion_matrix(Y_test, Y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
                                            dispay_labels = [False, True])

cm_display.plot()
plt.show()
