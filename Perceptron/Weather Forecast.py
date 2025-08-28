import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [1,1,0],
    [1,0,1],
    [0,1,1],
    [1,1,1],
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [0,0,1]
])

Y = np.array([0,1,0,0,1,0,1,1])

X_train, X_test, Y_train, Y_Test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=42)
#
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(X_test)
print(Y_pred)

accuracy = accuracy_score(Y_Test, Y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

new_weather = np.array([[1, 1, 0]])

prediction = model.predict(new_weather)
if prediction[0] == 1:
    print("Ngay mai troi nang!")
else:
    print("Ngay mai troi mua!")

