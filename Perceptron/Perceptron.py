import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

#Qua trinh bien dich du lieu
model = Perceptron(max_iter=100, eta0=0.1, random_state=42) #max_iter: so lan hoc; eta0: toc do hoc; random_state: lay du lieu

model.fit(X,Y) #fit: cho may hoc

X1 = np.array([[0.5, 9], [0.3, -2],[-2, -10], [0.5, 0.8]])
Y_pred = model.predict(X1)

print(Y_pred)

score = model.score(X1)

print("Ket qua cong AND va ??: ")
for inputs in X:
    output = model.predict([inputs])
    print(f"Input: {inputs} -> Output: {output[0]} ")


