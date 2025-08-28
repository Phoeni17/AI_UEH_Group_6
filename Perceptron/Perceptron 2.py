from sklearn.linear_model import Perceptron

X = ([[-0.5, -0.5], [-0.5, 0.5], [0.3, -0.5], [-0.1, 1]])
Y = ([1, 1, 0, 0])

model = Perceptron()

model.fit(X, Y)

score = model.score(X,Y)

print(score)
