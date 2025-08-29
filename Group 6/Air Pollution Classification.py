import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X = np.array([
    [20, 400, 50],
    [35, 420, 55],
    [15, 380, 45],
    [22, 410, 50],
    [30, 430, 52],
    [40, 460, 55],
    [18, 400, 48],
    [28, 450, 53],
    [35, 470, 54],
    [42, 490, 56],
    [25, 420, 51],
    [38, 480, 57],
    [45, 460, 55],
    [50, 480, 56],#
    [55, 600, 60],
    [80, 650, 65],
    [55, 520, 60],
    [62, 580, 62],
    [70, 600, 65],
    [80, 650, 66],
    [75, 630, 64],
    [68, 610, 63],
    [72, 640, 65],
    [85, 670, 67],
    [60, 590, 61],
    [78, 660, 66],
    [65, 620, 63],
    [90, 700, 68],#
    [110, 780, 70],
    [120, 800, 72],
    [130, 850, 74],
    [140, 900, 75],
    [150, 920, 77],
    [160, 950, 78],
    [170, 980, 79],
    [180, 1000, 80],
    [190, 1050, 82],
    [200, 1100, 83],
    [115, 810, 72],
    [135, 860, 74],
    [120, 800, 70],
    [150, 900, 75],
    [200, 1000, 80]
])

Y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])

# Chuẩn hóa dữ liệu đầu vào để Perceptron học tốt hơn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y,
                                                    test_size=0.3,random_state=42)

# Khởi tạo mô hình Perceptron
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(classification_report(Y_test, Y_pred, target_names=['Good', 'Average','Hazardous']))

cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good', 'Average','Hazardous'],
            yticklabels=['Good', 'Average','Hazardous'])
plt.title("Air Pollution Classification")
plt.xlabel("Du doan")
plt.ylabel("Thuc te")
plt.show()
