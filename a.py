from cProfile import label

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
x = np.linspace(-10, 10, 1000)
y = 2 * x ** 5  + 10 * x ** 4  - 7 * x ** 3  - 200 * x ** 2  - 200 * x + 200
y2 = 20 * x ** 4  - 5 * x ** 3  - 150 * x ** 2  - 140 * x + 100

plt.subplot(1, 2, 1)
plt.plot(x, y,"m-.", label= "PT bac 5")
plt.title("pt bac 5")
plt.xlim(-5, 5)
plt.ylim(-2000, 5000)
plt.xlabel("Truc x")
plt.ylabel("Truc y")
plt.legend(loc=4)
plt.grid

plt.subplot(1, 2, 2)
plt.plot(x, y2,"m-", label= "PT bac 4")
plt.title("pt bac 4")
plt.xlim(-5, 5)
plt.ylim(-2000, 10000)
plt.xlabel("Truc x")
plt.ylabel("Truc y")
plt.legend(loc=4)
plt.grid

plt.show()
