import numpy as np
import scipy
import math
from matplotlib import pylab as plt
%matplotlib inline

def f(x):
     return (math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2))

A1 = np.array([[1, 1], [1, 15]])
b1 = np.array([f(1.), f(15.)])

X1 = scipy.linalg.solve(A1, b1)
print X1

xx = np.arange(1, 15.1, 0.1)
y = X1[0] + X1[1]*xx
fx = np.array([f(x) for x in xx])

plt.figure(1)
plt.plot(xx, fx)

plt.figure(2)
plt.plot(xx, y)

plt.show()

A2 = np.array([[1, 1, 1], [1, 8, 64], [1, 15, 225]])
b2 = np.array([f(1.), f(8.), f(15.)])

X2 = scipy.linalg.solve(A2, b2)
print X2

xx = np.arange(1, 15.1, 0.1)
y = X2[0] + X2[1]*xx + X2[2]*xx*xx
fx = np.array([f(x) for x in xx])

plt.figure(1)
plt.plot(xx, fx)

plt.figure(2)
plt.plot(xx, y)

plt.show()

A3 = np.array([[1, 1, 1, 1], [1, 4, 16, 64], [1, 10, 100, 1000], [1, 15, 15*15, 15*15*15]])
b3 = np.array([f(1.), f(4.), f(10.), f(15.)])

X3 = scipy.linalg.solve(A3, b3)
print X3

xx = np.arange(1, 15.1, 0.1)
y = X3[0] + X3[1]*xx + X3[2]*xx*xx + X3[3]*xx*xx*xx
fx = np.array([f(x) for x in xx])

plt.figure(1)
plt.plot(xx, fx)

plt.figure(2)
plt.plot(xx, y)

plt.show()
