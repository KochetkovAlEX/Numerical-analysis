import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

n = 100
x = np.linspace(1.0, np.e, n + 1)
h = (np.e - 1.0) / n

A = np.zeros(n + 1)
B = np.zeros(n + 1)
C = np.zeros(n + 1)
F = np.zeros(n + 1)

for i in range(1, n):
    A[i] = 1 - h / (2 * x[i])
    B[i] = 2 + h**2 / x[i] ** 2
    C[i] = 1 + h / (2 * x[i])
    F[i] = h**2 * np.log(x[i]) / x[i] ** 2

B[0] = -1
C[0] = 0
F[0] = 2


A[n] = np.e
B[n] = np.e - h
F[n] = 0

K = np.zeros(n + 1)
L = np.zeros(n + 1)
K[0] = C[0] / B[0]
L[0] = -F[0] / B[0]

for i in range(1, n):
    K[i] = -C[i] / (A[i] * K[i - 1] - B[i])
    L[i] = (F[i] - A[i] * L[i - 1]) / (A[i] * K[i - 1] - B[i])

y = np.zeros(n + 1)

y[n] = (F[n] - A[n] * L[n - 1]) / (A[n] * K[n - 1] - B[n])

for i in range(n - 1, -1, -1):
    y[i] = K[i] * y[i + 1] + L[i]

print(f"{'i':>3} {'x[i]':>8} {'Точное':>10} {'М.прогонки':>15}")
for i in range(n + 1):
    print(f"{i:>3} {x[i]:8.4f} {2 * x[i] - np.log(x[i]):10.4f} {y[i]:12.4f}")


plt.figure(figsize=(12, 8))
plt.plot(x, 2 * x - np.log(x), "r", linewidth=5, label="Точное решение")
plt.plot(x, y, "g*", label="М. Прогонки")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
