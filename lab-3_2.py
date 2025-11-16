from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.pyplot import figure


def simple_iter_method(
    left_: int | float, right_: int | float, func: callable, eps: float, N: int = 1000
) -> list[int | Any]:
    func_diff = sp.diff(func, x, 1)
    func_diff = sp.lambdify(x, func_diff, "numpy")
    func = sp.lambdify(x, func, "numpy")

    x_test = np.linspace(left_, right_, 1000)
    values_for_q = np.abs(func_diff(x_test))
    q = np.max(values_for_q)

    print(f"Константа q = {q:.6f}")

    count = 0
    x_0 = (left_ + right_) / 2
    border = (1 - q) * eps / q
    x_1 = func(x_0)

    while abs(x_1 - x_0) >= border and count < N:
        x_0 = x_1
        x_1 = func(x_0)
        count += 1
    return [x_1, count]


"""
exp(-x^2) - x = 0
x = exp(-x^2)
phi = exp(-x^2)
"""
x = sp.symbols("x")
my_func = sp.exp(-(x**2)) - x
phi = sp.exp(-(x**2))
phi_diff = sp.diff(phi, x, 1)
print(f"{sp.nsolve(my_func, x, 0):.4e}")
# EPS = 1 * 10 ** -(abs(int(input("Введите точность EPS: "))))
# left = float(input("Введите значение левой границы: "))
# right = float(input("Введите значение правой границы: "))
EPS = 1e-5
left = -10
right = 10
result = simple_iter_method(left, right, phi, EPS)


if isinstance(result[0], sp.core.numbers.NaN) or result[1] == 1000:
    print("Корней тут нет")
else:
    print(
        f"Значение x* = {result[0]:.4e}\n"
        f"Значение функции = {phi.subs(x, result[0]).evalf():.4e}\n"
        f"Итерации = {result[1]}"
    )


x_vals = np.linspace(left, right, 1000)
y_vals_1 = [my_func.subs(x, val).evalf() for val in x_vals]
y_vals_2 = [phi_diff.subs(x, val).evalf() for val in x_vals]

plt.figure()
plt.subplot(211)
plt.plot(x_vals, y_vals_1, "b-", linewidth=2, label=f"{my_func}")
plt.grid(True)
plt.legend()
plt.title("Исходная функция")

plt.subplot(212)
plt.plot(x_vals, y_vals_2, "g--", linewidth=2, label=f"{phi_diff}")
plt.grid(True)
plt.legend()
plt.title("Производная от ФИ")
plt.show()
