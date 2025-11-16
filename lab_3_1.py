from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def dichotomy_method(
    left_: int | float, right_: int | float, func: callable, eps: float, N: int = 1000
) -> list[float | int | Any]:
    """Метод Дихотомии"""
    count = 0
    mid = 0
    while abs(right_ - left_) >= 2 * eps and count < N:
        mid = (left_ + right_) / 2
        if func.subs(x, left_).evalf() * func.subs(x, mid).evalf() < 0:
            right_ = mid
        elif func.subs(x, left_).evalf() * func.subs(x, mid).evalf() > 0:
            left_ = mid
        else:
            break
        count += 1
    return [mid, count]


def chord_method(
    left_: int | float, right_: int | float, func: callable, eps: float, N: int = 1000
) -> list[int | float | Any]:
    """Метод Хорд"""
    count = 0
    x_0 = left_
    x_1 = right_
    x_2 = x_1 - (func.subs(x, x_1).evalf() * (x_1 - x_0)) / (
        func.subs(x, x_1).evalf() - func.subs(x, x_0).evalf()
    )
    while abs(func.subs(x, x_2).evalf()) >= eps and count < N:
        if func.subs(x, x_0).evalf() * func.subs(x, x_2).evalf() < 0:
            x_1 = x_2
        else:
            x_0 = x_2
        x_2 = x_1 - (func.subs(x, x_1).evalf() * (x_1 - x_0)) / (
            func.subs(x, x_1).evalf() - func.subs(x, x_0).evalf()
        )
        count += 1
    return [x_2, count]


def secant_method(
    left_: int | float, right_: int | float, func: callable, eps: float, N: int = 1000
) -> list[int | float | Any]:
    """Метод Секущей"""
    x_0 = left_
    x_1 = right_
    count = 0
    x_2 = x_1 - (func.subs(x, x_1).evalf() * (x_1 - x_0)) / (
        func.subs(x, x_1).evalf() - func.subs(x, x_0).evalf()
    )
    while abs(x_2 - x_1) >= eps and count < N:
        x_0 = x_1
        x_1 = x_2
        x_2 = x_1 - (func.subs(x, x_1).evalf() * (x_1 - x_0)) / (
            func.subs(x, x_1).evalf() - func.subs(x, x_0).evalf()
        )
        count += 1
    return [x_2, count]


def newton_method(
    left_: int | float, right_: int | float, func: callable, eps: float, N: int = 1000
) -> list[int | float | Any]:
    """Метод Ньютона"""
    count = 0
    x_0 = left_
    first_diff = sp.diff(func, x, 1)

    x_0 = (
        left_
        if first_diff.subs(x, left_).evalf() * func.subs(x, x_0).evalf() > 0
        else right_
    )
    x_1 = x_0 - func.subs(x, x_0).evalf() / first_diff.subs(x, x_0).evalf()

    while abs(x_1 - x_0) >= eps and count < N:
        x_0 = x_1
        x_1 = x_0 - func.subs(x, x_0).evalf() / first_diff.subs(x, x_0).evalf()
        count += 1
    return [x_1, count]


FUNC_DICT = {
    secant_method: "Метод Секущей",
    dichotomy_method: "Метод Дихотомии",
    chord_method: "Метод Хорд",
    newton_method: "Метод Ньютона",
}

# class 'sympy.core.numbers.NaN
if __name__ == "__main__":
    x = sp.symbols("x")
    """
    Корни: 0 и ~1.256 (это -0.000652)
    EPS = 1e-5
    left = -1
    right = 2
    """
    EPS = 1 * 10 ** -(abs(int(input("Введите точность EPS: "))))
    left = float(input("Введите значение левой границы: "))
    right = float(input("Введите значение правой границы: "))

    # EPS = 1e-5
    # left = -1
    # right = 2
    my_func = sp.exp(x) - 2 * x - 1
    sp.pprint(my_func)
    roots = [sp.nsolve(my_func, x, 0), sp.nsolve(my_func, x, 1)]
    print(f"Корни = {roots}")
    x_vals = np.linspace(left, right, 1000)
    y_vals = [my_func.subs(x, val).evalf() for val in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, "b-", linewidth=2, label=f"{my_func}")
    plt.grid(True)
    plt.legend()
    plt.title("График функции")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    print(f"{'Метод':<30} {'Нач.приближение':<25} {'f(x*)':<10} {'Итерации'}")
    for i in list(FUNC_DICT.keys()):
        result = i(left, right, my_func, EPS)
        if isinstance(result[0], sp.core.numbers.NaN) or result[1] == 1000:
            result[1] = -1001

        if FUNC_DICT[i] in ["Метод Хорд", "Метод Секущей"]:
            print(
                f"{FUNC_DICT[i]:<30} x_0={left}, x_1={right:<10} {result[0]:<10.4e} {result[1]:>10}"
            )
        else:
            x_0 = (left + right) / 2 if i == dichotomy_method else left
            print(
                f"{FUNC_DICT[i]:<30} x_0={x_0:<18} {result[0]:<10.4e} {result[1]:>10}"
            )

# 32 27
