import random

import numpy as np
import sympy as sp
from scipy.special import lambertw

# yy'/x + exp(y) = 0, y(1) = 0, [1; 1,6]
x = sp.Symbol("x")
y = sp.Function("y")(x)


def euler_method(y_diff_np, y0, x_values, left, right, n=100):
    step = (right - left) / n

    y_values = np.zeros(n + 1)
    y_values[0] = y0  # покажи, что при других значениях ужас. при 0.0 массив из -inf
    for i in range(n):
        y_values[i + 1] = y_values[i] + step * y_diff_np(x_values[i], y_values[i])
    return y_values


def shy_euler_method(y_diff_np, y0, x_values, left, right, n=100):
    step = (right - left) / n

    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        y_predict = y_values[i] + step * y_diff_np(x_values[i], y_values[i])
        y_values[i + 1] = y_values[i] + step * y_diff_np(x_values[i + 1], y_predict)
    return y_values


def heun_method(y_diff_np, y0, x_values, left, right, n=100):
    step = (right - left) / n

    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        y_predict = y_values[i] + step * y_diff_np(x_values[i], y_values[i])
        y_values[i + 1] = y_values[i] + step / 2 * (
            y_diff_np(x_values[i + 1], y_predict) + y_diff_np(x_values[i], y_values[i])
        )

    return y_values


def runge_kutta_method(y_diff_np, y0, x_values, left, right, n=100):
    """
    Дано:
    p2 = p3 = 5/6

    Система:
    p3 * α3 + p2 * α2 = 1/2
    p1 + p2 + p3 = 1
    α2 = β21
    α3 = β31 + β32
    α3(α3 - α2) + β32α2(2 - 3*α2) = 0
    p3 * β32 * α2 = 1/6


    Результат вычислений:
    p1 = -4/6
    α2 = β21 = (3 + sqrt(11)) /10
    α3 = (3 - sqrt(11)) /10
    β32 = -(3 - sqrt(11))
    β31 = 11(3-sqrt(11))/10
    """
    alpha2 = betta21 = (3 + np.sqrt(11)) / 10
    alpha3 = (3 - np.sqrt(11)) / 10
    betta32 = -(3 - np.sqrt(11))
    betta31 = 11 * (3 - np.sqrt(11)) / 10
    p1 = -4 / 6
    p2 = p3 = 5 / 6

    step = (right - left) / n
    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        K1 = step * y_diff_np(x_values[i], y_values[i])
        K2 = step * y_diff_np(x_values[i] + alpha2 * step, y_values[i] + betta21 * K1)
        K3 = step * y_diff_np(
            x_values[i] + alpha3 * step, y_values[i] + betta31 * K1 + betta32 * K2
        )
        y_values[i + 1] = y_values[i] + p1 * K1 + p2 * K2 + p3 * K3

    return y_values


def solve_method(solution, y0, x_values, left, right, n=100):
    step = (right - left) / n

    y_values = np.zeros(n + 1)
    y_values[0] = y0

    for i in range(n):
        y_values[i + 1] = solution(x_values[i])

    return y_values


def error_method(y_real, y_unreal):
    error = 0.0
    for i in range(len(y_real)):
        if np.isnan(y_real[i]):
            error += y_unreal[i] ** 2

        else:
            error += (y_real[i] - y_unreal[i]) ** 2

    return np.sqrt(error)


func_list = (
    [(y * sp.diff(y, x)) / x + sp.exp(y), 0.01, 1.0, 1.6],
    [sp.diff(y, x) + y - sp.exp(x), -1, 0.0, 1.0],
    [sp.diff(y, x) - 2 * x * y, -2, 1, 1.1],
)


def main():
    for item in func_list:
        func, y0, left_border, right_border = item
        print()
        sp.pprint(func)
        print()
        y_diff = sp.solve(func, sp.diff(y, x))[0]
        all_solutions = sp.dsolve(func, y, ics={y.subs(x, 1): 0})

        if not isinstance(all_solutions, list):  # вдруг решение одно
            all_solutions = [all_solutions]

        one_random_solution = random.choice(all_solutions)
        one_random_solution_np = sp.lambdify(
            x, one_random_solution.rhs, modules=["numpy", {"LambertW": lambertw}]
        )
        fragmentations = [10, 100, 200]
        for fragmentation in fragmentations:
            y_diff_np = sp.lambdify((x, y), y_diff, "numpy")
            x_values = np.linspace(left_border, right_border, fragmentation + 1)

            real_solution_arr = solve_method(
                one_random_solution_np,
                y0,
                x_values,
                left_border,
                right_border,
                fragmentation,
            )
            euler_arr = euler_method(
                y_diff_np, y0, x_values, left_border, right_border, fragmentation
            )
            shy_euler_arr = shy_euler_method(
                y_diff_np, y0, x_values, left_border, right_border, fragmentation
            )
            heun_arr = heun_method(
                y_diff_np, y0, x_values, left_border, right_border, fragmentation
            )
            runge_kutta_arr = runge_kutta_method(
                y_diff_np, y0, x_values, left_border, right_border, fragmentation
            )

            METHODS_DICT = {
                "Явный Метод Эйлера": euler_arr,
                "Неявный Метод Эйлера": shy_euler_arr,
                "Метод Хойна": heun_arr,
                "Метод Рунге-Кутта": runge_kutta_arr,
            }

            print("    i | x[i]  |  точное  | я. Эйлер | н. Эйлер | м. Хойна | м. Р-Г ")
            print(
                "------|-------|----------|----------|----------|----------|---------"
            )
            for i in range(len(euler_arr)):
                print(
                    f"{i:>5} | {x_values[i]:<4.3f} | {real_solution_arr[i]:<+8.5f} |{euler_arr[i]:<+9.5f} | {shy_euler_arr[i]:<+.5f} | {heun_arr[i]:<+.5f} | {runge_kutta_arr[i]:<+9.5f}"
                )

            print()
            print("Метод - Ошибка")
            for i in METHODS_DICT.keys():
                print(f"{i} - {error_method(real_solution_arr, METHODS_DICT[i])}")
            print()


main()
