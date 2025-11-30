from pprint import pprint

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def runge_kutta_method(f, y0, x_values, left, right, n=100):
    alpha2 = betta21 = (3 + np.sqrt(7)) / 8
    alpha3 = (3 - np.sqrt(7)) / 8
    betta32 = 3 - np.sqrt(7)
    betta31 = -7 * (3 - np.sqrt(7)) / 8
    p1 = -1 / 3
    p2 = p3 = 2 / 3

    step = (right - left) / n
    sistem_size = len(y0)
    y_values = np.zeros((sistem_size, n + 1))
    y_values[:, 0] = y0

    for i in range(n):
        y_current = y_values[:, i]

        K1 = step * np.array(f(x_values[i], y_current))

        y_temp = y_current + betta21 * K1
        K2 = step * np.array(f(x_values[i] + alpha2 * step, y_temp))

        y_temp = y_current + betta31 * K1 + betta32 * K2
        K3 = step * np.array(f(x_values[i] + alpha3 * step, y_temp))

        y_values[:, i + 1] = y_current + p1 * K1 + p2 * K2 + p3 * K3

    return y_values


def get_rhs_4d(x, y_vec):
    # y = y1, y1' = y2, y2' = y3, y3' = y4, y4' = f(x, y1, y2, y3)
    y1, y2, y3, y4 = y_vec

    dy1_dx = y2
    dy2_dx = y3
    dy3_dx = y4
    dy4_dx = np.cos(x) - 2 * y3 + y2 - 3 * y1

    return [dy1_dx, dy2_dx, dy3_dx, dy4_dx]


def get_rhs_3d(x, y_vec):
    # y = y1, y1' = y2, y2' = y3, y3' = f(x, y1, y2, y3)
    # y_vec = [y, y', y'']
    y1, y2, y3 = y_vec

    dy1_dx = y2
    dy2_dx = y3
    dy3_dx = (6 * y1 - 3 * (2 * x + 3) * y2) / ((2 * x + 3) ** 3)

    return [dy1_dx, dy2_dx, dy3_dx]


def error_method(y_real, y_unreal):
    error = 0.0
    for i in range(len(y_real)):
        if np.isnan(y_real[i]):
            error += y_unreal[i] ** 2

        else:
            error += (y_real[i] - y_unreal[i]) ** 2

    return np.sqrt(error)


FUNC_LIST = [
    (get_rhs_3d, [-1, 1, 0], -1, 1),
    (get_rhs_4d, [1, 0, -1, 0.5], 0, 2),
]


def main():
    for item in FUNC_LIST:
        func, y0, left_border, right_border = item
        fragmentation = 10  # чем меньше разбиение, тем меньше погрешность
        x_values = np.linspace(left_border, right_border, fragmentation + 1)

        # result = get_analytic_solution()
        # sp.pprint(result)

        solution = solve_ivp(
            func,
            [left_border, right_border],
            y0,
            t_eval=x_values,
            method="RK45",
            rtol=1e-10,
        )
        result = runge_kutta_method(
            func, y0, x_values, left_border, right_border, fragmentation
        )
        print("    i |  x[i]  |    y_т   |    y*    |")
        print("------|--------|----------|----------|")
        for i in range(len(x_values)):
            print(
                f"{i:>5} | {x_values[i]:<+5.3f} | {solution.y[0, i]:<+.5f} | {result[0, i]:<+.5f} |"
            )

        print(f"Погрешность: {error_method(solution.y[0], result[0, :])}")
        print()


main()
# y_values = np.zeros((2, 3))
# print(y_values)
# y_values[:, 1] = np.array([1, 2])
# print(y_values)
# print(y_values[:, 1])
