from random import choice

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y, z = sp.symbols("x y, z")


def plot(F: list) -> None:
    func_colors = ["red", "blue", "green", "black"]
    plt.figure(figsize=(8, 6))
    x_plot, y_plot = np.meshgrid(np.linspace(-10, 10, 150), np.linspace(-10, 10, 150))
    plt.axis("equal")

    legend_elements = []
    for i in range(len(F)):
        func_to_draw = sp.lambdify((x, y), F[i])(x_plot, y_plot)
        plt.contour(
            x_plot,
            y_plot,
            func_to_draw,
            levels=[0],
            colors=func_colors[i],
        )
        legend_elements.append(
            mlines.Line2D(
                [], [], color=func_colors[i], linewidth=3, label=f"{F[i]} = 0"
            ),
        )

    plt.grid(True)
    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()


def result_F(F, X):
    res = F.subs({x: X[0], y: X[1]})
    return res


def jacobian_matrix(F):
    J = F.jacobian(sp.Matrix([x, y]))
    return J


def newton_system(F, start_data, eps=1e-5, N=25):
    X_cur = start_data
    history_data = [X_cur]
    for i in range(N):
        X_next = X_cur - sp.Inverse(jacobian_matrix(F)).subs(
            {x: X_cur[0], y: X_cur[1]}
        ) * result_F(F, X_cur)

        diff = (X_next - X_cur).norm()
        history_data.append(X_next)
        print(
            f"{i + 1:>2} | {float(X_cur[0]):>7.3f} {float(X_cur[1]):>7.3f} | {float(diff):>7.3e}"
        )
        if diff < eps * eps:
            return i + 1, history_data
        X_cur = X_next

    print("Достигнут лимит итераций")
    return N, history_data


def check_with_real_roots(F, X_last):
    "Посчитать значение системы и сравнить со значением по невязке"
    roots = list(sp.nonlinsolve(F, [x, y]))
    print(f"Корни: {roots}")
    random_root = choice(roots)
    print(f"Выбранное решение: {random_root}")
    res = 0.0
    for i in range(2):
        res += (random_root[i] - X_last[i]) ** 2
    return sp.sqrt(res)


FUNC_LIST = [
    [x**2 + y**2 - 5, x * y - 2],
    [x + y + 2 * x * y - 7, 2 * x + 2 * y + x * y - 8],
]


def main():
    for F in FUNC_LIST:
        print("-" * 10)
        plot(F)
        sp_F = sp.Matrix(F)
        find_and_check_solution(sp_F)
        try:
            input_data = list(map(float, input("Введите зачения: ").split()))
            while input_data:
                # print(input_data)
                sp_X = sp.Matrix(input_data)

                iter, X_data = newton_system(sp_F, sp_X)

                print(
                    f"Невязка последнего приближенного корня и одного из точных решений {check_with_real_roots(F, X_data[-1])}"
                )
                input_data = list(map(float, input("Введите зачения: ").split()))

        except Exception as e:
            if isinstance(e, sp.matrices.exceptions.NonInvertibleMatrixError):
                print("Определитель равен нулю")
            elif isinstance(e, IndexError):
                break
            else:
                continue


if __name__ == "__main__":
    main()
