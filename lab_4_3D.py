import sympy as sp

x, y, z = sp.symbols("x y, z")


def result_F(F, X):
    res = F.subs({x: X[0], y: X[1], z: X[2]})
    return res


def jacobian_matrix(F):
    J = F.jacobian(sp.Matrix([x, y, z]))
    return J


def newton_system(F, start_data, eps=1e-5, N=25):
    X_cur = start_data
    history_data = []

    for i in range(N):
        X_next = X_cur - sp.Inverse(jacobian_matrix(F)).subs(
            {x: X_cur[0], y: X_cur[1], z: X_cur[2]}
        ) * result_F(F, X_cur)

        diff = (X_next - X_cur).norm()
        history_data.append(diff)
        print(
            f"{i + 1:>2} | {float(X_cur[0]):>7.3f} {float(X_cur[1]):>7.3f} {float(X_cur[2]):>7.3f} | {float(diff):>7.3e}"
        )
        if diff < eps * eps:
            return i + 1, history_data
        X_cur = X_next

    print("Достигнут лимит итераций")
    return N, history_data


F = [x - y - 5, x * y * z, z - x - y]

sp_F = sp.Matrix(F)


def main():
    try:
        input_data = list(map(float, input("Введите зачения: ").split()))
        while input != "q":
            print(input_data)
            sp_X = sp.Matrix(input_data)

            iter, history = newton_system(sp_F, sp_X)

            input_data = list(map(float, input("Введите зачения: ").split()))
    except Exception as e:
        if isinstance(e, sp.matrices.exceptions.NonInvertibleMatrixError):
            print("Определитель равен нулю")
        elif isinstance(e, IndexError):
            exit()


main()
