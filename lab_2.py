import numpy                as np
import sympy                as sp
import matplotlib.pyplot    as plt
import pprint               as pp


EPS = 1e-5
left_border, right_border = 0, 1  # шаг
x = sp.symbols('x')

my_func34 = 2 ** (3 * x)


def _integrate(func):  # вычисление точного значения моей функции
    return sp.integrate(func, (x, left_border, right_border)).evalf()


def square_method(func, left_, right_, N=10):
    """Формула Средних Прямоугольников"""
    res = 0
    H = (right_ - left_) / N
    for i in range(N):
        current_val = left_border + i * H
        nex_val = current_val + H
        res += func.subs(x, (current_val + nex_val) / 2).evalf()
    return res * H


def trap_method(func, left_, right_, N=10):
    """Формула Трапеции"""
    res = func.subs(x, left_) + func.subs(x, right_)
    H = (right_ - left_) / N
    for i in range(1, N):
        current_val = left_ + i * H
        res += 2 * func.subs(x, current_val)
    return res * H / 2


def simpson_method(func, left_, right_, N=10):
    """Формула Симпсона"""
    res = func.subs(x, left_) + func.subs(x, right_)
    H = (right_ - left_) / N
    for i in range(1, N):
        current_val = left_ + i * H
        if i%2==0:
            res += 2 * func.subs(x, current_val)
        else:
            res += 4 * func.subs(x, current_val)
    return res * H / 3


def method_of_runge(method):
    """Правило Рунге"""
    N = 10
    print(METHODS_DICT[method][0])
    while abs(method(my_func34, left_border, right_border, 2*N) - \
              method(my_func34, left_border, right_border, N)) >= METHODS_DICT[method][1]*EPS:
        # print(f"Значение разбиения N = {N} не подходит")
        N+=15
    else:
        print(f"При помощи правила Рунге найдено допустимое значение разбиения N = {N}")
        return N


def square_error(func, left_, right_, N):
    """Погрешность Средних Прямоугольников"""
    second_diff = sp.diff(func, x, 2)
    H = (right_ - left_) / N
    max_diff_value = 0
    for i in range(N+1):
        current_val = left_border + i * H
        value = abs(second_diff.subs(x, current_val).evalf())
        if value>max_diff_value:
            max_diff_value = value
    return (right_ - left_) * max_diff_value * H**2 / 24


def trap_error(func, left_, right_, N):
    """Погрешность Трапеции"""
    second_diff = sp.diff(func, x, 2)
    H = (right_ - left_) / N
    max_diff_value = 0
    for i in range(N + 1):
        current_val = left_border + i * H
        value = abs(second_diff.subs(x, current_val).evalf())
        if value > max_diff_value:
            max_diff_value = value
    return (right_ - left_) * max_diff_value * H ** 2 / 12


def simpson_error(func, left_, right_, N):
    """Погрешность Симпсона"""
    second_diff = sp.diff(func, x, 4)
    H = (right_ - left_) / N
    max_diff_value = 0
    for i in range(N + 1):
        current_val = left_border + i * H
        value = abs(second_diff.subs(x, current_val).evalf())
        if value > max_diff_value:
            max_diff_value = value
    return (right_ - left_) * max_diff_value * H ** 4 / 180


def print_method(table_):
    print(*table_['data'])
    for i in keys[1:]:
        values_str = " ".join(f"{x:.2e}" for x in table_[i])
        print(f"{METHODS_DICT[i][0]:<35} : {values_str}")


METHODS_DICT = {
    square_method:  ("Формула Средних Прямоугольников", 1/3),
    simpson_method: ("Формула Симпсона", 1/15),
    trap_method:    ("Формула Трапеции", 1/3)
}

ERRORS_METHOD = {
    square_error:   "Погрешность Средних Прямоугольников",
    trap_error:     "Погрешность Трапеции",
    simpson_error:  "Погрешность Симпсона"
}

table_one = {
    "data" : [2, 10, 100, 1000, 5000, [], []], # я хотел взять 10000, но это много
    square_method: [],
    trap_method: [],
    simpson_method : []
}

keys = list(table_one.keys())
for i in keys[1:]:
    print(METHODS_DICT[i][0])
    for j in table_one['data'][:5]:
        print(
            f"{i(my_func34, left_border, right_border, j):.6e}, "
            f"{abs(_integrate(my_func34) - i(my_func34, left_border, right_border, j)):.2e}"
        )
        table_one[i].append(abs(_integrate(my_func34) - i(my_func34, left_border, right_border, j)))
    print()
    print(10*'-')
    print()

for i in keys[1:]:
    N = method_of_runge(i)
    print(
        f"{i(my_func34, left_border, right_border, N):.6e}, "
        f"{abs(_integrate(my_func34) - i(my_func34, left_border, right_border, N)):.2e}"
    )
    table_one['data'][5].append(N)
    table_one[i].append(abs(_integrate(my_func34) - i(my_func34, left_border, right_border, N)))
    print()


errors = list(ERRORS_METHOD.keys())
for i in range(3):
    print(ERRORS_METHOD[errors[i]])
    N = 10
    while EPS < errors[i](my_func34, left_border, right_border, N):
        N+=10

    print(
        f"{keys[1:][i](my_func34, left_border, right_border, N):.6e}, "
        f"{abs(_integrate(my_func34) - keys[1:][i](my_func34, left_border, right_border, N)):.2e}"
    )

    table_one['data'][6].append(N)
    table_one[keys[1:][i]].append(abs(_integrate(my_func34) - keys[1:][i](my_func34, left_border, right_border, N)))
    print()

print_method(table_one)

# График
# Сбор основных точек
n_basis = table_one['data'][:5]
n_runge = table_one['data'][5]
n_errors = table_one['data'][6]

# сбор значений, соответствующих точкам
square = table_one[square_method]
trap = table_one[trap_method]
simpson = table_one[simpson_method]

plt.figure(figsize=(12, 8))

methods = [
    ('Прямоугольник', square, 'blue', 'o'),
    ('Трапеция', trap, 'red', 's'),
    ('Симпсон', simpson, 'green', '^')
]

for method_name, errors, color, marker in methods:
    # построение основных точек
    plt.semilogy(n_basis, errors[:5], color=color, marker=marker,
                 linestyle='-', linewidth=1, markersize=6, label=f'{method_name}')

    # построение точек метода Рунге
    plt.semilogy(n_runge[methods.index((method_name, errors, color, marker))],
                 errors[5], color=color, marker=marker, markersize=10,
                 markerfacecolor='none', markeredgewidth=2, label=f'{method_name} - Рунге')
    # markerfacecolor='none' - пустая заливка, markeredgewidth=2 - толщина контура маркера

    # построение точек после оценки Погрешности
    plt.semilogy(n_errors[methods.index((method_name, errors, color, marker))],
                 errors[6], color=color, marker=marker, markersize=10,
                 markerfacecolor=color, label=f'{method_name} - Погрешность')

plt.xlabel('Число разбиений n', fontsize=12)
plt.ylabel('Невязка', fontsize=12)
plt.title('Зависимость невязки от числа разбиений', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
# легенда лежит так, что её левый верхний угол находился в точке (1.05, 1) (справа, чуть ниже угла)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




""" Проверка работы функций
print(_integrate(my_func34))
print(square_method(my_func34, left_border, right_border, 1000))
print(trap_method(my_func34, left_border, right_border, 1000))
print(simpson_method(my_func34, left_border, right_border, 1000))

print(10*"-")
print()

method_of_runge(square_method)
method_of_runge(trap_method)
method_of_runge(simpson_method)
"""