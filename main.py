import math
from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, latex, exp, Symbol, ln

from manager_io import InputManager, OutputManager


class Equation:
    """
    Класс обертка для функций
    """

    def __init__(self, equation_func) -> None:
        self.equation_func = equation_func

    def get_string(self) -> str:
        return latex(self.equation_func)

    def get_diff(self):
        return diff(self.equation_func)


class SolutionFunction(ABC):
    """
    Базовый абстрактный класс для классов реализаций аппроксимирующих функций
    """

    def __init__(self, field_names_table: list, kind_function: str, initial_data: list) -> None:
        self._field_names_table = field_names_table
        self._kind_function: str = kind_function
        self._initial_data: list = initial_data
        self._function_solution = None
        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._d: float = 0.0
        self._s: float = 0.0  # мера отклонения
        self._delta: float = 0.0  # среднеквадратичное отклонение
        self._r_square: float = 0.0  # достоверность аппроксимации
        self._is_calc: bool = False

    @property
    def kind_function(self) -> str:
        return self._kind_function

    @property
    def function_solution(self) -> Equation:
        return self._function_solution

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def c(self) -> float:
        return self._c

    @property
    def d(self) -> float:
        return self._d

    @property
    def s(self) -> float:
        return self._s

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def r_square(self) -> float:
        return self._r_square

    @property
    def is_calc(self) -> bool:
        return self._is_calc

    def _calc_delta(self) -> float:
        x_symbol: Symbol = Symbol('x')
        return math.sqrt(
            sum([math.pow(self._function_solution.equation_func.subs(x_symbol, x) - y, 2)
                 for x, y in zip(self._initial_data[0], self._initial_data[1])]) / len(self._initial_data[0])
        )

    def _calc_r_square(self) -> float:
        x_symbol: Symbol = Symbol('x')
        phi = self._function_solution.equation_func
        n: int = len(self._initial_data[0])
        return (1 -
                (
                    sum([math.pow(y - phi.subs(x_symbol, x), 2)
                         for x, y in zip(self._initial_data[0], self._initial_data[1])]) /
                    (
                        sum([math.pow(phi.subs(x_symbol, x), 2) for x in self._initial_data[0]]) -
                        1 / n * pow(sum([phi.subs(x_symbol, x) for x in self._initial_data[0]]), 2))
                    )
                )

    def _form_result(self) -> PrettyTable:
        x_symbol: Symbol = Symbol('x')
        table: PrettyTable = PrettyTable()
        table.field_names = self._field_names_table
        i: int = 1
        for x, y in zip(self._initial_data[0], self._initial_data[1]):
            f_x: float = self._function_solution.equation_func.subs(x_symbol, x)
            self._s += (f_x - y) ** 2
            table.add_row([i, x, y, f_x, f_x - y])
            i += 1
        self._delta = self._calc_delta()
        self._r_square = self._calc_r_square()
        return table

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    def output_result(self) -> str:
        return ""


class LinearFunction(SolutionFunction):
    """
    Класс линейной аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P1(x)=ax+b', 'εi'], 'phi = ax + b', initial_data)
        self._r: float = self._calc_r()  # коэффициент корреляции

    def _calc_r(self) -> float:
        n: int = len(self._initial_data[0])
        x_mean: float = sum(self._initial_data[0]) / n
        y_mean: float = sum(self._initial_data[1]) / n
        return sum([(x - x_mean) * (y - y_mean) for x, y
                    in zip(self._initial_data[0], self._initial_data[1])]) / math.sqrt(
            sum([math.pow(x - x_mean, 2) for x in self._initial_data[0]]) *
            sum([math.pow(y - y_mean, 2) for y in self._initial_data[1]])
        )

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum(self._initial_data[0])
        sxx: float = sum([math.pow(x, 2) for x in self._initial_data[0]])
        sy: float = sum(self._initial_data[1])
        sxy: float = sum([x * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = sxx * n - sx * sx
        delta_1: float = sxy * n - sx * sy
        delta_2: float = sxx * sy - sx * sxy
        self._a: float = delta_1 / delta
        self._b: float = delta_2 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(self._a * x_symbol + self._b)
        self._is_calc = True
        return self._form_result()

    def output_result(self) -> str:
        return f"Коэффициент корреляции Пирсона {self._r}\n"


class SquareFunction(SolutionFunction):
    """
    Класс квадратичной аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P2(x)=ax^2+bx+c', 'εi'], 'phi = ax^2+bx+c', initial_data)

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum(self._initial_data[0])
        sx_2: float = sum([math.pow(x, 2) for x in self._initial_data[0]])
        sx_3: float = sum([math.pow(x, 3) for x in self._initial_data[0]])
        sx_4: float = sum([math.pow(x, 4) for x in self._initial_data[0]])
        sy: float = sum(self._initial_data[1])
        sxy: float = sum([x * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        sx_2y: float = sum([math.pow(x, 2) * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = numpy.linalg.det(
            numpy.array([[n, sx, sx_2], [sx, sx_2, sx_3], [sx_2, sx_3, sx_4]])
        )
        delta_1: float = numpy.linalg.det(
            numpy.array([[sy, sx, sx_2], [sxy, sx_2, sx_3], [sx_2y, sx_3, sx_4]])
        )
        delta_2: float = numpy.linalg.det(
            numpy.array([[n, sy, sx_2], [sx, sxy, sx_3], [sx_2, sx_2y, sx_4]])
        )
        delta_3: float = numpy.linalg.det(
            numpy.array([[n, sx, sy], [sx, sx_2, sxy], [sx_2, sx_3, sx_2y]])
        )
        self._a: float = delta_3 / delta
        self._b: float = delta_2 / delta
        self._c: float = delta_1 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(self._a * x_symbol ** 2 + self._b * x_symbol + self._c)
        self._is_calc = True
        return self._form_result()


class CubeFunction(SolutionFunction):
    """
    Класс кубической аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P3(x)=ax^3+bx^2+cx+d', 'εi'], 'phi = ax^3+bx^2+cx+d', initial_data)

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum(self._initial_data[0])
        sx_2: float = sum([math.pow(x, 2) for x in self._initial_data[0]])
        sx_3: float = sum([math.pow(x, 3) for x in self._initial_data[0]])
        sx_4: float = sum([math.pow(x, 4) for x in self._initial_data[0]])
        sx_5: float = sum([math.pow(x, 5) for x in self._initial_data[0]])
        sx_6: float = sum([math.pow(x, 6) for x in self._initial_data[0]])
        sy: float = sum(self._initial_data[1])
        sxy: float = sum([x * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        sx_2y: float = sum([math.pow(x, 2) * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        sx_3y: float = sum([math.pow(x, 3) * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = numpy.linalg.det(numpy.array(
            [[n, sx, sx_2, sx_3], [sx, sx_2, sx_3, sx_4], [sx_2, sx_3, sx_4, sx_5], [sx_3, sx_4, sx_5, sx_6]]
        ))
        delta_1: float = numpy.linalg.det(numpy.array(
            [[sy, sx, sx_2, sx_3], [sxy, sx_2, sx_3, sx_4], [sx_2y, sx_3, sx_4, sx_5], [sx_3y, sx_4, sx_5, sx_6]]
        ))
        delta_2: float = numpy.linalg.det(numpy.array(
            [[n, sy, sx_2, sx_3], [sx, sxy, sx_3, sx_4], [sx_2, sx_2y, sx_4, sx_5], [sx_3, sx_3y, sx_5, sx_6]]
        ))
        delta_3: float = numpy.linalg.det(numpy.array(
            [[n, sx, sy, sx_3], [sx, sx_2, sxy, sx_4], [sx_2, sx_3, sx_2y, sx_5], [sx_3, sx_4, sx_3y, sx_6]]
        ))
        delta_4: float = numpy.linalg.det(numpy.array(
            [[n, sx, sx_2, sy], [sx, sx_2, sx_3, sxy], [sx_2, sx_3, sx_4, sx_2y], [sx_3, sx_4, sx_5, sx_3y]]
        ))
        self._a: float = delta_4 / delta
        self._b: float = delta_3 / delta
        self._c: float = delta_2 / delta
        self._d: float = delta_1 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(
            self._a * x_symbol ** 3 + self._b * x_symbol ** 2 + self._c * x_symbol + self._d
        )
        self._is_calc = True
        return self._form_result()


class ExpFunction(SolutionFunction):
    """
    Класс экспоненциальной аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P(x)=a*e^{bx}', 'εi'], 'phi = a*e^{bx}', initial_data)

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum(self._initial_data[0])
        sx_2: float = sum([math.pow(x, 2) for x in self._initial_data[0]])
        sy: float = sum([math.log(y) for y in self._initial_data[1]])
        sxy: float = sum([x * math.log(y) for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = sx_2 * n - sx * sx
        delta_1: float = sx_2 * sy - sx * sxy
        delta_2: float = sxy * n - sx * sy
        self._a: float = math.exp(delta_1 / delta)
        self._b: float = delta_2 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(
            self._a * exp(self._b * x_symbol)
        )
        self._is_calc = True
        return self._form_result()


class LogarithmFunction(SolutionFunction):
    """
    Класс логарифмической аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P(x)=a*lnx+b', 'εi'], 'phi = a*lnx+b', initial_data)

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum([math.log(x) for x in self._initial_data[0]])
        sx_2: float = sum([math.pow(math.log(x), 2) for x in self._initial_data[0]])
        sy: float = sum(self._initial_data[1])
        sxy: float = sum([math.log(x) * y for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = sx_2 * n - sx * sx
        delta_1: float = sxy * n - sx * sy
        delta_2: float = sx_2 * sy - sx * sxy
        self._a: float = delta_1 / delta
        self._b: float = delta_2 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(
            self._a * ln(x_symbol) + self._b
        )
        self._is_calc = True
        return self._form_result()


class PowerFunction(SolutionFunction):
    """
    Класс показательной аппроксимирующей функции
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P(x)=a*x^b', 'εi'], 'phi = a*x^b', initial_data)

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data[0])
        sx: float = sum([math.log(x) for x in self._initial_data[0]])
        sx_2: float = sum([math.pow(math.log(x), 2) for x in self._initial_data[0]])
        sy: float = sum([math.log(y) for y in self._initial_data[1]])
        sxy: float = sum([math.log(x) * math.log(y) for x, y in zip(self._initial_data[0], self._initial_data[1])])
        delta: float = sx_2 * n - sx * sx
        delta_1: float = sx_2 * sy - sx * sxy
        delta_2: float = sxy * n - sx * sy
        self._a: float = math.exp(delta_1 / delta)
        self._b: float = delta_2 / delta
        x_symbol: Symbol = Symbol('x')
        self._function_solution = Equation(
            self._a * x_symbol ** self._b
        )
        self._is_calc = True
        return self._form_result()


def create_table_result(solution_functions: tuple) -> PrettyTable:
    table: PrettyTable = PrettyTable()
    table.field_names = ['Вид функции', 'a', 'b', 'c', 'd', 'Мера отклонения S',
                         'Среднеквадратическое отклонение 𝛿', 'Достоверность аппроксимации R^2']
    for solution_function in solution_functions:
        if not solution_function.is_calc:
            continue
        table.add_row([solution_function.kind_function, solution_function.a,
                       solution_function.b, solution_function.c, solution_function.d,
                       solution_function.s, solution_function.delta, solution_function.r_square])
    return table


def find_best_function(solution_functions: tuple) -> str:
    best_function: SolutionFunction = solution_functions[0]
    for solution_function in solution_functions:
        if not solution_function.is_calc:
            continue
        if solution_function.delta < best_function.delta:
            best_function = solution_function
    return f"Наилучшая аппроксимирующая функция {best_function.kind_function}\na={best_function.a}\nb={best_function.b}\nc={best_function.c}\ns={best_function.s}\n𝛿={best_function.delta}\nR^2={best_function.r_square}"


def draw(functions: iter, initial_data: list) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x_symbol = Symbol('x')
    x_values = numpy.arange(initial_data[0][0] - 0.2, initial_data[0][-1] + 0.2, 0.01)
    for func in functions:
        if not func.is_calc:
            continue
        y_values = [func.function_solution.equation_func.subs(x_symbol, x_iter) for x_iter in x_values]
        try:
            plt.plot(x_values, y_values, linestyle='--', label=f"${func.kind_function}$")
        except TypeError:
            x_values_error = numpy.arange(initial_data[0][0], initial_data[0][-1], 0.01)
            y_values_error = [func.function_solution.equation_func.subs(x_symbol, x_iter) for x_iter in x_values_error]
            plt.plot(x_values_error, y_values_error, linestyle='--', label=f"${func.kind_function}$")
    plt.legend(loc='upper left')
    x_values = []
    y_values = []
    for x, y in zip(initial_data[0], initial_data[1]):
        x_values.append(x)
        y_values.append(y)
    plt.scatter(x_values, y_values, color='red', marker='o')
    plt.show()


def main():
    input_manager: InputManager = InputManager()
    initial_data = input_manager.input()
    if initial_data is None:
        return
    solution_functions = (
        LinearFunction(initial_data),
        SquareFunction(initial_data),
        CubeFunction(initial_data),
        ExpFunction(initial_data),
        LogarithmFunction(initial_data),
        PowerFunction(initial_data),
    )
    output_manager: OutputManager = OutputManager()
    output_manager.choice_method_output()
    for solution_function in solution_functions:
        try:
            output_manager.output(f"{solution_function.calc()}\n")
            output_manager.output(solution_function.output_result())
        except ValueError:
            output_manager.output(f"у функции {solution_function.kind_function} недопустимый аргумент")
    output_manager.output(f"{create_table_result(solution_functions)}\n")
    output_manager.output(find_best_function(solution_functions))
    draw(solution_functions, initial_data)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
