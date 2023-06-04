import math
from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, latex, sin, exp, Symbol

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
    Базовый абстрактный класс для классов реализаций методов решения ур-й
    """

    def __init__(self, field_names_table: list, kind_function: str, initial_data: list) -> None:
        self._field_names_table = field_names_table
        self._kind_function: str = kind_function
        self._initial_data: list = initial_data
        self._function_solution = None
        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._s: float = 0.0  # мера отклонения
        self._delta: float = 0.0  # среднеквадратичное отклонение
        self._r_square: float = 0.0  # достоверность аппроксимации

    @property
    def function_solution(self) -> Equation:
        return self._function_solution

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

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    def output_result(self) -> str:
        return ""


class LinearFunction(SolutionFunction):
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P1(x)=ax+b', 'εi'], '𝝋 = ax + b', initial_data)
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

    def output_result(self) -> str:
        return f"Коэффициент корреляции Пирсона {self._r}\n"


def draw(functions: iter, initial_data: list) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$F(x)$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x = Symbol('x')
    x_values = numpy.arange(initial_data[0][0] - 1, initial_data[0][-1] + 1, 0.01)
    for func in functions:
        y_values = [func.function_solution.equation_func.subs(x, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values)
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
    )
    output_manager: OutputManager = OutputManager()
    output_manager.choice_method_output()
    for solution_function in solution_functions:
        output_manager.output(solution_function.calc())
        output_manager.output(solution_function.output_result())
    draw(solution_functions, initial_data)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
