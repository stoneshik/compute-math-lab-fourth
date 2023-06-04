from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, latex, sin, exp, Symbol

from io import InputManager, OutputManager


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
        self._R: float = 0.0  # достоверность аппроксимации

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    @abstractmethod
    def output_result(self) -> str:
        pass


class LinearFunction(SolutionFunction):
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P1(x)=ax+b', 'εi'], '𝝋 = ax + b', initial_data)
        self._r: float = 0.0  # коэффициент корреляции

    def calc(self) -> PrettyTable:
        n: int = len(self._initial_data)
        sx: float = 0
        sxx: float = 0
        sy: float = 0
        sxy: float = 0
        delta: float = sxx * n - sx * sx
        delta_1: float = sxy * n - sx * sy
        delta_2: float = sxx * sy - sx * sxy
        self._a: float = delta_1 / delta
        self._b: float = delta_2 / delta
        x: Symbol = Symbol('x')
        self._function_solution = self._a * x + self._b
        table: PrettyTable = PrettyTable()
        table.field_names = self._field_names_table
        x_mean: float = 0.0
        y_mean: float = 0.0
        for i, dot in enumerate(self._initial_data):
            f_x: float = self._function_solution.subs(x, dot.x)
            self._s += (f_x - dot.y) ** 2
            x_mean += dot.x
            y_mean += dot.y
            table.add_row([i + 1, dot.x, dot.y, f_x, f_x - dot.y])
        x_mean /= n
        y_mean /= n
        for dot in self._initial_data:

        return table

    def output_result(self) -> str:
        pass


def draw(functions: iter, initial_data: list) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$F(x)$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x = Symbol('x')
    x_values = numpy.arange(initial_data[0][0] - 1, initial_data[0][-1] + 1, 0.01)
    for func in functions:
        y_values = [func.equation_func.subs(x, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values)
    x_values = []
    y_values = []
    for x, y in zip(initial_data):
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
