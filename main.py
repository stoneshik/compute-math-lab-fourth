from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, latex, sin, exp, Symbol


class Dot:
    """
    Класс обёртка для точки
    """
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


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
        self._r: float = 0.0  # достоверность аппроксимации

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    @abstractmethod
    def output_result(self) -> str:
        pass


class LinearFunction(SolutionFunction):
    def __init__(self, initial_data: list) -> None:
        super().__init__(['X', 'Y', 'P1(x)=ax+b', 'εi'], '𝝋 = ax + b', initial_data)

    def calc(self) -> PrettyTable:
        pass

    def output_result(self) -> str:
        pass


def draw(functions: iter, initial_data: list) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$F(x)$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x = Symbol('x')
    x_values = numpy.arange(initial_data[0].x - 1, initial_data[-1].x + 1, 0.01)
    for func in functions:
        y_values = [func.equation_func.subs(x, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values)
    x_values = []
    y_values = []
    for i in initial_data:
        x_values.append(i.x)
        y_values.append(i.y)
    plt.scatter(x_values, y_values, color='red', marker='o')
    plt.show()


def main():
    x = Symbol('x')
    initial_data: list = input_data()
    solution_functions = (
        LinearFunction(initial_data),
    )
    output_manager: OutputManager = OutputManager()
    output_manager.choice_method_output()
    for solution_function in solution_functions:



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
