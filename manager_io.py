class InputManager:
    def __init__(self) -> None:
        self._method_input: int = 1
        self._file_path: str = ''

    def input(self) -> list:
        while True:
            print("Выберите способ ввода данных")
            print("1. Через консоль\n2. Через файл")
            num_variant = int(input("Введите номер выбранного варианта...\n"))
            if num_variant < 1 or num_variant > 2:
                print("Введен неправильной номер, повторите ввод")
                continue
            break
        if num_variant == 2:
            self._file_path: str = input("Введите название файла\n")
            return self._input_from_file()
        return self._input_from_console()

    def _input_from_console(self) -> list:
        equation = None
        while True:
            print("Выберите уравнение:")
            [print(f"{i + 1}. {equation_iter.get_string()}") for i, equation_iter in enumerate(equations)]
            equation_num = int(input("Введите номер выбранного уравнения...\n"))
            if equation_num < 1 or equation_num > len(equations):
                print("Номер уравнения не найден, повторите ввод")
                continue
            equation = equations[equation_num - 1]
            break
        while True:
            print("Выберите границы интервала:")
            a, b = (float(i) for i in input("Введите значения a и b через пробел...\n").split())
            if a == b:
                print("Значения должны быть различны")
                continue
            elif a > b:
                print("Значение a должно быть меньше b")
                continue
            break
        solution_method = None
        while True:
            print("Выберите метод решения")
            [print(f"{i + 1}. {solution_method_iter.name}") for i, solution_method_iter in enumerate(solution_methods)]
            solution_num = int(input("Введите номер выбранного метода решения...\n"))
            if solution_num < 1 or solution_num > len(solution_methods):
                print("Номер метода не найден, повторите ввод")
                continue
            solution_method = solution_methods[solution_num - 1]
            break
        while True:
            epsilon = input(
                "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,001 нажмите Enter)...\n")
            if epsilon == '':
                solution_method = solution_method(equation, a, b)
                break
            epsilon = float(epsilon)
            if epsilon <= 0:
                print("Значение погрешности должно быть больше нуля")
                continue
            solution_method = solution_method(equation, a, b, epsilon)
            break
        return solution_method

    def _input_from_file(self) -> list:
        file_name: str = input("Введите название файла\n")
        with open(file_name, 'r', encoding='utf-8') as file:
            equation_num = int(file.readline())
            if equation_num < 1 or equation_num > len(equations):
                print("Номер уравнения не найден, повторите ввод")
                return None
            equation = equations[equation_num - 1]
            a, b = (float(i) for i in file.readline().split())
            if a == b:
                print("Значения должны быть различны")
                return None
            elif a > b:
                print("Значение a должно быть меньше b")
                return None
            solution_num = int(file.readline())
            if solution_num < 1 or solution_num > len(solution_methods):
                print("Номер метода не найден, повторите ввод")
                return None
            solution_method = solution_methods[solution_num - 1]
            epsilon = file.readline()
            if epsilon == '':
                return solution_method(equation, a, b)
            epsilon = float(epsilon)
            if epsilon <= 0:
                print("Значение погрешности должно быть больше нуля")
                return None
            return solution_method(equation, a, b, epsilon)


class OutputManager:
    def __init__(self) -> None:
        self._method_output: int = 1
        self._file_path: str = ''

    def choice_method_output(self) -> None:
        while True:
            print("Выберите способ вывода данных")
            print("1. Через консоль\n2. Через файл")
            num_variant: int = int(input("Введите номер выбранного варианта...\n"))
            if num_variant < 1 or num_variant > 2:
                print("Введен неправильной номер, повторите ввод")
                continue
            break
        self._method_output = num_variant
        if num_variant == 1:
            return
        self._file_path: str = input("Введите название файла\n")

    def output(self, data) -> None:
        if self._method_output == 2:
            self._print_in_file(data)
            return
        self._print_in_console(data)

    def _print_in_console(self, data) -> None:
        print(str(data))

    def _print_in_file(self, data) -> None:
        with open(self._file_path, 'w', encoding='utf-8') as file:
            file.write(str(data))