class InputManager:
    def __init__(self) -> None:
        self._method_input: int = 1
        self._file_path: str = ''

    def input(self) -> (list, None):
        print("Формат входного файла:\nКоличество точек 8<=N<=12\nКоординаты X и Y через пробел\n")
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
        while True:
            n: int = int(input("Введите количество вводимых точек...\n"))
            if not (8 <= n <= 12):
                print("Количество вводимых точек должно находится в интервале от 8 до 12")
                continue
            break
        initial_data: list = [[], []]
        print("Вводите значения X и Y через пробел")
        for _ in range(n):
            x, y = (float(i) for i in input().split())
            initial_data[0].append(x)
            initial_data[1].append(y)
        return initial_data

    def _input_from_file(self) -> (list, None):
        initial_data: list = [[], []]
        with open(self._file_path, 'r', encoding='utf-8') as file:
            n: int = int(file.readline())
            if not (8 <= n <= 12):
                print("Количество вводимых точек должно находится в интервале от 8 до 12")
                return None
            for _ in range(n):
                x, y = (float(i) for i in file.readline().split())
                initial_data[0].append(x)
                initial_data[1].append(y)
        return initial_data


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
        file = open(self._file_path, 'w')
        file.close()


    def output(self, data) -> None:
        if self._method_output == 2:
            self._print_in_file(data)
            return
        self._print_in_console(data)

    def _print_in_console(self, data) -> None:
        print(str(data))

    def _print_in_file(self, data) -> None:
        with open(self._file_path, 'a', encoding='utf-8') as file:
            file.write(str(data))
