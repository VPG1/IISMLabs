import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
import sympy as sp


# Функция для генерации случайных чисел
def generate_random_numbers():
    try:
        # Получение текста функции распределения
        dist_func_str = dist_func_var.get()

        # Преобразование строки в символьную функцию
        y = sp.symbols('y')
        G_y = sp.sympify(dist_func_str)

        # Нахождение обратной функции
        x = sp.symbols('x')
        G_inv_x = sp.solve(G_y - x, y)
        G_inv_x = G_inv_x[0]  # Используем первый корень (предположительно, правильный)

        # Преобразование обратной функции в лямбда-функцию для вычислений
        G_inv_func = sp.lambdify(x, G_inv_x)

        # Шаг 1: Генерация равномерных случайных чисел
        n = int(num_points_var.get())
        epsilon = 1e-10
        X = np.random.uniform(0, 1 - epsilon, n)

        # Шаг 2: Преобразование случайных чисел с помощью обратной функции
        Y = G_inv_func(X)

        # Шаг 3: Построение гистограммы
        plt.hist(Y, bins=50, density=True, alpha=0.6, color='g', label='Сгенерированные данные')

        # Построение теоретической плотности на основе введённой функции
        y_vals = np.linspace(0, np.max(Y), 1000)

        # Найдем производную распределения G(y), чтобы получить плотность вероятности f(y)
        f_y = sp.diff(G_y, y)
        f_y_func = sp.lambdify(y, f_y)

        # Вычисление теоретической плотности вероятности
        pdf_theoretical = f_y_func(y_vals)

        plt.plot(y_vals, pdf_theoretical, 'r-', label='Теоретическая плотность')

        plt.title('Гистограмма и теоретическая плотность распределения')
        plt.xlabel('Значение Y')
        plt.ylabel('Плотность вероятности')
        plt.legend()
        plt.show()

        # Шаг 4: Статистическое исследование - тест Колмогорова-Смирнова
        D, p_value = kstest(Y, 'uniform', args=(0, 1))
        result_text = f"Статистика Колмогорова-Смирнова: D = {D}, p-value = {p_value}"

        if p_value > 0.05:
            result_text += "\nГипотеза о соответствии закону распределения не отвергается."
        else:
            result_text += "\nГипотеза о соответствии закону распределения отвергается."

        # Вывод результата теста
        messagebox.showinfo("Результаты теста", result_text)

    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")


# Создание основного окна программы
root = Tk()
root.title("Генерация случайных величин")

# Метка и поле для ввода функции распределения
Label(root, text="Введите функцию распределения G(y):").grid(row=0, column=0, padx=10, pady=10)
dist_func_var = StringVar()
Entry(root, textvariable=dist_func_var).grid(row=0, column=1, padx=10, pady=10)

# Метка и поле для ввода количества точек
Label(root, text="Количество точек для генерации:").grid(row=1, column=0, padx=10, pady=10)
num_points_var = StringVar(value="1000")
Entry(root, textvariable=num_points_var).grid(row=1, column=1, padx=10, pady=10)

# Кнопка для генерации случайных чисел и построения гистограммы
Button(root, text="Генерировать", command=generate_random_numbers).grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Запуск основного цикла приложения
root.mainloop()
