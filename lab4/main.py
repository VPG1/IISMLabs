import numpy as np
import tkinter as tk
from tkinter import messagebox


# Функция для вычисления характеристик СМО
def calculate():
    try:
        lambda_rate = float(entry_lambda.get())  # Интенсивность потока заявок
        mu = float(entry_mu.get())  # Интенсивность обслуживания на фазе
        k = int(entry_k.get())  # Число фаз обслуживания

        mu_k = k * mu  # Интенсивность обслуживания по закону Эрланга

        # Составляем систему уравнений для стационарных вероятностей
        A = np.array([
            [1, 1, 1, 1],  # Условие нормировки
            [lambda_rate, -mu_k, 0, 0],  # Уравнение для состояния p_0
            [0, lambda_rate, -mu_k, 0],  # Уравнение для состояния p_1
            [0, 0, lambda_rate, -mu_k]  # Уравнение для состояния p_2
        ])

        b = np.array([1, 0, 0, 0])  # Вектор правой части (условие нормировки + стационарные уравнения)

        # Решаем систему уравнений
        p = np.linalg.solve(A, b)

        # Вероятности состояний
        p0, p1, p2, p3 = p

        # 1. Коэффициент занятости системы
        rho = p1 + p2 + p3

        # 2. Среднее число заявок в системе
        L_s = p1 + 2 * p2 + 3 * p3

        # 3. Среднее число заявок в очереди
        L_q = p2 + 2 * p3

        # 4. Среднее время пребывания заявки в системе (по формуле Литтла)
        W_s = L_s / (lambda_rate * (1 - p3))

        # 5. Среднее время ожидания заявки в очереди
        W_q = L_q / (lambda_rate * (1 - p3))

        # 6. Вероятность отказа
        P_otkaz = p3

        # Выводим результаты в графический интерфейс
        result_text = f"Вероятности состояний:\np_0 = {p0:.4f}\np_1 = {p1:.4f}\np_2 = {p2:.4f}\np_3 = {p3:.4f}\n\n" \
                      f"Коэффициент занятости (ρ): {rho:.4f}\n" \
                      f"Среднее число заявок в системе (L_s): {L_s:.4f}\n" \
                      f"Среднее число заявок в очереди (L_q): {L_q:.4f}\n" \
                      f"Среднее время пребывания в системе (W_s): {W_s:.4f}\n" \
                      f"Среднее время ожидания в очереди (W_q): {W_q:.4f}\n" \
                      f"Вероятность отказа (P_отказ): {P_otkaz:.4f}"

        label_result.config(text=result_text)

    except Exception as e:
        messagebox.showerror("Ошибка", f"Неверный ввод данных: {str(e)}")


# Создаем графический интерфейс с помощью Tkinter
root = tk.Tk()
root.title("Одноканальная СМО с ограниченной очередью")

# Метки и поля для ввода параметров
label_lambda = tk.Label(root, text="Интенсивность потока (λ):")
label_lambda.grid(row=0, column=0)
entry_lambda = tk.Entry(root)
entry_lambda.grid(row=0, column=1)

label_mu = tk.Label(root, text="Интенсивность обслуживания (μ):")
label_mu.grid(row=1, column=0)
entry_mu = tk.Entry(root)
entry_mu.grid(row=1, column=1)

label_k = tk.Label(root, text="Число фаз обслуживания (k):")
label_k.grid(row=2, column=0)
entry_k = tk.Entry(root)
entry_k.grid(row=2, column=1)

# Кнопка для расчета
button_calculate = tk.Button(root, text="Рассчитать", command=calculate)
button_calculate.grid(row=3, column=0, columnspan=2)

# Поле для вывода результата
label_result = tk.Label(root, text="", justify="left")
label_result.grid(row=4, column=0, columnspan=2)

# Запуск основного цикла программы
root.mainloop()
