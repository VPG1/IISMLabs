import numpy as np
import tkinter as tk
from tkinter import messagebox


def calculate_smo_parameters(lambda_rate, mu_rate, nu_rate):
    # Находим P0
    denominator = 1 + (lambda_rate / nu_rate) + (lambda_rate ** 2) / (mu_rate ** 2 * (1 - (lambda_rate / mu_rate)))
    P0 = 1 / denominator

    # Находим P1, P2, ... Pn
    P = [P0]
    n = 1
    while True:
        if n == 1:
            Pn = (lambda_rate / nu_rate) * P0
        else:
            Pn = (lambda_rate / mu_rate) * P[n - 1]

        if Pn < 1e-6:
            break

        P.append(Pn)
        n += 1

    # Подсчет средних характеристик
    L_s = sum(n * P[n] for n in range(len(P)))  # Среднее число заявок в системе
    L_q = sum((n - 1) * P[n] for n in range(2, len(P)))  # Среднее число заявок в очереди

    W_s = L_s / lambda_rate  # Среднее время пребывания заявки в системе
    W_q = L_q / lambda_rate  # Среднее время ожидания заявки в очереди

    return {
        "P": P,
        "L_s": L_s,
        "L_q": L_q,
        "W_s": W_s,
        "W_q": W_q,
    }


def on_calculate():
    try:
        lambda_rate = float(entry_lambda.get())
        mu_rate = float(entry_mu.get())
        nu_rate = float(entry_nu.get())

        results = calculate_smo_parameters(lambda_rate, mu_rate, nu_rate)

        # Вывод результатов
        result_text = "Вероятности состояний:\n"
        for n, Pn in enumerate(results["P"]):
            result_text += f"P({n}) = {Pn:.4f}\n"
        result_text += f"\nСреднее число заявок в системе (L_s) = {results['L_s']:.4f}\n"
        result_text += f"Среднее число заявок в очереди (L_q) = {results['L_q']:.4f}\n"
        result_text += f"Среднее время пребывания заявки в системе (W_s) = {results['W_s']:.4f}\n"
        result_text += f"Среднее время ожидания заявки в очереди (W_q) = {results['W_q']:.4f}\n"

        messagebox.showinfo("Результаты", result_text)
    except ValueError:
        messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числовые значения.")


# Создание основного окна
root = tk.Tk()
root.title("Система массового обслуживания")

# Создание полей ввода
tk.Label(root, text="Интенсивность потока (λ):").grid(row=0, column=0, padx=10, pady=10)
entry_lambda = tk.Entry(root)
entry_lambda.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Параметр обслуживания (μ):").grid(row=1, column=0, padx=10, pady=10)
entry_mu = tk.Entry(root)
entry_mu.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Параметр разогрева (ν):").grid(row=2, column=0, padx=10, pady=10)
entry_nu = tk.Entry(root)
entry_nu.grid(row=2, column=1, padx=10, pady=10)

# Кнопка для расчета
button_calculate = tk.Button(root, text="Рассчитать", command=on_calculate)
button_calculate.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Запуск основного цикла
root.mainloop()
