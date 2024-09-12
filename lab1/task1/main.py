import time

from lab1.generators.simple_generator import SimpleGenerator

import tkinter as tk
from tkinter import messagebox


def handler():
    try:
        probability = float(entry.get())
        if not 0 <= probability <= 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Ошибка", "Введите число от 0 до 1")
        return

    a = SimpleGenerator(time.time(), probability)
    success = 0
    for _ in range(0, 10 ** 6):
        if a.random():
            success += 1

    messagebox.showinfo("Результат",
                        "Ожидаемое: " + str(probability) + "\n" +
                        "Для 10^6 итераций: " + str(success / 10 ** 6))


win = tk.Tk()
win.title('1.Имитация простого случаного события')
tk.Label(win, text='Введите вероятность:').grid(row=1, column=0)
entry = tk.Entry(win)
entry.grid(row=1, column=1)
tk.Button(win, text='Вычислить', command=handler).grid(row=1, column=2)

win.mainloop()
