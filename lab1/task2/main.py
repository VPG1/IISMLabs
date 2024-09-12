import time

from lab1.generators.complex_generator import ComplexGenerator

import tkinter as tk
from tkinter import messagebox


def handler():
    probabilities = []
    for entry in entries:
        try:
            probability = float(entry.get())
            if not 0 <= probability <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите числа от 0 до 1")
            return
        probabilities.append(probability)

    generator = ComplexGenerator(time.time(), probabilities)

    successes = [0] * len(probabilities)
    for _ in range(0, 10 ** 6):
        cur_res = generator.random()
        for i in range(len(cur_res)):
            if cur_res[i]:
                successes[i] += 1

    res = [success / 10 ** 6 for success in successes]

    res_str = ""
    for i in range(len(probabilities)):
        res_str += f"Событие {i} \nОжидание: {probabilities[i]}\n Для 10^6 итераций: {res[i]}\n"

    messagebox.showinfo("Результат", res_str)


def push_event():
    row = win.grid_size()
    row = row[1]

    label = tk.Label(win, text=f"Вероятность {len(labels) + 1}")
    label.grid(row=row, column=0)
    entry = tk.Entry(win)
    entry.grid(row=row, column=1)

    labels.append(label)
    entries.append(entry)


def pop_event():
    if len(labels) == 0 or len(entries) == 0:
        return

    labels[-1].destroy()
    entries[-1].destroy()
    labels.pop()
    entries.pop()


win = tk.Tk()

tk.Label(win, text='2.Имитация сложного случаного события').grid(row=0, column=0)

tk.Button(win, text='+', command=push_event).grid(row=0, column=1)
tk.Button(win, text='-', command=pop_event).grid(row=0, column=2)
tk.Button(win, text='Вычислить', command=handler).grid(row=0, column=3)

labels = []
entries = []

win.mainloop()
