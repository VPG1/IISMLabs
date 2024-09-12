import time

from lab1.generators.complex_dependent_generator import ComplexDependentGenerator

import tkinter as tk
from tkinter import messagebox


def handler():
    try:
        p_a = float(entry1.get())
        if not 0 <= p_a <= 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Ошибка", "Введите число от 0 до 1")
        return

    try:
        p_b_if_a = float(entry2.get())
        if not 0 <= p_b_if_a <= 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Ошибка", "Введите число от 0 до 1")
        return

    generator = ComplexDependentGenerator(time.time(), p_a, p_b_if_a)

    events = [0, 0, 0, 0]
    for _ in range(0, 10 ** 6):
        events[generator.random()] += 1

    res = [el / 10 ** 6 for el in events]

    expected = list(map(lambda p: round(p, 5), list(generator.probabilities(p_a, p_b_if_a))))

    messagebox.showinfo("Результат",
                        f"Ожидаемое:\nP(AB)={expected[0]}\nP(A¬B)={expected[1]}\nP(¬AB)={expected[2]}\nP(¬A¬B)={expected[3]}\n "
                        f"Для 10^6 итераций:\nP(AB)={res[0]}\nP(A¬B)={res[1]}\nP(¬AB)={res[2]}\nP(¬A¬B)={res[3]}")


win = tk.Tk()
win.title('3.Имитация сложного события из зависимых событий')

tk.Label(win, text='Введите вероятность P(A):').grid(row=0, column=0)
entry1 = tk.Entry(win)
entry1.grid(row=0, column=1)

tk.Label(win, text='Введите вероятность P(B|A):').grid(row=1, column=0)
entry2 = tk.Entry(win)
entry2.grid(row=1, column=1)

tk.Button(win, text='Вычислить', command=handler).grid(row=2, column=0)

win.mainloop()
