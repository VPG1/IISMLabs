import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D


class TwoDimRandomVariableApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Двумерная ДСВ")

        self.prob_matrix = None
        self.samples = None

        # Элементы интерфейса
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Введите матрицу вероятностей (разделяйте пробелами):").pack()
        self.matrix_input = tk.Entry(self.master)
        self.matrix_input.pack()

        self.generate_button = tk.Button(self.master, text="Сгенерировать выборку", command=self.generate_samples)
        self.generate_button.pack()

        self.results_button = tk.Button(self.master, text="Показать результаты", command=self.show_results)
        self.results_button.pack()

        self.conditional_button = tk.Button(self.master, text="Показать условные вероятности",
                                            command=self.show_conditional_probabilities)
        self.conditional_button.pack()

        self.histogram_button = tk.Button(self.master, text="Показать гистограммы X и Y", command=self.plot_histograms)
        self.histogram_button.pack()

        self.plot_3d_button = tk.Button(self.master, text="Показать 3D-гистограмму", command=self.plot_3d_histogram)
        self.plot_3d_button.pack()

        self.hypothesis_button = tk.Button(self.master, text="Проверить гипотезы", command=self.test_hypotheses)
        self.hypothesis_button.pack()

    def generate_samples(self):
        # Получение матрицы вероятностей от пользователя
        matrix_str = self.matrix_input.get()
        try:
            self.prob_matrix = np.array([[float(num) for num in row.split()] for row in matrix_str.split(';')])
            self.prob_matrix /= np.sum(self.prob_matrix)  # Нормализация

            # Генерация выборки
            self.samples = self.generate_2d_random_variable(self.prob_matrix, num_samples=1000)
            messagebox.showinfo("Информация", "Выборка успешно сгенерирована.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def generate_2d_random_variable(self, prob_matrix, num_samples=1):
        if not np.isclose(np.sum(prob_matrix), 1):
            raise ValueError("Сумма элементов матрицы вероятностей должна равняться 1.")

        rows, cols = prob_matrix.shape
        samples = []
        for _ in range(num_samples):
            random_value = np.random.rand()
            cumulative_probability = 0.0

            for i in range(rows):
                for j in range(cols):
                    cumulative_probability += prob_matrix[i, j]
                    if random_value < cumulative_probability:
                        samples.append((i, j))
                        break
                else:
                    continue
                break

        return samples

    def show_results(self):
        if self.samples is None:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте выборку.")
            return

        # Проверка на независимость
        independence_result = self.check_independence(self.samples, self.prob_matrix)
        independence_text = "Независимы" if np.all(independence_result) else "Зависимы"

        # Характеристики
        expected_x, expected_y, variance_x, variance_y, covariance, correlation = self.calculate_statistics(
            self.prob_matrix)
        expected_x_practical, expected_y_practical, variance_x_practical, variance_y_practical, covariance_practical, correlation_practical = self.practical_statistics(
            self.samples)

        results = (
            f"Независимость: {independence_text}\n"
            f"Теоретические значения:\n"
            f"E(X): {expected_x:.2f}, E(Y): {expected_y:.2f}\n"
            f"Var(X): {variance_x:.2f}, Var(Y): {variance_y:.2f}\n"
            f"Cov(X, Y): {covariance:.2f}, Corr(X, Y): {correlation:.2f}\n\n"
            f"Практические значения:\n"
            f"E(X): {expected_x_practical:.2f}, E(Y): {expected_y_practical:.2f}\n"
            f"Var(X): {variance_x_practical:.2f}, Var(Y): {variance_y_practical:.2f}\n"
            f"Cov(X, Y): {covariance_practical:.2f}, Corr(X, Y): {correlation_practical:.2f}"
        )

        messagebox.showinfo("Результаты", results)

    def show_conditional_probabilities(self):
        if self.samples is None:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте выборку.")
            return

        # Практические условные вероятности
        conditional_p_y_given_x, conditional_p_x_given_y = self.compute_conditional_probabilities(self.samples)
        # Теоретические условные вероятности
        theoretical_conditional_p_y_given_x, theoretical_conditional_p_x_given_y = self.theoretical_conditional_probabilities(
            self.prob_matrix)

        results = (
            f"Практические условные вероятности P(Y | X):\n{conditional_p_y_given_x}\n\n"
            f"Практические условные вероятности P(X | Y):\n{conditional_p_x_given_y}\n\n"
            f"Теоретические условные вероятности P(Y | X):\n{theoretical_conditional_p_y_given_x}\n\n"
            f"Теоретические условные вероятности P(X | Y):\n{theoretical_conditional_p_x_given_y}"
        )

        messagebox.showinfo("Условные вероятности", results)

    def check_independence(self, samples, prob_matrix):
        rows, cols = prob_matrix.shape
        freq_table = np.zeros((rows, cols))

        for sample in samples:
            freq_table[sample] += 1

        total_samples = len(samples)
        p_x = np.sum(freq_table, axis=1) / total_samples  # P(X)
        p_y = np.sum(freq_table, axis=0) / total_samples  # P(Y)

        independence_check = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                independence_check[i, j] = p_x[i] * p_y[j]  # P(X) * P(Y)

        independence_result = np.isclose(freq_table / total_samples, independence_check)
        return independence_result

    def calculate_statistics(self, prob_matrix):
        x_values = np.arange(prob_matrix.shape[0])
        y_values = np.arange(prob_matrix.shape[1])

        expected_x = np.sum(x_values[:, np.newaxis] * prob_matrix)
        expected_y = np.sum(y_values[np.newaxis, :] * prob_matrix)

        variance_x = np.sum((x_values[:, np.newaxis] - expected_x) ** 2 * prob_matrix)
        variance_y = np.sum((y_values[np.newaxis, :] - expected_y) ** 2 * prob_matrix)

        covariance = np.sum(
            (x_values[:, np.newaxis] - expected_x) * (y_values[np.newaxis, :] - expected_y) * prob_matrix)
        correlation = covariance / np.sqrt(variance_x * variance_y)

        return expected_x, expected_y, variance_x, variance_y, covariance, correlation

    def practical_statistics(self, samples):
        x_samples = np.array([sample[0] for sample in samples])
        y_samples = np.array([sample[1] for sample in samples])

        expected_x = np.mean(x_samples)
        expected_y = np.mean(y_samples)

        variance_x = np.var(x_samples, ddof=1)
        variance_y = np.var(y_samples, ddof=1)

        covariance = np.cov(x_samples, y_samples)[0, 1]
        correlation = np.corrcoef(x_samples, y_samples)[0, 1]

        return expected_x, expected_y, variance_x, variance_y, covariance, correlation

    def theoretical_conditional_probabilities(self, prob_matrix):
        marginal_x = np.sum(prob_matrix, axis=1)
        marginal_y = np.sum(prob_matrix, axis=0)

        conditional_p_y_given_x = prob_matrix / marginal_x[:, np.newaxis]
        conditional_p_x_given_y = prob_matrix / marginal_y[np.newaxis, :]

        return conditional_p_y_given_x, conditional_p_x_given_y

    def compute_conditional_probabilities(self, samples):
        x_samples = np.array([sample[0] for sample in samples])
        y_samples = np.array([sample[1] for sample in samples])

        rows = np.unique(x_samples)
        cols = np.unique(y_samples)

        conditional_p_y_given_x = np.zeros((len(rows), len(cols)))
        conditional_p_x_given_y = np.zeros((len(cols), len(rows)))

        for i in rows:
            for j in cols:
                conditional_p_y_given_x[i, j] = np.sum((x_samples == i) & (y_samples == j)) / np.sum(x_samples == i)
                conditional_p_x_given_y[j, i] = np.sum((x_samples == i) & (y_samples == j)) / np.sum(y_samples == j)

        return conditional_p_y_given_x, conditional_p_x_given_y

    def plot_histograms(self):
        if self.samples is None:
            return

        # Создание нового окна для гистограмм
        histogram_window = tk.Toplevel(self.master)
        histogram_window.title("Гистограммы составляющих X и Y")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Гистограмма для X
        x_samples = [sample[0] for sample in self.samples]
        axs[0].hist(x_samples, bins=range(max(x_samples) + 2), edgecolor='black')
        axs[0].set_title('Гистограмма X')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Частота')

        # Гистограмма для Y
        y_samples = [sample[1] for sample in self.samples]
        axs[1].hist(y_samples, bins=range(max(y_samples) + 2), edgecolor='black')
        axs[1].set_title('Гистограмма Y')
        axs[1].set_xlabel('Y')
        axs[1].set_ylabel('Частота')

        # Вставляем график в новое окно
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def plot_3d_histogram(self):
        if self.samples is None:
            return

        # Создание нового окна для 3D-гистограммы
        histogram_window_3d = tk.Toplevel(self.master)
        histogram_window_3d.title("3D-гистограмма двумерной ДСВ")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Преобразование выборки в массивы X и Y
        x_vals = [sample[0] for sample in self.samples]
        y_vals = [sample[1] for sample in self.samples]

        # Построение 3D гистограммы
        hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=(range(max(x_vals) + 2), range(max(y_vals) + 2)))

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.8 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Частота')
        ax.set_title('3D-гистограмма двумерной ДСВ')

        # Вставляем график в новое окно
        canvas = FigureCanvasTkAgg(fig, master=histogram_window_3d)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def test_hypotheses(self):
        if self.samples is None:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте выборку.")
            return

        # Теоретические значения
        expected_x, expected_y, variance_x, variance_y, covariance, correlation = self.calculate_statistics(
            self.prob_matrix)

        # Практические значения
        expected_x_practical, expected_y_practical, variance_x_practical, variance_y_practical, covariance_practical, correlation_practical = self.practical_statistics(
            self.samples)

        # Проверка гипотезы для математического ожидания
        n = len(self.samples)
        t_x = (expected_x_practical - expected_x) / (np.sqrt(variance_x_practical / n))
        t_y = (expected_y_practical - expected_y) / (np.sqrt(variance_y_practical / n))

        p_value_x = 2 * (1 - stats.t.cdf(np.abs(t_x), df=n - 1))
        p_value_y = 2 * (1 - stats.t.cdf(np.abs(t_y), df=n - 1))

        # Проверка гипотезы для дисперсии
        chi2_x = (n - 1) * variance_x_practical / variance_x
        chi2_y = (n - 1) * variance_y_practical / variance_y
        p_value_var_x = 1 - stats.chi2.cdf(chi2_x, df=n - 1)
        p_value_var_y = 1 - stats.chi2.cdf(chi2_y, df=n - 1)

        # Проверка гипотезы для корреляции
        z_corr = (correlation_practical - correlation) / (1 / np.sqrt(n))
        p_value_corr = 2 * (1 - stats.norm.cdf(np.abs(z_corr)))

        # Результаты
        hypothesis_results = (
            f"Гипотезы для математического ожидания:\n"
            f"t-статистика для X: {t_x:.3f}, p-значение: {p_value_x:.3f}\n"
            f"t-статистика для Y: {t_y:.3f}, p-значение: {p_value_y:.3f}\n\n"
            f"Гипотезы для дисперсии:\n"
            f"хи-квадрат для X: {chi2_x:.3f}, p-значение: {p_value_var_x:.3f}\n"
            f"хи-квадрат для Y: {chi2_y:.3f}, p-значение: {p_value_var_y:.3f}\n\n"
            f"Гипотеза для корреляции:\n"
            f"Z-статистика: {z_corr:.3f}, p-значение: {p_value_corr:.3f}"
        )

        messagebox.showinfo("Результаты тестирования гипотез", hypothesis_results)


if __name__ == "__main__":
    root = tk.Tk()
    app = TwoDimRandomVariableApp(root)
    root.mainloop()

# 0.1 0.1 0.1; 0.3 0.1 0.1; 0.1 0.05 0.05