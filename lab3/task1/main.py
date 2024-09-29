import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Определим функцию совместной плотности вероятности f(x, y)
def f_xy(x, y):
    return 0.5 * np.cos(x - y)


# Маргинальная плотность f_X(x) = 0.5 * (sin(x) + cos(x))
def marginal_density_x(x):
    return 0.5 * (np.sin(x) + np.cos(x))


def marginal_density_y(y):
    return 0.5 * (np.sin(y) + np.cos(y))


# Метод отклонений для генерации двумерной случайной величины
def generate_sample(n_samples=10000):
    samples_x = []
    samples_y = []

    for _ in range(n_samples):
        while True:
            # Генерируем кандидатов x и y из равномерного распределения на [0, pi/2]
            x_candidate = np.random.uniform(0, np.pi / 2)
            y_candidate = np.random.uniform(0, np.pi / 2)

            # Генерируем случайное число u для метода отклонений
            u = np.random.uniform(0, 0.5)  # Максимальное значение плотности f(x, y) = 0.5

            # Проверяем условие метода отклонений
            if u <= f_xy(x_candidate, y_candidate):
                samples_x.append(x_candidate)
                samples_y.append(y_candidate)
                break

    return np.array(samples_x), np.array(samples_y)


# Условная плотность f_{X|Y}(x|y)
def conditional_density_x_given_y(x, y):
    return f_xy(x, y) / marginal_density_y(y)


# Условная плотность f_{Y|X}(y|x)
def conditional_density_y_given_x(y, x):
    return f_xy(x, y) / marginal_density_x(x)


# Функция для вычисления корреляции между X и Y
def compute_correlation(x_samples, y_samples):
    # Вычисляем корреляцию Пирсона
    correlation_matrix = np.corrcoef(x_samples, y_samples)
    correlation = correlation_matrix[0, 1]  # Извлекаем корреляцию из матрицы
    return correlation


# Функция для построения 3D-графика распределения
def plot_3d_distribution(x_samples, y_samples):
    # Создание 3D-графика
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Создание двумерной гистограммы
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=30, density=True)

    # Создание координат для 3D-графика
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.025, yedges[:-1] + 0.025, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Высота столбиков соответствует гистограмме
    dx = dy = 0.05 * np.ones_like(zpos)
    dz = hist.ravel()

    # Построение гистограммы
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.6, color='cyan')

    # Создание сетки для графика плотности
    x_vals = np.linspace(0, np.pi / 2, 100)
    y_vals = np.linspace(0, np.pi / 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_xy(X, Y) + 0.1

    # Построение графика плотности
    ax.plot_surface(X, Y, Z, alpha=0.8, color='orange', edgecolor='none')

    # Настройка меток
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Плотность')
    ax.set_title('Гистограмма распределения и график плотности')

    # Установим пределы для осей
    ax.set_xlim(0, np.pi / 2)
    ax.set_ylim(0, np.pi / 2)
    ax.set_zlim(0, 0.5)

    plt.show()


# Функция для построения гистограмм и графиков условных плотностей
def plot_conditional_densities(x_samples, y_samples):
    # Фиксируем значение Y = pi/4
    fixed_y = np.pi / 4
    # Условия для фильтрации выборок
    condition_y = (y_samples > fixed_y - 0.05) & (y_samples < fixed_y + 0.05)
    x_conditional_samples = x_samples[condition_y]

    # Фиксируем значение X = pi/4
    fixed_x = np.pi / 4
    # Условия для фильтрации выборок
    condition_x = (x_samples > fixed_x - 0.05) & (x_samples < fixed_x + 0.05)
    y_conditional_samples = y_samples[condition_x]

    # Диапазон значений для построения плотностей
    x_vals = np.linspace(0, np.pi / 2, 500)
    y_vals = np.linspace(0, np.pi / 2, 500)

    # Создание общего графика
    plt.figure(figsize=(12, 6))

    # Гистограмма и условная плотность для X|Y
    plt.subplot(1, 2, 1)
    plt.hist(x_conditional_samples, bins=50, density=True, alpha=0.6, color='blue',
             label=f'Гистограмма X|Y={fixed_y:.2f}')
    plt.plot(x_vals, conditional_density_x_given_y(x_vals, fixed_y), label=f'Плотность X|Y={fixed_y:.2f}',
             color='darkblue', linewidth=2)
    plt.title(f'Условная плотность X при Y={fixed_y:.2f}')
    plt.xlabel('Значение X')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)

    # Гистограмма и условная плотность для Y|X
    plt.subplot(1, 2, 2)
    plt.hist(y_conditional_samples, bins=50, density=True, alpha=0.6, color='green',
             label=f'Гистограмма Y|X={fixed_x:.2f}')
    plt.plot(y_vals, conditional_density_y_given_x(y_vals, fixed_x), label=f'Плотность Y|X={fixed_x:.2f}',
             color='darkgreen', linewidth=2)
    plt.title(f'Условная плотность Y при X={fixed_x:.2f}')
    plt.xlabel('Значение Y')
    plt.ylabel('Плотность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Генерация 10000 случайных величин
    x_samples, y_samples = generate_sample(1000000)

    # Вычисление корреляции
    correlation = compute_correlation(x_samples, y_samples)

    print(f"Корреляция между X и Y: {correlation:.4f}")

    # Построение 3D-графика
    plot_3d_distribution(x_samples, y_samples)

    # Построение условных плотностей
    plot_conditional_densities(x_samples, y_samples)
