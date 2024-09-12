import os
import time

from lab1.generators.full_group_generator import FullGroupGenerator


def show_donation_table():
    all_donations = sum(donation_table.values())
    print('Название\tСумма\tВероятность')
    for name, price in sorted(donation_table.items()):
        print(f'{name}\t\t{price}\t\t{round((price / all_donations) * 100, 5)}%')


def donate():
    print('Введите название игры: ')
    name = input()

    print('Введите сумму пожертвования')
    try:
        donation = int(input())
    except ValueError:
        print('Сумма пожертвование должно быть числом')
        input()
        return

    if donation_table.get(name) is None:
        donation_table[name] = donation
    else:
        donation_table[name] += donation


def spin():
    all_donations = sum(donation_table.values())
    donation_list = [(name, donation) for name, donation in donation_table.items()]

    generator = FullGroupGenerator(time.time(), [donation / all_donations for _, donation in donation_list])
    res_index = generator.random()

    print(f'Победитель: {donation_list[res_index][0]} '
          f'с шансом {round((donation_list[res_index][1] / all_donations) * 100, 5)}%')


donation_table = {}


while True:
    print('1. Таблица пожертвований')
    print('2. Сделать пожертвование')
    print('3. Крутить колесо')

    try:
        choice = int(input())
        if choice != 1 and choice != 2 and choice != 3:
            raise ValueError
    except ValueError:
        print('Невалидный ввод')
        continue

    if choice == 1:
        show_donation_table()
    elif choice == 2:
        donate()
    elif choice == 3:
        spin()
        input()
        exit(0)

    os.system('clear')
