import networkx as nx
import matplotlib.pyplot as plt


class PetriNetF:
    def __init__(self, places, transitions, arcs, initial_marking):
        self.places = places
        self.transitions = transitions
        self.arcs = arcs
        self.initial_marking = initial_marking

    def can_fire(self, marking, transition):
        """Проверяет, можно ли сработать переход из текущей маркировки."""
        for (place, trans), weight in self.arcs.items():
            # Проверяем только те дуги, которые идут от места к переходу
            if trans == transition and place in marking:
                if marking[place] < weight:
                    return False
        return True

    def fire(self, marking, transition):
        """Срабатывает переход, возвращает новую маркировку."""
        new_marking = marking.copy()
        # Уменьшение маркеров на местах перед переходом
        for (place, trans), weight in self.arcs.items():
            if trans == transition:
                new_marking[place] -= weight
        # Добавление маркеров на места после перехода
        for (trans, place), weight in self.arcs.items():
            if trans == transition:
                new_marking[place] += weight
        return new_marking

    def get_successors(self, marking):
        """Возвращает список возможных маркировок после срабатывания переходов."""
        successors = []
        for transition in self.transitions:
            if self.can_fire(marking, transition):
                new_marking = self.fire(marking, transition)
                successors.append((transition, new_marking))
        return successors


def marking_to_str(marking):
    """Преобразует маркировку в строку для удобства отображения."""
    return str([marking[place] for place in sorted(marking.keys())])


# Определение сети Петри
places = ["P1", "P2", "P3", "P4", "P5"]
transitions = ["A", "B", "C", "D"]

# Дуги (арки), заданные вами
arcs = {
    ("A", "P1"): 1,  # Из A в P1
    ("P1", "C"): 1,  # Из P1 в C
    ("C", "P4"): 1,  # Из C в P4
    ("C", "P3"): 1,  # Из C в P3
    ("P4", "A"): 1,  # Из P4 в A
    ("P4", "D"): 1,  # Из P4 в D
    ("D", "P5"): 1,  # Из D в P5
    ("P3", "B"): 1,  # Из P3 в B
    ("B", "P2"): 1,  # Из B в P2
    ("P2", "C"): 1,  # Из P2 в C
}

# Начальная маркировка (можно изменить при необходимости)
initial_marking = {"P1": 0, "P2": 0, "P3": 1, "P4": 1, "P5": 0}

# Создание модели сети Петри
petri_net = PetriNetF(places, transitions, arcs, initial_marking)

# Создание графа для диаграммы маркировок
G = nx.DiGraph()

# Исследование всех возможных маркировок
visited = set()
to_visit = [initial_marking]

# Для отладки добавим вывод
print("Начальная маркировка:", marking_to_str(initial_marking))

while to_visit:
    current_marking = to_visit.pop()
    marking_str = marking_to_str(current_marking)

    if marking_str not in visited:
        print(f"Текущая маркировка: {marking_str}")
        visited.add(marking_str)
        successors = petri_net.get_successors(current_marking)

        if not successors:
            print(f"Переходы не могут сработать из состояния {marking_str}")

        for transition, new_marking in successors:
            new_marking_str = marking_to_str(new_marking)
            print(f" Переход '{transition}' срабатывает -> Новая маркировка: {new_marking_str}")
            G.add_edge(marking_str, new_marking_str, label=transition)

            if new_marking_str not in visited:
                to_visit.append(new_marking)

# Если не было сгенерировано ни одной новой маркировки
if len(G.nodes) == 0:
    print("Не было найдено ни одного нового состояния.")
else:
    # Рисование диаграммы маркировок
    pos = nx.spring_layout(G)  # Расположение вершин графа
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Диаграмма маркировок сети Петри')
    plt.show()
