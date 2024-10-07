import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Place:
    def __init__(self, name, tokens=0):
        self.name = name
        self.tokens = tokens


class Transition:
    def __init__(self, name):
        self.name = name


class PetriNet:
    def __init__(self):
        self.places = []
        self.transitions = []
        self.arcs = {}  # arc: (place, transition)

    def add_place(self, place):
        self.places.append(place)

    def add_transition(self, transition):
        self.transitions.append(transition)

    def add_arc(self, vertex1, vertex2):
        self.arcs[(vertex1.name, vertex2.name)] = (vertex1, vertex2)

    def fire_transition(self, transition_name):
        # Проверяем возможность срабатывания перехода
        for (vertex1_name, vertex2_name), (vertex1, vertex2) in self.arcs.items():
            if vertex2.name == transition_name and vertex1.tokens == 0:
                return False

        for (vertex1_name, vertex2_name), (vertex1, vertex2) in self.arcs.items():
            if vertex1_name == transition_name:
                vertex2.tokens += 1
            if vertex2_name == transition_name:
                vertex1.tokens -= 1

        return True

    def visualize(self):
        G = nx.DiGraph()

        # Добавляем места в граф
        for place in self.places:
            G.add_node(place.name)

        # Добавляем переходы в граф
        for transition in self.transitions:
            G.add_node(transition.name)

        # Добавляем дуги
        for (vertex1_name, vertex2_name) in self.arcs.keys():
            G.add_edge(vertex1_name, vertex2_name)

        return G


class PetriNetGUI:
    def __init__(self, root, petri_net):
        self.root = root
        self.petri_net = petri_net
        self.root.title("Petri Net Simulation")

        self.create_widgets()
        self.canvas = None  # Переменная для хранения виджета canvas
        self.node_positions = self.set_node_positions()  # Установка статичных координат
        self.draw_graph()  # Рисуем граф сразу при инициализации

    def create_widgets(self):
        tk.Label(self.root, text="Petri Net Simulation").pack()

        self.place_frame = tk.Frame(self.root)
        self.place_frame.pack()

        tk.Label(self.place_frame, text="Places:").grid(row=0, column=0)
        for place in self.petri_net.places:
            tk.Label(self.place_frame, text=f"{place.name}: {place.tokens} tokens").grid(
                row=self.petri_net.places.index(place) + 1, column=0)

        self.transition_frame = tk.Frame(self.root)
        self.transition_frame.pack()

        tk.Label(self.transition_frame, text="Transitions:").grid(row=0, column=0)
        for transition in self.petri_net.transitions:
            tk.Button(self.transition_frame, text=f"Fire {transition.name}",
                      command=lambda t=transition.name: self.fire_transition(t)).grid(
                row=self.petri_net.transitions.index(transition) + 1, column=0)

        # Кнопка для построения диаграммы состояний
        self.state_diagram_button = tk.Button(self.root, text="Show State Diagram", command=self.show_state_diagram)
        self.state_diagram_button.pack()

    def set_node_positions(self):
        # Устанавливаем статичные координаты для узлов
        return {
            "P1": (1, 1),   # Координаты для места P1
            "P2": (2, 1),   # Координаты для места P2
            "P3": (2, -1),  # Координаты для места P3
            "P4": (1, -1),  # Координаты для места P4
            "P5": (0, -1),  # Координаты для места P5
            "a": (0.5, 0),  # Координаты для перехода a
            "b": (2.5, 0),  # Координаты для перехода b
            "c": (1.5, 0),  # Координаты для перехода c
            "d": (0.5, -1)   # Координаты для перехода d
        }

    def fire_transition(self, transition_name):
        if self.petri_net.fire_transition(transition_name):
            messagebox.showinfo("Success", f"Transition {transition_name} fired successfully.")
        else:
            messagebox.showerror("Error", f"Cannot fire transition {transition_name}.")
        self.update_place_display()
        self.draw_graph()  # Обновляем граф после выполнения перехода

    def update_place_display(self):
        for widget in self.place_frame.winfo_children():
            widget.destroy()

        tk.Label(self.place_frame, text="Places:").grid(row=0, column=0)
        for place in self.petri_net.places:
            tk.Label(self.place_frame, text=f"{place.name}: {place.tokens} tokens").grid(
                row=self.petri_net.places.index(place) + 1, column=0)

    def draw_graph(self):
        G = self.petri_net.visualize()

        # Очищаем старый граф
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        # Создаем новую фигуру для графа
        plt.clf()  # Очищаем текущую фигуру

        # Используем статичные координаты
        pos = self.node_positions
        labels = {node: node if node not in self.node_positions else f"{node}\nTokens: {self.get_tokens(node)}" for node in G.nodes()}

        # Рисуем граф
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
        plt.title("Petri Net Visualization")

        # Встраиваем график в Tkinter
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def get_tokens(self, node_name):
        # Возвращаем количество токенов для отображения
        for place in self.petri_net.places:
            if place.name == node_name:
                return place.tokens
        return ""

    def show_state_diagram(self):
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

    def generate_states(self, G, visited_states, current_petri_net):
        # Получаем текущее состояние
        current_state = self.get_current_state(current_petri_net)
        if current_state in visited_states:
            return
        visited_states.add(current_state)

        # Добавляем текущее состояние как вершину графа
        G.add_node(current_state)

        # Определяем переходы и создаем новые состояния
        for transition in self.petri_net.transitions:
            # Создаем копию текущих токенов для имитации перехода
            temp_petri_net = PetriNet()
            temp_petri_net.places = [Place(p.name, p.tokens) for p in current_petri_net.places]
            temp_petri_net.transitions = current_petri_net.transitions.copy()
            temp_petri_net.arcs = current_petri_net.arcs.copy()

            if temp_petri_net.fire_transition(transition.name):
                # Создаем новое состояние после перехода
                new_state = self.get_current_state(temp_petri_net)
                G.add_node(new_state)
                G.add_edge(current_state, new_state)  # Добавляем ребро между состояниями
                # Рекурсивно генерируем состояния для нового состояния
                self.generate_states(G, visited_states, temp_petri_net)

    def get_current_state(self, petri_net=None):
        if petri_net is None:
            petri_net = self.petri_net

        state_labels = []
        for place in petri_net.places:
            state_labels.append(f"{place.name}: {place.tokens}")
        return ', '.join(state_labels)


def main():
    # Создаем сеть Петри с указанными местами и переходами
    petri_net = PetriNet()

    # Создаем места и переходы
    p1 = Place("P1", tokens=0)
    p2 = Place("P2", tokens=0)
    p3 = Place("P3", tokens=1)
    p4 = Place("P4", tokens=1)
    p5 = Place("P5", tokens=0)

    a = Transition("a")
    b = Transition("b")
    c = Transition("c")
    d = Transition("d")

    # Добавляем места и переходы в сеть Петри
    petri_net.add_place(p1)
    petri_net.add_place(p2)
    petri_net.add_place(p3)
    petri_net.add_place(p4)
    petri_net.add_place(p5)

    petri_net.add_transition(a)
    petri_net.add_transition(b)
    petri_net.add_transition(c)
    petri_net.add_transition(d)

    petri_net.add_arc(a, p1)
    petri_net.add_arc(p1, c)
    petri_net.add_arc(c, p3)
    petri_net.add_arc(c, p4)
    petri_net.add_arc(p4, a)
    petri_net.add_arc(p4, d)
    petri_net.add_arc(d, p5)
    petri_net.add_arc(p3, b)
    petri_net.add_arc(b, p2)
    petri_net.add_arc(p2, c)

    # Создаем GUI
    root = tk.Tk()
    gui = PetriNetGUI(root, petri_net)
    root.mainloop()
##


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





if __name__ == "__main__":
    main()




