import random


class FullGroupGenerator(random.Random):
    def __init__(self, seed, probability_list):
        if sum(probability_list) != 1:
            raise ValueError("Sum of probabilities must be 1")
        super().__init__(seed)
        self.dist_function = [sum(probability_list[1:i + 1]) for i in range(len(probability_list))]

    def random(self) -> int:
        rand_num = super().random()
        for i in range(len(self.dist_function)):
            if rand_num <= self.dist_function[i]:
                return i

        return 0
