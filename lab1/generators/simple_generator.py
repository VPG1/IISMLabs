import random


class SimpleGenerator(random.Random):
    def __init__(self, seed, probability):
        super().__init__(seed)
        self.probability = probability

    def random(self):
        random_num = super().random()

        if random_num <= self.probability:
            return True

        return False
