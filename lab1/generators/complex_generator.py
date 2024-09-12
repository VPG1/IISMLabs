import random


class ComplexGenerator(random.Random):
    def __init__(self, seed, probability_list):
        super().__init__(seed)
        self.probability_list = probability_list

    def random(self) -> list:
        res = list()
        for probability in self.probability_list:
            if super().random() <= probability:
                res.append(True)
            else:
                res.append(False)

        return res
