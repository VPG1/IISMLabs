import random


class ComplexDependentGenerator(random.Random):
    def __init__(self, seed, p_a, p_b_if_a):
        super().__init__(seed)
        self.p_a = p_a
        self.p_b_if_a = p_b_if_a

    @staticmethod
    def probabilities(p_a, p_b_if_a):
        p_ab = p_a * p_b_if_a
        p_anb = p_a * (1 - p_b_if_a)
        p_nab = (1 - p_a) * (1 - p_b_if_a)
        p_nanb = (1 - p_a) * p_b_if_a

        return p_ab, p_anb, p_nab, p_nanb

    def random(self) -> int:
        rand_num = super().random()
        p_ab, p_anb, p_nab, p_nanb = self.probabilities(self.p_a, self.p_b_if_a)

        if rand_num <= p_ab:
            return 0
        elif p_ab < rand_num <= p_ab + p_anb:
            return 1
        elif p_ab + p_anb < rand_num <= p_ab + p_anb + p_nab:
            return 2
        else:
            return 3
