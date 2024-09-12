import time

from lab1.generators.simple_generator import SimpleGenerator
from lab1.generators.complex_generator import ComplexGenerator
from lab1.generators.complex_dependent_generator import ComplexDependentGenerator
from lab1.generators.full_group_generator import FullGroupGenerator

# Task1
a = SimpleGenerator(time.time(), 0.5)

success = 0
for i in range(0, 10 ** 6):
    if a.random():
        success += 1

print(success / 10 ** 6)

# task 2
probability_list = [0.3, 0.7]
b = ComplexGenerator(time.time(), probability_list)

successes = [0] * len(probability_list)
for _ in range(0, 10 ** 6):
    curRes = b.random()
    for i in range(len(curRes)):
        if curRes[i]:
            successes[i] += 1

print([success / 10 ** 6 for success in successes])

# Task 3
c = ComplexDependentGenerator(time.time(), 0.6, 0.7)
print(c.probabilities(0.6, 0.7))

events = [0, 0, 0, 0]
for _ in range(0, 10 ** 6):
    events[c.random()] += 1

print([el / 10 ** 6 for el in events])

# Task 4
group_probability_list = [0.1, 0.2, 0.7]
print(group_probability_list)
d = FullGroupGenerator(time.time(), group_probability_list)

events = [0] * len(group_probability_list)
for _ in range(0, 10 ** 6):
    event_index = d.random()
    events[event_index] += 1

print([el / 10 ** 6 for el in events])
