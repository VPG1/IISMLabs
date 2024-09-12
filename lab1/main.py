import time

from lab1.generators.simple_generator import SimpleGenerator
from lab1.generators.complex_generator import ComplexGenerator
from lab1.generators.complex_dependent_generator import ComplexDependentGenerator

a = SimpleGenerator(time.time(), 0.5)

success = 0
for i in range(0, 10**6):
    if a.random():
        success += 1

print(success / 10**6)


probability_list = [0.3, 0.7]
b = ComplexGenerator(time.time(), probability_list)

successes = [0] * len(probability_list)
for _ in range(0, 10**6):
    curRes = b.random()
    for i in range(len(curRes)):
        if curRes[i]:
            successes[i] += 1


print([success / 10**6 for success in successes])


c = ComplexDependentGenerator(time.time(), 0.6, 0.7)
print(c.probabilities(0.6, 0.7))

events = [0, 0, 0, 0]
for _ in range(0, 10**6):
    events[c.random()] += 1

print([el / 10**6 for el in events])