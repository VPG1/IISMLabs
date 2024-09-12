import time

from lab1.generators.simple_generator import SimpleGenerator
from lab1.generators.complex_generator import  ComplexGenerator

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