import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Gen:
    best_fitness: int
    avg_fitness: float

    def __init__(self, best, avg):
        self.best_fitness = best
        self.avg_fitness = avg


array_size: int
base_value: int
max_value: int
generations = []
target: int


def mutation(x: int):
    start = np.random.randint(0, math.floor(base_value/2))
    end = np.random.randint(start, math.floor(base_value))
    x_str = str(bin(x))[2:]

    if len(x_str) != base_value:
        for i in range(base_value-len(x_str)):
            x_str = '0' + x_str

    x_str = list(x_str)
    for i in range(start, end):
        x_str[i] = '1' if x_str[i] == '0' else '0'

    return int(''.join(x_str), 2)


def combine(x, y):
    rnd = np.random.randint(0, 2)
    x_str = str(bin(x if rnd == 0 else y))[2:]
    y_str = str(bin(x if rnd == 1 else y))[2:]
    n = 0
    if len(x_str) % 2 == 0:
        n = len(x_str) / 2
    else:
        if fitness(x if rnd == 0 else y) > fitness(x if rnd == 1 else y):
            n = math.ceil(len(x_str) / 2)
        else:
            n = math.floor(len(x_str) / 2)

    n = int(n)
    child = x_str[:n]+y_str[n:]
    return int(child, 2)


def fitness(x: int):
    return -np.abs(x - target)


def get_new_combination(gen: zip):
    new_gen = []
    t = 0
    best = 0
    second = 0

    for i in range(len(gen)):
        if best > gen[i][0] > second != gen[i][1]:
            second = gen[i][1]

        if gen[i][0] > best != gen[i][1]:
            second = best
            best = gen[i][1]

    for i in gen:
        new_gen.append(combine(i[1], best))
        new_gen.append(combine(i[1], second))

    return new_gen


def get_new_gen(old_gen):
    new_gen = get_new_combination(old_gen)

    for i in np.random.randint(0, array_size, size=math.floor(array_size/2)):
        new_gen[i] = mutation(new_gen[i])

    return new_gen


def train():
    i = 0
    fs = []
    while True:

        fitnesses = [fitness(x) for x in generations[i]]
        fs.append(fitnesses)
        # Pick 5 best numbers
        gen = sorted(zip(fitnesses, generations[i]), reverse=True)[
            :(math.floor(array_size/2))]
        if max(fitnesses) == 0:
            print(generations)
            return [Gen(max(f), sum(f)/len(f)) for f in fs], generations

        new_gen = get_new_gen(gen)
        # print(new_gen)
        # print(self.target)
        generations.append(new_gen)

        i += 1


def train_time():
    i = 0
    sts = []

    for t in range(20):
        print(i)
        st = time.time()
        while True:

            fitnesses = [fitness(x) for x in generations[i]]
            # Pick 5 best numbers
            gen = sorted(zip(fitnesses, generations[i]), reverse=True)[
                :(math.floor(array_size/2))]
            if max(fitnesses) == 0:
                et = time.time()
                sts.append(et-st)
                break
            # print(gen)
            new_gen = get_new_gen(gen)
            generations.append(new_gen)

            i += 1

    return sum(sts)/len(sts)


def print_gens(gen, best, avg, i):
    print(f"Gen {i}: ")
    print(gen)
    print(f"Best Fitness: {best}")
    print(f"Avg Fitness: {avg}")
    print("")


def new_evo(new_max_value, new_size, new_base, target):
    array_size = new_size
    base_value = new_base
    max_value = new_max_value
    start_gen = np.random.randint(0, max_value, size=array_size)
    generations.append(start_gen)
    target = np.random.randint(0, max_value)
    print(f"pleaaaase {target}")


def task_1_1():

    new_evo(256, 10, 8, target)

    gen, gens = train()

    for i in range(len(gens)):
        print_gens(gens[i], gen[i].best_fitness, gen[i].avg_fitness, i)

    print(f"Target: {target}")


def task_1_2():
    n = np.arange(8, 18)
    time = []

    for i in range(8, 18):
        new_evo(2**i, 20, i)
        time.append(train_time())
        print(f"Done with: {2**i}")
        print(f"Target: {target}")
    plt.xlabel("Bit length")
    plt.ylabel("Time (s)")
    plt.plot(n, time, target)
    plt.show()


task_1_1()
# task_1_2()
