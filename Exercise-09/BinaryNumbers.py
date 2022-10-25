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


class Evolution():
    def __init__(self, max=256, size=10, base=8):
        self.size = size
        self.base = base
        self.max = max
        self.generations = []
        self.start_gen = np.random.randint(0, max, size=size)
        self.generations.append(self.start_gen)
        self.target = np.random.randint(0, max)

    def fitness(self, x: int):
        return -np.abs(x - self.target)

    def mutate(self, x: int):
        start = np.random.randint(0, math.floor(self.base/2))
        end = np.random.randint(start, math.floor(self.base))
        x_str = str(bin(x))[2:]

        if len(x_str) != self.base:
            for i in range(self.base-len(x_str)):
                x_str = '0' + x_str

        x_str = list(x_str)
        for i in range(start, end):
            x_str[i] = '1' if x_str[i] == '0' else '0'

        return int(''.join(x_str), 2)

    def combine(self, x, y):
        rnd = np.random.randint(0, 2)
        x_str = str(bin(x if rnd == 0 else y))[2:]
        y_str = str(bin(x if rnd == 1 else y))[2:]
        n = 0
        if len(x_str) % 2 == 0:
            n = len(x_str) / 2
        else:
            if self.fitness(x if rnd == 0 else y) > self.fitness(x if rnd == 1 else y):
                n = math.ceil(len(x_str) / 2)
            else:
                n = math.floor(len(x_str) / 2)

        n = int(n)
        child = x_str[:n]+y_str[n:]
        return int(child, 2)

    def get_new_combination(self, gen: zip):
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
            new_gen.append(self.combine(i[1], best))
            new_gen.append(self.combine(i[1], second))

        return new_gen

    def get_new_gen(self, old_gen):
        new_gen = self.get_new_combination(old_gen)

        for i in np.random.randint(0, self.size, size=math.floor(self.size/2)):
            new_gen[i] = self.mutate(new_gen[i])

        return new_gen

    def train(self):
        i = 0
        fs = []
        while True:

            fitnesses = [self.fitness(x) for x in self.generations[i]]
            fs.append(fitnesses)
            # Pick 5 best numbers
            gen = sorted(zip(fitnesses, self.generations[i]), reverse=True)[
                :(math.floor(self.size/2))]
            if max(fitnesses) == 0:
                print(self.generations)
                return [Gen(max(f), sum(f)/len(f)) for f in fs], self.generations

            new_gen = self.get_new_gen(gen)
            # print(new_gen)
            # print(self.target)
            self.generations.append(new_gen)

            i += 1

    def train_time(self):
        i = 0
        sts = []

        for t in range(20):
            print(i)
            st = time.time()
            while True:

                fitnesses = [self.fitness(x) for x in self.generations[i]]
                # Pick 5 best numbers
                gen = sorted(zip(fitnesses, self.generations[i]), reverse=True)[
                    :(math.floor(self.size/2))]
                if max(fitnesses) == 0:
                    et = time.time()
                    sts.append(et-st)
                    break
                # print(gen)
                new_gen = self.get_new_gen(gen)
                self.generations.append(new_gen)

                i += 1

        return sum(sts)/len(sts)


def print_info(gen, best, avg, i):
    print(f"Generation {i}, best: {best}, average: {avg}")
    print(gen)
    print("")


def task_1_1():
    evo = Evolution()

    gen, gens = evo.train()

    for i in range(len(gens)):
        print_info(gens[i], gen[i].best_fitness, gen[i].avg_fitness, i)

    print(f"Target: {evo.target}")


def task_1_2():
    n = np.arange(8, 18)
    time = []

    for i in range(8, 18):
        evo = Evolution(2**i, 20, i)
        time.append(evo.train_time())
        print(f"Done with: {2**i}")
        print(f"Target: {evo.target}")
    plt.xlabel("Bit length")
    plt.ylabel("Time (s)")
    plt.plot(n, time)
    plt.show()


if __name__ == "__main__":
    task_1_1()
    # task_1_2()
