import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)

        self.population_size = self.params.get("population_size", 30)
        self.generations = self.params.get("generations", 100)
        self.crossover_rate = self.params.get("crossover_rate", 0.8)
        self.mutation_rate = self.params.get("mutation_rate", 0.2)
        self.elitism = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:

        population = []

        for i in range(self.population_size):
            state = self.problem.generate_initial_state()
            population.append(state)

        best = min(population, key=lambda state: self.problem.evaluate(state))
        best_cost = self.problem.evaluate(best)

        self.history = [best_cost]

        for generation in range(self.generations):

            population.sort(key=lambda state: self.problem.evaluate(state))

            new_population = population[:self.elitism]

            while len(new_population) < self.population_size:

                candidates = random.sample(
                    population,
                    min(3, len(population))
                )
                parent1 = min(
                    candidates,
                    key=lambda state: self.problem.evaluate(state)
                )

                candidates = random.sample(
                    population,
                    min(3, len(population))
                )
                parent2 = min(
                    candidates,
                    key=lambda state: self.problem.evaluate(state)
                )

                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2, self.problem)
                else:
                    child = parent1

                if random.random() < self.mutation_rate:
                    neighbors = get_neighbors(child, self.problem)

                    if neighbors:
                        child = random.choice(neighbors)

                new_population.append(child)

            population = new_population

            current_best = min(
                population,
                key=lambda state: self.problem.evaluate(state)
            )
            current_cost = self.problem.evaluate(current_best)

            if current_cost < best_cost:
                best = current_best
                best_cost = current_cost

            self.history.append(best_cost)

        return best, best_cost