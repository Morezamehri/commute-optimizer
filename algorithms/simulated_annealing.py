import math
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:


        current = self.problem.generate_initial_state()
        current_cost = self.problem.evaluate(current)

        best = current
        best_cost = current_cost
        temp = self.initial_temp

        self.history = [best_cost]

        for i in range(self.max_iterations):
            if temp <= self.min_temp:
                break

            neighbors = get_neighbors(current, self.problem)

            if not neighbors:
                break

            next_state = random.choice(neighbors)
            next_cost = self.problem.evaluate(next_state)

            delta = next_cost - current_cost

            if delta < 0:
                current = next_state
                current_cost = next_cost

            elif random.random() < math.exp(-delta / temp):
                current = next_state
                current_cost = next_cost

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

            self.history.append(best_cost)

            temp = temp * self.cooling_rate

        return best, best_cost
