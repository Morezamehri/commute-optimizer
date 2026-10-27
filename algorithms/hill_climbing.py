from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:

        best = self.problem.generate_initial_state()
        best_cost = self.problem.evaluate(best)

        self.history = [best_cost]

        for restart in range(self.max_restarts + 1):

            if restart == 0:
                current = best
            else:
                current = self.problem.generate_initial_state()

            current_cost = self.problem.evaluate(current)
            sideways = 0

            while True:
                neighbors = get_neighbors(current, self.problem)

                if not neighbors:
                    break

                next_state = min(
                    neighbors,
                    key=lambda state: self.problem.evaluate(state)
                )
                next_cost = self.problem.evaluate(next_state)

                if next_cost < current_cost:
                    current = next_state
                    current_cost = next_cost
                    sideways = 0

                elif next_cost == current_cost and sideways < self.max_sideways:
                    current = next_state
                    current_cost = next_cost
                    sideways += 1

                else:
                    break

                if current_cost < best_cost:
                    best = current
                    best_cost = current_cost

                self.history.append(best_cost)

        return best, best_cost
