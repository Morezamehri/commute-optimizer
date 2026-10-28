from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)

        self.beam_width = self.params.get("beam_width", 5)
        self.max_iterations = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
 
        beam = []

        for i in range(self.beam_width):
            state = self.problem.generate_initial_state()
            beam.append(state)

        best = min(beam, key=lambda state: self.problem.evaluate(state))
        best_cost = self.problem.evaluate(best)

        self.history = [best_cost]

        for iteration in range(self.max_iterations):

            all_neighbors = []

            for state in beam:
                neighbors = get_neighbors(state, self.problem)
                all_neighbors.extend(neighbors)

            if not all_neighbors:
                break

            unique_neighbors = list(set(all_neighbors))

            unique_neighbors.sort(
                key=lambda state: self.problem.evaluate(state)
            )

            new_beam = unique_neighbors[:self.beam_width]

            current_best = min(
                new_beam,
                key=lambda state: self.problem.evaluate(state)
            )
            current_cost = self.problem.evaluate(current_best)

            if current_cost >= best_cost:
                break

            beam = new_beam
            best = current_best
            best_cost = current_cost

            self.history.append(best_cost)

        return best, best_cost