import random
from typing import List

from core.state import State
from core.problem import CommuteProblem


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:

    neighbors = []
    path = list(state.path)

    for change in [-problem.t_step, problem.t_step]:
        new_time = state.departure_time + change

        if problem.t_min <= new_time <= problem.t_max:
            new_state = State(tuple(path), new_time)

            if problem.is_valid_state(new_state) and new_state not in neighbors:
                neighbors.append(new_state)

    for i in range(len(path) - 1):
        start = path[i]

        for j in range(i + 1, len(path)):
            end = path[j]

            for node in problem.network.get_neighbors(start):
                if node == end:
                    continue

                if not problem.network.has_edge(node, end):
                    continue

                if node in path[:i + 1] or node in path[j:]:
                    continue

                new_path = path[:i + 1] + [node] + path[j:]
                new_state = State(tuple(new_path), state.departure_time)

                if problem.is_valid_state(new_state) and new_state not in neighbors:
                    neighbors.append(new_state)

    return neighbors


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:

    path1 = list(parent1.path)
    path2 = list(parent2.path)

    times = problem.get_valid_time_domain()
    average_time = (parent1.departure_time + parent2.departure_time) / 2
    new_time = min(times, key=lambda t: abs(t - average_time))

    common_nodes = []

    for i in range(1, len(path1) - 1):
        for j in range(1, len(path2) - 1):
            if path1[i] == path2[j]:
                common_nodes.append((i, j))

    random.shuffle(common_nodes)

    for i, j in common_nodes:
        new_path = path1[:i] + path2[j:]

        child = State(tuple(new_path), new_time)

        if problem.is_valid_state(child):
            return child

    child = State(tuple(path1), new_time)

    if problem.is_valid_state(child):
        return child

    return parent1