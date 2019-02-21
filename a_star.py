import time
import math
from PriorityQueue import PriorityQueue
from engineering_notation import eng


def a_star(initial_state, max_cost=1000, max_time=30*60, max_states=50000, progress_report=None):
    """
An implementation of the A* search algorithm.
See https://www.redblobgames.com/pathfinding/a-star/introduction.html
    initial_state   an object supporting the following methods:
        cost        the cost associated with reaching this state from the initial state
        neighbors   an iterator returning neighboring states
        is_goal     returns True if the state is a goal state
        heuristic   returns an underestimate of the number of moves to reach a goal state
    max_cost        search terminates if an underestimate of the cost exceeds max_cost
    max_time        search terminates if the search time exceeds max_time
    max_states      search terminates if the number of states visited exceeds max_states
    progress_report (progress_fn, progress_interval)
        progress_fn a function called every progress_interval seconds.
        Arguments passed to progress_fn are (current_state, states_reached, priority_queue, elapsed_time).
        If progress_fn returns non-None, the search is terminated.

    Returns (solution_state, solution_info)
"""
    frontier = PriorityQueue(initial_state, 0)
    states_reached = [initial_state]
    time0 = time.time()
    termination_condition = ""
    if progress_report:
        progress_fn, progress_interval = progress_report
        progress_update_time = time0
    else:
        progress_fn, progress_interval = None, None
        progress_update_time = math.inf

    while not frontier.empty():
        current = frontier.pop()
        if current.is_goal():
            return current, f"cost-optimal solution (cost = {current.cost()}) found in {eng(time.time() - time0, 2)}s" \
                          + f" after examining {len(states_reached)} states."
        # These checks could be in the next_state loop
        if time.time() - time0 > max_time:
            termination_condition += f"Time limit ({max_time}s) exceeded."
        if current.cost() > max_cost:
            termination_condition += f"Max cost ({max_cost}) exceeded."
        if len(states_reached) > max_states:
            termination_condition += f"Max states ({max_states}) exceeded."
        if termination_condition:
            return None, termination_condition
        if time.time() > progress_update_time:
            return_value = progress_fn(current, states_reached, frontier, time.time() - time0)
            if return_value is not None:
                msg = f"halted after {eng(time.time() - time0, 2)}s after examining {len(states_reached)} states. "
                if type(return_value) is str:
                    msg += return_value
                return None, msg
            progress_update_time = time.time() + progress_interval

        for next_state in current.neighbors():
            next_added = False
            try:
                i = states_reached.index(next_state)
                if states_reached[i].cost() > next_state.cost():
                    states_reached[i] = next_state
                    next_added = True
            except ValueError:
                states_reached.append(next_state)
                next_added = True
            if next_added:
                h = next_state.heuristic()
                # if h is 0:
                #     print(f'Found a solution with cost = {next_state.cost()} (possibly non-optimal).')
                priority = next_state.cost() + h
                frontier.insert(next_state, priority)
    return None, "No solution."
