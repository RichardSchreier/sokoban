import time
import math
from PriorityQueue import PriorityQueue
from engineering_notation import eng


def a_star(initial_state, max_cost=1000, max_time=30*60, max_states=50000, progress_report=None):
    """
:param initial_state: an object supporting the following methods: cost, neighbors, is_goal, heuristic.
:param max_cost: search terminates if the estimated cost exceeds max_cost
:param max_time: search terminates if the search time exceeds max_time
:param max_states: search terminates if the number of states visited exceeds max_states
:param progress_report: (progress_fn, progress_interval) progress_fn: a function called every progress_interval seconds. Arguments passed to progress_fn are (current_state, states_reached, priority_queue, elapsed_time). If progress_fn returns non-None, the search is terminated.
:return (solution_state, solution_info)
An implementation of the A* search algorithm.
See https://www.redblobgames.com/pathfinding/a-star/introduction.html
cost() is the cost associated with reaching this state from the initial state
neighbors() is an iterator returning neighboring states
is_goal() returns True if the state is a goal state
heuristic() returns an underestimate of the number of moves to reach a goal state
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
            return current, f"solution with cost = {current.cost()} found in {eng(time.time() - time0, 2)}s" +\
                   f" after examining {len(states_reached)} states."
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
            return_string = progress_fn(current, states_reached, frontier, time.time() - time0)
            if return_string is not None:
                msg = f"halted after {eng(time.time() - time0, 2)}s after examining {len(states_reached)} states. "
                return None, msg + return_string
            progress_update_time = time.time() + progress_interval

        for next_state in current.neighbors():
            next_added = False
            # if next_state not in states_reached or next_state.cost < cost of entry in states_reached
            try:
                i = states_reached.index(next_state)
                if states_reached[i].cost() > next_state.cost():
                    states_reached[i] = next_state
                    next_added = True
            except ValueError:
                states_reached.append(next_state)
                next_added = True
            if next_added:
                priority = next_state.cost() + next_state.heuristic()
                # Could the old version of next_state be in the priority queue?
                frontier.insert(next_state, priority)
    return None, "No solution."
