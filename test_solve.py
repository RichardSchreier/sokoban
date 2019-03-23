"""Script for testing the Game.solve() function, or for generating solutions."""
import os
from Game import Game, GameWorld

save_world = 1
do_solve = False
solve_all = 0   # 0 to solve only unsolved
print_stats = 1

WORLD_DIR = "Worlds"
world_files = sorted(os.listdir(WORLD_DIR))

if print_stats:
    for world_file in world_files:
        world = GameWorld(world_file, "Worlds")
        levels = world.levels
        n_unsolved, n_solved_optimally, n_solved_manually, = 0, 0, 0
        for level_i, solution in enumerate(world.solutions):
            if solution[0] is None:
                n_unsolved += 1
                # print(world.level_ids[level_i])
            elif "cost-optimal" in solution[1]:
                n_solved_optimally += 1
            elif "manual moves" in solution[1] or "manually solved" in solution[1]:
                n_solved_manually += 1
            else:
                print(f"unintelligible solution info: {solution[1]}")
        print(f'{world_file} contains {n_unsolved} unsolved, {n_solved_optimally} optimally solved, and ' +
              f'{n_solved_manually} (at least partially) manually-solved puzzles.')


if do_solve:
    for world_file in world_files:
        world = GameWorld(world_file, "Worlds")
        levels = world.levels
        for level_i in range(len(levels)):
            level_id = world.level_ids[level_i]
            full_id = f"{world_file}#{level_id}"
            game = Game(levels[level_i], level_id, False, world.solutions[level_i])
            if solve_all or game.solution_state is None:
                game.solve()
                msg1 = f"{full_id}: {game.solution_info}"
                if game.solution_state:
                    msg2 = world.check_and_update_solution(level_i, (game.solution_string(), game.solution_info))
                    if save_world:
                        world.save()
                else:
                    msg2 = ""
                print(f"{msg1:110} {msg2}")

