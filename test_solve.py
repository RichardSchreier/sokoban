"""Script for testing the Game.solve() function, or for generating solutions."""
from Game import Game, GameWorld

save_world = 1
solve_all = 0   # 0 to solve only unsolved

# for world_file in ["01", "02", "03"]:
for world_file in ["MB"]:
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

