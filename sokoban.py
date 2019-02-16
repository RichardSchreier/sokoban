#!../bin/python
# Original code downloaded from https://github.com/morenod/sokoban.git
"""
Key         Action
←↑↓→    Move worker left, up, down, right
n/p     Next/previous level
N/P     Next/previous unsolved level
>/<     Next/previous level file
q       Quit
r       Re-start
s       If no solution, solve from the current state;
        space   pause solve
        esc     exit solve
S       Solve from the initial state, even if a solution exists
R       Replay solution
u/U     Undo/Re-do
mouse   Move worker to specified square, can push box adjacent to worker

Debug
^a      show NO_BOX squares
^h      print heuristic
^n      print neighbors
^s      print solution string
"""
import sys
import os
import pygame
from Game import Game, GameWorld, UP, DOWN, LEFT, RIGHT
from Point import Point

WORLD_DIR = "Worlds"
SOKOBAN_INIT = "sokoban.init"


def main():
    def initialize_game():
        level_id = world.level_ids[min(level_i, len(world.level_ids))]
        world_id = worlds[min(world_i, len(worlds))]
        return Game(world.levels[level_i], f"{world_id}-{level_id}", True, world.solutions[level_i])

    def initialize_world():
        return GameWorld(worlds[world_i], WORLD_DIR)

    def read_world_i_and_level_i():
        try:
            file = open(SOKOBAN_INIT, "r")
            line = file.readline()
            file.close()
            words = line.split()
            return int(words[0]), int(words[1])
        except (FileNotFoundError, IndexError):
            return 1, 0

    def save_world_i_and_level_i():
        with open(SOKOBAN_INIT, "w") as file:
            file.write(f"{world_i} {level_i}")

    pygame.init()
    pygame.display.set_icon(pygame.image.load('Images/icon.png'))
    move_dict = {pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT}
    worlds = os.listdir(WORLD_DIR)
    worlds.sort()
    world_i, level_i = read_world_i_and_level_i()
    world = initialize_world()
    game = initialize_game()
    while 1:
        game.display()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                save_world_i_and_level_i()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in move_dict and not game.solved():
                    game.move(move_dict[event.key])
                elif event.unicode == 'n':  # go to next level
                    if level_i < len(world.levels) - 1:
                        level_i += 1
                        game = initialize_game()
                    elif world_i < len(worlds) - 1:
                        level_i = 0
                        world_i += 1
                        world = initialize_world()
                        game = initialize_game()
                elif event.unicode == 'N':  # go to next unsolved level
                    saved_vars = level_i, world_i, world
                    found_unsolved_game = False
                    while not found_unsolved_game:
                        if level_i < len(world.levels) - 1:
                            level_i += 1
                        elif world_i < len(worlds) - 1:
                            world_i += 1
                            world = initialize_world()
                            level_i = 0
                        else:
                            break
                        if not world.solutions[level_i][0]:
                            found_unsolved_game = True
                    if found_unsolved_game:
                        game = initialize_game()
                    else:
                        level_i, world_i, world = saved_vars
                elif event.unicode == 'p':  # go to previous level
                    if level_i > 0:
                        level_i -= 1
                        game = initialize_game()
                    elif world_i > 0:
                        world_i -= 1
                        world = initialize_world()
                        level_i = len(world.levels) - 1
                        game = initialize_game()
                elif event.unicode == 'P':  # go to previous unsolved level
                    saved_vars = level_i, world_i, world
                    found_unsolved_game = False
                    while not found_unsolved_game:
                        if level_i > 0:
                            level_i -= 1
                        elif world_i > 0:
                            world_i -= 1
                            world = initialize_world()
                            level_i = len(world.levels) - 1
                        else:
                            break
                        if not world.solutions[level_i][0]:
                            found_unsolved_game = True
                    if found_unsolved_game:
                        game = initialize_game()
                    else:
                        level_i, world_i, world = saved_vars
                elif event.unicode == 'r':
                    game.restart()
                elif event.unicode == 'R':
                    game.replay_solution()
                elif event.unicode == 's' and pygame.key.get_mods() is pygame.KMOD_NONE:
                    if game.solution_state is not None:
                        game.current_state = game.solution_state
                    else:
                        game.solve()
                        if game.solution_state:
                            world.update_solution(level_i, (game.solution_string(), game.solution_info))
                elif event.unicode == 'S':
                    game.current_state = game.initial_state
                    game.solve()
                    if game.solution_state:
                        world.update_solution(level_i, (game.solution_string(), game.solution_info))
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_META:
                    world.save()
                elif event.unicode == 'u':
                    game.undo()
                elif event.unicode == 'U':
                    game.redo()
                elif event.unicode == '>':
                    world_i = min(world_i + 1, len(worlds) - 1)
                    world = initialize_world()
                    level_i = 0
                    game = initialize_game()
                elif event.unicode == '<':
                    world_i = max(world_i - 1, 0)
                    world = initialize_world()
                    level_i = 0
                    game = initialize_game()
                # debug commands
                elif event.key is pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    game.show_annotated_map = True
                elif event.key is pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    print(f"heuristic = {game.current_state.heuristic()}")
                elif event.key is pygame.K_n and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    print("Neighbors:")
                    for i, n in enumerate(game.current_state.neighbors()):
                        print(f"{i}: {len(n.previous_moves)} moves")
                        n.full_map.print()
                    print("Done")
                elif event.key is pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    print(game.solution_string())
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    game.show_annotated_map = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.solved():
                game.move_to(pygame.mouse.get_pos())
        pygame.display.update()


if __name__ == '__main__':
    main()
    sys.exit(0)
