"""Sokoban game class"""
import math
import pygame
import os
import sys
import datetime
import time
from itertools import permutations
from copy import deepcopy
from find_path import find_path
from Point import Point
from a_star import a_star

WALL = '#'
SPACE = ' '
GOAL = '.'
WORKER = '@'
WORKER_ON_GOAL = '+'
BOX = '$'
BOX_ON_GOAL = '*'
NO_BOX = 'X'  # Only used in annotated_map
CAUTION_BOX = 'x'
# TODO: Switch from list to string
WORKER_CHARS = [WORKER, WORKER_ON_GOAL]
BOX_CHARS = [BOX, BOX_ON_GOAL]
OPEN_FOR_BOX = [SPACE, GOAL]
OPEN_FOR_WORKER = OPEN_FOR_BOX + [NO_BOX]
GOAL_CHARS = [GOAL, BOX_ON_GOAL, WORKER_ON_GOAL]
VALID_CHARS = frozenset([WALL] + OPEN_FOR_BOX + WORKER_CHARS + BOX_CHARS)
file_dict = {WALL: 'wall', SPACE: 'floor', BOX: 'box', BOX_ON_GOAL: 'box_docked', WORKER: 'worker',
             WORKER_ON_GOAL: 'worker_dock', GOAL: 'dock', NO_BOX: 'no_box', CAUTION_BOX: 'caution_box'}
blit_dict = {key: pygame.image.load(f'Images/{file}.png') for (key, file) in file_dict.items()}
CELL_SIZE = 32
UP = Point(0, -1)
DOWN = Point(0, 1)
LEFT = Point(-1, 0)
RIGHT = Point(1, 0)
MOVE_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
MAX_LINE_LENGTH = 120


class GameWorld:
    def __init__(self, filename=None, directory=""):
        self.full_filename = os.path.join(directory, filename)
        self.levels = []
        self.title = None
        self.author = None
        self.contents = []
        self.level_ids = []
        self.level_index = []
        self.solutions = []
        if filename:
            self.read()

    def read(self):
        """Read a file to define the levels of that world"""
        # File format:
        # Optional Header consisting of any text
        #   ; Title =
        #   ; Author =
        #   ; URL =
        # Level description (Level name can be specified before or after, with or without ;)
        #   ; Solution = {UDLR}
        def line_is_level_description():
            return bool(line) and not bool(set(line).difference(VALID_CHARS))

        def rhs_of_assignment():
            ii = line.find('=')
            if ii + 1 < len(line):
                return line[ii + 1:].strip()
            else:
                return ""

        def add_level():
            if matrix:
                self.level_ids.append(level_id)
                self.levels.append(matrix)
                self.level_index.append(starting_line_number)
                self.solutions.append((solution, solution_info))

        with open(self.full_filename, 'r') as file:
            matrix = []
            solution, solution_info = None, None
            level_id = None
            preceding_comment = None
            header_passed = False
            reading_a_level = False
            logical_line_number = 0
            actual_line_number = 1
            for line in file:
                line = line.rstrip('\n')
                if line_is_level_description():
                    header_passed = True
                    if not reading_a_level:
                        starting_line_number = logical_line_number
                        if not level_id and preceding_comment:
                            level_id = preceding_comment.lstrip('; ').strip
                    reading_a_level = True
                    matrix.append(list(line))
                elif line.startswith(';'):
                    if line[1:].lstrip().startswith('Solution ='):
                        while line.rstrip().endswith('\\'):     # Support multi-line solution comments
                            line = line.rstrip(' \\') + file.readline().rstrip(' \n').lstrip(' ;')
                            actual_line_number += 1
                        solution = rhs_of_assignment()
                    elif line[1:].lstrip().startswith('Solution_info ='):
                        solution_info = rhs_of_assignment()
                    elif (reading_a_level or header_passed) and not level_id:
                        level_id = line.lstrip('; ').rstrip()
                    reading_a_level = False
                    if not header_passed:
                        if 'Title' in line:
                            self.title = rhs_of_assignment()
                        elif 'Author' in line:
                            self.author = rhs_of_assignment()
                elif line.strip() == "":
                    header_passed = True
                    add_level()
                    matrix = []
                    solution, solution_info = None, None
                    level_id = None
                    preceding_comment = None
                    reading_a_level = False
                elif line.startswith("Level"):
                    level_id = line[5:].rstrip('\n').strip()
                    assert level_id
                else:
                    print(f"Problem in line {actual_line_number} of {self.full_filename}.")
                    sys.exit(1)  # ToDo: Throw an error instead?
                self.contents.append(line)
                logical_line_number += 1
                actual_line_number += 1
            if self.contents[-1] is not '':
                self.contents.append('')
            add_level()

    def save(self):
        with open(self.full_filename, 'w') as file:
            for line in self.contents:
                if len(line) > MAX_LINE_LENGTH and line[0] == ';' and line[1:].lstrip().startswith('Solution ='):
                    # Support multi-line solution comments
                    while len(line) > MAX_LINE_LENGTH:
                        file.write(line[:MAX_LINE_LENGTH - 1] + '\\\n')
                        line = '; ' + line[MAX_LINE_LENGTH - 1:]
                file.write(line + '\n')

    def check_and_update_solution(self, level_i, solution):
        if not self.solutions[level_i]:
            self.update_solution(level_i, solution)
        else:
            # Compare existing solution against new one
            old_solution = self.solutions[level_i]
            level_id = self.level_ids[level_i]
            l1 = len(old_solution[0])
            l2 = len(solution[0])
            if l1 > l2:
                print(f':Old solution for level {level_id} is longer than new solution ({l1} vs. {l2}). Updating.')
                self.update_solution(level_id, solution)
            elif l1 < l2:
                print(f':Old solution for level {level_id} is shorter than new solution ({l1} vs. {l2}). Not updating.')

    def update_solution(self, level_i, solution):
        self.solutions[level_i] = solution
        line_number = self.level_index[level_i]
        solution_line_updated = False
        solution_info_updated = False
        solution_line = f';Solution = {solution[0]}'
        solution_info = f';Solution_info = {solution[1]}'
        while self.contents[line_number]:
            line = self.contents[line_number]
            if line[0] == ';':
                if line[1:].lstrip().startswith('Solution ='):
                    self.contents[line_number] = solution_line
                    solution_line_updated = True
                elif line[1:].lstrip().startswith('Solution_info ='):
                    self.contents[line_number] = solution_info
                    solution_info_updated = True
            line_number += 1
        lines_added = 0
        if not solution_info_updated:
            self.contents.insert(line_number, solution_info)
            lines_added += 1
        if not solution_line_updated:
            self.contents.insert(line_number, solution_line)
            lines_added += 1
        if lines_added:
            for i in range(level_i + 1, len(self.level_index)):
                self.level_index[i] += lines_added


class Game:
    def __init__(self, full_map, level_id, initialize_screen=True, solution=(None, None)):
        self.raw_map = GameMap(full_map)
        worker, boxes, goals = self.raw_map.make_raw()
        self.annotated_map = deepcopy(self.raw_map)
        self.annotated_map.annotate(worker)
        self.show_annotated_map = False
        self.initial_state = GameState(worker, boxes, self)
        self.current_state = self.initial_state
        self.goals = goals
        self.goals.sort()
        self.level_id = level_id
        if solution[0]:
            self.solution_state = self.verify_solution(solution[0])
            if self.solution_state:
                self.solution_info = solution[1]
        else:
            self.solution_state = None
            self.solution_info = ""
        if initialize_screen:
            self.screen = pygame.display.set_mode(self.size)  # Generates KEYUP events for all active keys, incl. SHIFT!
        else:
            self.screen = None

    @property
    def size(self):
        x, y = self.raw_map.size
        return x * CELL_SIZE, y * CELL_SIZE

    @staticmethod
    def grid_point(screen_pos):
        return Point(screen_pos[0] // CELL_SIZE, screen_pos[1] // CELL_SIZE)

    def display(self):
        if self.show_annotated_map:
            self.display_annotated_map()
        else:
            self.display_full_map()

    def display_full_map(self):
        self.current_state.full_map.display(self.screen)
        if self.solved():
            caption = f"Solved in {self.current_state.move_count} moves!!"
        else:
            caption = f"{self.level_id}, move {self.current_state.move_count}."
        pygame.display.set_caption(caption)

    def display_annotated_map(self):
        self.annotated_map.display(self.screen)
        pygame.display.set_caption("Annotated Map")

    def solved(self):
        return self.current_state.solved()

    def restart(self):
        self.current_state = self.initial_state

    def move(self, d):
        new_state = self.current_state.move(d)
        if new_state:
            self.current_state = new_state
            self.solution_state = None

    def undo(self):
        if self.current_state.predecessor:
            successor = self.current_state
            self.current_state = self.current_state.predecessor
            self.current_state.successor = successor

    def redo(self):
        if self.current_state.successor:
            self.current_state = self.current_state.successor

    def move_to(self, screen_pos):
        p = Game.grid_point(screen_pos)
        if self.current_state.full_map[p] in BOX_CHARS and (self.current_state.worker - p).l1_norm == 1:
            self.move(p - self.current_state.worker)
        else:
            new_state = self.current_state.move_to(p)
            if new_state:
                self.current_state = new_state

    def replay_solution(self):
        def display(_state):
            _state.full_map.display(self.screen)
            pygame.display.update()
            pygame.event.pump()  # Need to ping the event queue to actually update the display
            time.sleep(0.1)

        if not self.solution_state:
            return
        solution_states = [self.solution_state]
        while solution_states[-1].predecessor:
            solution_states.append(solution_states[-1].predecessor)
        solution_states.reverse()
        display(solution_states[0])
        for state in solution_states[1:]:
            intermediate_state = state.predecessor
            for move in state.previous_moves[:-1]:
                intermediate_state = intermediate_state.move(move)
                display(intermediate_state)
            display(state)
        self.current_state = self.solution_state

    def solve(self):
        # Todo: animate the solution process; respond to keyboard
        solution, solution_info = self.current_state.solve()
        self.solution_state = solution
        date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.solution_info = f"{date_str}, {solution_info}"
        if solution is not None:
            self.current_state = solution
            if self.screen:
                print(solution_info, self.solution_string())

    def solution_string(self):
        """Convert the solution to a string"""
        if self.solution_state is None:
            return ""
        move_dict = {DOWN: 'd', UP: 'u', RIGHT: 'r', LEFT: 'l'}
        solution_states = [self.solution_state]
        while solution_states[-1].predecessor:
            solution_states.append(solution_states[-1].predecessor)
        solution_states.reverse()
        move_string = ""
        if len(solution_states) > 1:
            for state in solution_states[1:]:
                intermediate_state = state.predecessor
                for move in state.previous_moves:
                    if intermediate_state.full_map[intermediate_state.worker + move] in OPEN_FOR_WORKER:
                        move_string += move_dict[move]
                    else:
                        assert intermediate_state.full_map[intermediate_state.worker + move] in BOX_CHARS
                        move_string += move_dict[move].upper()
                    intermediate_state = intermediate_state.move(move)
        return move_string

    def verify_solution(self, solution_str):
        move_dict = {'d': DOWN, 'u': UP, 'r': RIGHT, 'l': LEFT}
        current_state = self.initial_state
        for c in solution_str:
            d = move_dict[c.lower()]
            current_state = current_state.move(d)
        if current_state.solved():
            return current_state
        else:
            print(f"Saved solution for {self.level_id} didn't work!")


class GameState:
    def __init__(self, worker, boxes, game, predecessor=None, previous_moves=None):
        self.worker = worker
        self.boxes = boxes
        self.boxes.sort()
        self.game = game
        self.predecessor = predecessor
        self.successor = None
        if predecessor:
            if previous_moves:
                self.move_count = predecessor.move_count + len(previous_moves)
                self.previous_moves = previous_moves
            else:
                self.move_count = predecessor.move_count + 1
                self.previous_moves = [worker - predecessor.worker]
        else:
            self.move_count = 0
            self.previous_moves = None
        self._full_map = None

    @property
    def full_map(self):  # full_map uses lazy initialization; ToDo: Try using a try statement
        if getattr(self, '_full_map', None) is None:
            self._full_map = self.game.raw_map.fill(self.worker, self.boxes)
        return self._full_map

    def __eq__(self, game_state):  # to make == work and to support set of GameStates
        return self.worker == game_state.worker and set(self.boxes) == set(game_state.boxes)

    def __hash__(self):  # to make == work and to support set of GameStates
        return hash((self.worker,))

    def solved(self):
        return self.boxes == self.game.goals

    def move(self, d):
        new_worker = self.worker + d
        if self.full_map[new_worker] in OPEN_FOR_WORKER:
            return GameState(new_worker, self.boxes, self.game, self)
        elif self.full_map[new_worker] in BOX_CHARS and self.full_map[new_worker + d] in OPEN_FOR_BOX:
            return self.move_box(new_worker, d)
        else:
            return None

    def move_box(self, new_worker, d, moves=None):
        i = self.boxes.index(new_worker)
        new_boxes = self.boxes.copy()
        new_boxes[i] = new_worker + d
        return GameState(new_worker, new_boxes, self.game, self, moves)

    def move_to(self, p):
        moves = find_path(self.worker, p, self.full_map.open_for_worker)
        if moves:
            return GameState(p, self.boxes, self.game, self, moves)

    def check_2x2(self, b, dd):
        """Check the 2x2 square with one corner at b and the opposite corner ad b + dd.
        The box is deadlocked if the square contains only walls or boxes, and at least one box is not on a goal."""
        walls = 0
        c = self.full_map[b]
        if c in SPACE + WORKER:     # The box will be pushed to b
            boxes = 1
            boxes_on_goal = 0
        else:
            assert c in GOAL + WORKER_ON_GOAL
            boxes = 0
            boxes_on_goal = 1
        for d in [Point(dd.x, 0), Point(dd.x, dd.y), Point(0, dd.y)]:
            c = self.full_map[b + d]
            if c == WALL:
                walls += 1
            elif c == BOX:
                boxes += 1
            elif c == BOX_ON_GOAL:
                boxes_on_goal += 1
        if boxes + boxes_on_goal + walls == 4 and boxes != 0:
            return True
        else:
            return False

    def count_boxes_and_goals(self, p0, d):
        """Count the number of boxes and goals along the line segment perpendicular to d"""
        n_boxes = 0
        n_goals = int(self.full_map[p0] is GOAL)
        for dp in [Point(d.y, -d.x), Point(-d.y, d.x)]:
            p = p0
            while True:
                p += dp
                c = self.full_map[p]
                if c in GOAL_CHARS:
                    n_goals += 1
                elif c in BOX_CHARS:
                    n_boxes += 1
                elif c is WALL:
                    break
        return n_boxes, n_goals

    def anticipate_deadlock(self, b, d):
        """Check if moving the box at b in the direction d causes a deadlock"""
        b = b + d
        if self.game.annotated_map[b] is NO_BOX:
            return True
        if self.game.annotated_map[b] is CAUTION_BOX and self.game.annotated_map[b - d] is not CAUTION_BOX:
            n_boxes, n_goals = self.count_boxes_and_goals(b, d)
            if n_boxes >= n_goals:
                return True
        # Check two 2x2 boxes in the direction of d
        dp = Point(d.y, -d.x)   # perpendicular to d
        for dd in [d + dp, d - dp]:
            if self.check_2x2(b, dd):
                return True
        return False

    def number_of_boxes_along_path(self, p, pp):
        """Count the number of boxes on the path joining p to pp"""
        n = 0
        if p.x == pp.x:
            for b in self.boxes:
                if b.x == p.x and (b.y - p.y) * (b.y - pp.y) <= 0:
                    n += 1
        else:
            assert p.y == pp.y
            for b in self.boxes:
                if b.y == p.y and (b.x - p.x) * (b.x - pp.x) <= 0:
                    n += 1
        return n

    def neighbors(self):
        """Generate neighboring states by looking for boxes that can be pushed."""
        for b_i, b in enumerate(self.boxes):
            for d in MOVE_DIRECTIONS:
                new_worker = b - d
                if self.full_map[b + d] in OPEN_FOR_BOX + WORKER_CHARS and not self.anticipate_deadlock(b, d):
                    moves = find_path(self.worker, new_worker, self.full_map.open_for_worker)
                    if moves is not None:
                        moves.append(d)
                        new_state = self.move_box(b, d, moves)  # todo: check if new_state is new?
                        if not self.predecessor or new_state != self.predecessor:
                            yield new_state

    def is_goal(self):  # TODO: see if I can pass this as an argument. Use getattr(object, name[, default])
        return self.solved()

    def cost(self):  # TODO: see if I can pass this as an argument. Use getattr(object, name[, default])
        return self.move_count

    # heuristic() underestimates the number of moves needed to reach a solution; used by a_star().
    def heuristic(self):
        """Sum the distances from each box to the nearest goal."""
        if self.solved():
            return 0
        move_sum = 0
        for b in self.boxes:
            min_move_count = math.inf
            for g in self.game.goals:
                min_move_count = min(min_move_count, (b - g).l1_norm)
            move_sum += min_move_count
        return move_sum

    def heuristic1(self):
        """Sum the distances from each box to the nearest goal."""
        if self.solved():
            return 0
        move_sum = 0
        for b in self.boxes:
            min_move_count = math.inf
            for g in self.game.goals:
                min_move_count = min(min_move_count, len(find_path(b, g, self.game.raw_map.open_for_worker)))
            move_sum += min_move_count
        return move_sum

    def heuristic0(self):
        """For each pairing of boxes with goals, count the number of moves to the box and then to the associated goal.
        Return the lowest move count. THIS HEEURISTIC IS NOT NECESSARILY AN UNDERESTIMATE"""
        if self.solved():
            return 0
        min_move_count = math.inf
        for p in permutations(range(len(self.boxes))):
            move_count = 0
            worker = self.worker
            for i, goal_i in enumerate(p):
                b = self.boxes[i]
                g = self.game.goals[goal_i]
                try:
                    move_count += len(find_path(worker, b, self.game.raw_map.open_for_worker))
                    move_count += len(find_path(b, g, self.game.annotated_map.open_for_worker))
                except TypeError:
                    move_count = math.inf
                    break
                if move_count >= min_move_count:
                    break
                worker = g
            if move_count < min_move_count:
                min_move_count = move_count
        return min_move_count

    def solve(self):
        """Use the A-star algorithm to search for a solution"""
        def progress_fn(state, states_seen, pq, elapsed_time):
            if self.game.screen:
                t = f"{elapsed_time:6.1f}s: "
                s = f"{len(states_seen)} states, "
                q = f"{len(pq)} queued, "
                m = f"{state.move_count + state.heuristic()} moves"
                print(t + s + q + m)
                state.full_map.display(self.game.screen)
                pygame.display.set_caption(t + m)
                pygame.display.flip()
                pygame.event.pump()   # Need to tap the event queue in order for the display to update
                first_pass = True
                pause = False
                while first_pass or pause:
                    first_pass = False
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return False    # Todo: Should exit
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                return False
                            elif event.unicode == ' ':
                                pause = not pause

        return a_star(self, 1000, 60 * 60, 200000, (progress_fn, 0.5))


class GameMap:
    def __init__(self, matrix):
        self.matrix = deepcopy(matrix)

    def make_raw(self):
        def line_find(chars):
            for i, char in enumerate(line):
                if char in chars:
                    yield i

        worker = None
        boxes = []
        goals = []
        for y, line in enumerate(self.matrix):
            for x in line_find(WORKER_CHARS):
                assert worker is None  # Only one worker is allowed
                p = Point(x, y)
                worker = p
                self.remove_worker(p)
            for x in line_find(BOX_CHARS):
                p = Point(x, y)
                boxes.append(p)
                self.remove_box(p)
            for x in line_find(GOAL_CHARS):
                goals.append(Point(x, y))
        assert worker is not None
        assert len(boxes) == len(goals)
        return worker, boxes, goals

    def annotate(self, worker):
        def constrained_path(c1, c2):
            return c1 is WALL or c2 is WALL or (c1 is NO_BOX and c2 is NO_BOX)

        def check_direction():
            map_modified = False
            pp = p + d
            dp = Point(d.y, -d.x)
            while self[pp] in SPACE and constrained_path(self[pp + dp], self[pp - dp]):
                pp += d
            if self[pp] in WALL + NO_BOX and pp - d != p:
                pp -= d
                map_modified = True
                while pp != p:
                    self[pp] = NO_BOX
                    pp -= d
            return map_modified

        def fill_inaccessible_with_wall():
            accessible_map = deepcopy(self)
            frontier = [worker]
            while frontier:
                p = frontier.pop()
                accessible_map[p] = WORKER
                for d in MOVE_DIRECTIONS:
                    pp = p + d
                    if accessible_map[pp] in SPACE + GOAL:
                        frontier.append(pp)
            for y, row in enumerate(self.matrix):
                for x, c in enumerate(row):
                    p = Point(x, y)
                    if self[p] is SPACE and accessible_map[p] is not WORKER:
                        self[p] = WALL

        fill_inaccessible_with_wall()
        # mark inside corners with NO_BOX or CAUTION_BOX
        marker_dict = {SPACE: NO_BOX, GOAL: CAUTION_BOX}
        corners = []
        for y, row in enumerate(self.matrix):
            for x, c in enumerate(row):
                p = Point(x, y)
                if c in SPACE + GOAL and \
                        (self[p + LEFT] == WALL or self[p + RIGHT] == WALL) and \
                        (self[p + UP] == WALL or self[p + DOWN] == WALL):
                    self[p] = marker_dict[c]
                    corners.append(p)
        # Repeatedly join NO_BOX squares along wall edges or similarly constrained vertical/horizontal paths
        keep_going = True
        while keep_going:
            keep_going = False
            for y, row in enumerate(self.matrix):
                for x, c in enumerate(row):
                    if c in NO_BOX:
                        p = Point(x, y)
                        for d in [DOWN, RIGHT]:    # Only need to check in 2 directions
                            if check_direction():
                                keep_going = True
        # Join corners along constrained vertical/horizontal paths containing goal with CAUTION_BOX
        disallow_box = WALL + NO_BOX
        for p in corners:
            for d in [DOWN, RIGHT]:
                pp = p + d
                dp = Point(d.y, -d.x)
                while self[pp] in SPACE + GOAL and constrained_path(self[pp + dp], self[pp - dp]):
                    pp += d
                if self[pp] in disallow_box and pp - d != p:
                    pp -= d
                    while pp != p:
                        self[pp] = CAUTION_BOX
                        pp -= d

    def fill(self, worker, boxes):
        # ToDo: only copy rows that are changed
        full_map = deepcopy(self)
        full_map.add_worker(worker)
        for b in boxes:
            full_map.add_box(b)
        return full_map

    # query/set map contents using self[point]
    def __getitem__(self, point):
        if point.x < 0 or point.y < 0:
            return WALL
        try:
            return self.matrix[point.y][point.x]
        except IndexError:
            return WALL

    def __setitem__(self, point, content):
        if point.x < 0 or point.y < 0:
            raise IndexError
        self.matrix[point.y][point.x] = content

    def display(self, screen):
        background = 255, 226, 191
        screen.fill(background)
        for y, row in enumerate(self.matrix):
            for x, c in enumerate(row):
                screen.blit(blit_dict[c], (x * CELL_SIZE, y * CELL_SIZE))

    def remove_box(self, p):
        if self[p] == BOX:
            self[p] = SPACE
        else:
            assert self[p] == BOX_ON_GOAL
            self[p] = GOAL

    def add_box(self, p):
        if self[p] == SPACE:
            self[p] = BOX
        else:
            assert self[p] == GOAL
            self[p] = BOX_ON_GOAL

    def remove_worker(self, p):
        if self[p] == WORKER:
            self[p] = SPACE
        else:
            assert self[p] == WORKER_ON_GOAL
            self[p] = GOAL

    def add_worker(self, p):
        if self[p] == SPACE:
            self[p] = WORKER
        else:
            assert self[p] == GOAL
            self[p] = WORKER_ON_GOAL

    @property
    def size(self):
        x = 0
        y = len(self.matrix)
        for row in self.matrix:
            if len(row) > x:
                x = len(row)
        return x, y

    def print(self):
        for row in self.matrix:
            print(str.join("", row))

    def open_for_worker(self, p):
        return self[p] in OPEN_FOR_WORKER

    def open_for_box(self, p):
        return self[p] in OPEN_FOR_BOX
