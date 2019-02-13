from PriorityQueue import PriorityQueue
from Point import Point


def find_path(a, b, is_open):
    """
    :param a: Start Point
    :param b: Finish Point
    :param is_open: Function returning True if the Point argument is an open square
    :return: A list of Points containing the moves needed to get from a to b
    """
    if a == b:
        return []
    if not is_open(b):
        return None
    moves = direct_path(a, b, is_open) or rectilinear_path(a, b, is_open) or find_path_using_a_star(a, b, is_open)
    return moves


def sgn(x):
    if x > + 0:
        return 1
    else:
        return -1


def sign_and_magnitude(x):
    if x >= 0:
        return 1, x
    else:
        return -1, -x


def hv45_path(a, b, is_open):
    """Horizontal or vertical path with a 45-degree tail"""
    horizontal_move = Point(sgn(b.x - a.x), 0)
    vertical_move = Point(0, sgn(b.y - a.y))
    moves = []
    while a != b:
        if abs(a.x - b.x) > abs(a.y - b.y):
            move = horizontal_move
        else:
            move = vertical_move
        a += move
        if not is_open(a):
            return None
        moves.append(move)
    return moves


# I suspect a first-order delta-sigma modulator could give the same result
def direct_path(a, b, is_open):
    dx, n_x = sign_and_magnitude(b.x - a.x)
    dy, n_y = sign_and_magnitude(b.y - a.y)
    n = n_x + n_y
    p = a
    moves = []
    m = 0
    while p != b:
        m += 1
        p_m = a + (b - a) * (m / n)
        if (p + Point(dx, 0) - p_m).l1_norm <= (p + Point(0, dy) - p_m).l1_norm:
            move = Point(dx, 0)
        else:
            move = Point(0, dy)
        p += move
        if not is_open(p):
            return None
        moves.append(move)
    return moves


def rectilinear_path(a, b, is_open):
    def check_moves(moves):
        p = a
        for move in moves:
            p += move
            if not is_open(p):
                return None
        return moves

    horizontal_moves = [Point(sgn(b.x - a.x), 0)] * abs(b.x - a.x)
    vertical_moves = [Point(0, sgn(b.y - a.y))] * abs(b.y - a.y)
    if abs(a.x - b.x) > abs(a.y - b.y):
        return check_moves(horizontal_moves + vertical_moves) or \
               (vertical_moves and check_moves(vertical_moves + horizontal_moves))
    else:
        return check_moves(vertical_moves + horizontal_moves) or \
               (horizontal_moves and check_moves(horizontal_moves + vertical_moves))


def find_path_using_a_star(a, b, is_open):
    """Find a path from a to b using a uniform-step-cost version of the A* algorithm"""
    def successor(p_):
        moves = [Point(1, 0), Point(-1, 0), Point(0, 1), Point(0, -1)]
        for move in moves:
            pp = p_ + move
            if is_open(pp):
                yield pp

    def traceback():
        moves = []
        p_ = b
        pp = predecessor(p_)
        while pp:
            moves.append(p_ - pp)
            p_ = pp
            pp = predecessor(p_)
        moves.reverse()
        return moves

    # I am not comfortable using a dict to hold values that I would normally stuff into an array
    def predecessor(p_):
        return node_dict[p_][0]

    def g(p_):
        return node_dict[p_][1]

    def h(p_):
        return (p_ - b).l1_norm

    pq = PriorityQueue(a, h(a))
    node_dict = {a: (None, 0)}  # dict contents are (predecessor, g)
    while pq:
        p0 = pq.pop()
        if p0 == b:
            return traceback()
        g1 = g(p0) + 1
        for p in successor(p0):
            f = g1 + h(p)
            if p not in node_dict:  # or f < existing_f:  Not necessary since nodes are on a uniform-cost grid
                pq.insert(p, f)
                node_dict[p] = p0, g1
    return None


def test():
    def initialize_map():
        nonlocal a, b, matrix
        board = """
###############
#             #
#       #     #
#       #     #
#       #     #
#a      #     #
#       #     #
#       #     #
############# #
#       b     #
###############
"""
        lines = board.splitlines()
        y = 0
        for line in lines:
            if line:
                matrix.append(list(line))
                if not a and 'a' in line:
                    a = Point(line.index('a'), y)
                    matrix[a.y][a.x] = ' '
                if not b and 'b' in line:
                    b = Point(line.index('b'), y)
                    matrix[b.y][b.x] = ' '
                y += 1

    def mark_map():
        matrix[a.y][a.x] = 'a'
        matrix[b.y][b.x] = 'b'
        p = a
        if moves:
            for move in moves:
                p += move
                if p != b:
                    matrix[p.y][p.x] = '+'
            if p != b:
                matrix[p.y][p.x] = 'X'

    def print_map():
        for line in matrix:
            print(str.join("", line))

    def is_open(p):
        if p.x < 0 or p.y < 0:
            return False
        try:
            return matrix[p.y][p.x] is " "
        except IndexError:
            return False

    a, b, matrix = Point(), Point(), []
    initialize_map()
    moves = find_path(a, b, is_open)
    mark_map()
    print_map()


if __name__ == '__main__':
    test()
