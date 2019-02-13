class Point:
    """A 2D point class."""
    __slots__ = ['x', 'y']  # Prevents the creation of an instance dictionary

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __eq__(self, d):  # to make == work
        return self.x == d.x and self.y == d.y

    def __hash__(self):  # to support dictionary indexing
        return hash((self.x, self.y))

    def __bool__(self):  # to support conditionals
        return self.x is not None and self.y is not None

    def __iter__(self):  # to support tuple(point)
        yield self.x
        yield self.y

    def __add__(self, d):
        return Point(self.x + d.x, self.y + d.y)

    def __sub__(self, d):
        return Point(self.x - d.x, self.y - d.y)

    def __neg__(self):   # unary minus
        return Point(-self.x, -self.y)

    def __mul__(self, k):   # to support point * k; also supports k * point
        return Point(self.x * k, self.y * k)

    def __str__(self):
        return f"({self.x},{self.y})"

    def __repr__(self):
        return "Point" + self.__str__()

    def __lt__(self, p2):   # to allow a list of Points to be sorted
        return self.x < p2.x or (self.x == p2.x and self.y < p2.y)

    @property
    def l1_norm(self):
        return abs(self.x) + abs(self.y)
