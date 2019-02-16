class PriorityQueue:
    """A queue in which the items are stored in increasing priority"""
    def __init__(self, item=None, priority=None):
        if priority is None:
            self.queue = []
        elif isinstance(priority, list):
            self.queue = list(zip(priority, item))
            self.check()
        else:
            self.queue = [(priority, item)]

    def __len__(self):
        return len(self.queue)

    def empty(self):
        return len(self.queue) == 0

    def insert_using_linear_search(self, item, priority):
        for i in range(len(self.queue)):
            priority_i, _ = self.queue[i]
            if priority_i >= priority:
                self.queue.insert(i, (priority, item))
                return
        self.queue.append((priority, item))

    def insert(self, item, priority):
        # Binary search
        m = -1
        n = len(self.queue)
        while n > m + 1:
            i = (m + n) // 2
            priority_i, _ = self.queue[i]
            if priority_i < priority:    # < ensures most recent insertion is above equal-valued items
                m = i
            else:
                n = i
        self.queue.insert(n, (priority, item))

    def pop(self):
        try:
            priority, item = self.queue[0]
            del self.queue[0]
            return item
        except IndexError:
            return None

    def check(self):
        """Verify priorities are non-decreasing"""
        p0 = None
        for priority, _ in self.queue:
            if p0 is not None:
                assert priority >= p0
            p0 = priority
        self.print()

    def print(self):
        for priority, item in self.queue:
            print(f"{priority:5d}, {item}")
        print()


def __test__():
    pq = PriorityQueue("a", 5)
    pq.check()
    pq.insert("b", 6)
    pq.check()
    pq.insert("c", 3)
    pq.check()
    pq.insert("d", 3)
    pq.check()
    pq.insert("e", 4)
    pq.check()
    item = pq.pop()
    assert item is "d"


if __name__ == '__main__':
    __test__()
