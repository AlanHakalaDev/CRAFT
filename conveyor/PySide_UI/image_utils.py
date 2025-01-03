# queue for data
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# create a queue and finds the average
class MovingAverageQ:
    def __init__(self, size):
        self.size = size
        self.head = None
        self.tail = None
        self.sum = 0
        self.count = 0

    # adds a new number to the queue, released the first node
    def add(self, val):
        if self.count == self.size:
            self.sum -= self.head.data
            self.head = self.head.next
            self.count -= 1

        if self.head is None:
            self.head = Node(val)
            self.tail = self.head
        else:
            self.tail.next = Node(val)
            self.tail = self.tail.next

        self.sum += val
        self.count += 1

        return int(self.sum / self.count)

    # return the average of the items in the queue
    def get(self):
        return int(self.sum / self.count)