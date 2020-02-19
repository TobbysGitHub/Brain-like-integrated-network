class LoopQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * self.capacity
        self.counter = 0

    def offer(self, x):
        index = self.counter % self.capacity

        x_ = self.queue[index]

        self.queue[index] = x

        self.counter += 1

        return x_

    def peek(self):
        index = self.counter % self.capacity

        return self.queue[index]


def main():
    l = LoopQueue(4)
    print(l.peek())

    for i in range(5):
        print(i, ': ', l.offer(i))
        print(i, ': ', l.peek())


if __name__ == '__main__':
    main()
