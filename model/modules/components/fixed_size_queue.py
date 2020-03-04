class FixedSizeQueue:
    '''
    一个固定容量的队列，当充满才能读取队列开头的元素。当充满后，加入新元素会挤出队列开头的元素。
    '''
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
    l = FixedSizeQueue(4)

    for i in range(3):
        assert l.offer(i) is None and l.peek() is None

    assert l.offer(3) is None
    assert l.peek() == 0
    assert l.offer(4) == 0
    assert l.peek() == 1
    assert l.offer(5) == 1

if __name__ == '__main__':
    main()
