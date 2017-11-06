from Node import Node


class Layer:
    def __init__(self, node_count, *last_layer):
        if len(last_layer) == 0:
            self.last_layer = EmptyLayer()
        else:
            self.last_layer = last_layer[0]
        self.nodes = []
        for i in range(0, node_count):
            self.nodes.append(Node(self.last_layer))

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return "nodes:%s" % (self.nodes)

    def evaluate(self):
        for o in self.nodes:
            o.calc_val()


class EmptyLayer:
    def __init__(self):
        self.nodes = []

    def __len__(self):
        return 0

    def __str__(self):
        return "Empty Layer"

    def evaluate(self):
        pass
