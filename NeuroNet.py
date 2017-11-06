import time
import json
import Layer
import Node

class NeuroNet():
    def __init__(self, *params):
        self.__create_net(params)

    def __create_net(self, params):
        i = len(params)
        if i < 3:
            print("Too few Arguments")
            exit(1)
        self.layers = []
        self.hl_nodes = []
        print(params)
        i = len(params)
        for o in params:
            self.hl_nodes.append(o)

        self.layers.append(Layer.EmptyLayer())
        for i in range(0, len(self.hl_nodes)):
            self.layers.append(Layer.Layer(self.hl_nodes[i], self.layers[i]))

    def export_net(self, file):
        with open(file, 'w+') as f:
            f.write(json.dumps(self, cls=MyNetEncoder, sort_keys=True, indent=4))
        print("Exported net: \"%s\"" % (file))

    def import_net(self, file):
        with open(file, 'r') as f:
            j = json.load(f)
            print("Importing net: \"%s\" %s" % (file, tuple(j['settings'])))
            self.__create_net(tuple(j['settings']))
            for h, o in enumerate(self.layers):
                for i, p in enumerate(o.nodes):
                    p.d = j['layers'][h][i]['bias']
                    p.weight_j = j['layers'][h][i]['wji']

    def train(self, file, max_amt=100000, verbose=False, epsilon=0.01, learn_rate=0.9):
        self.learn_rate = learn_rate
        with open(file) as f:
            train = json.load(f)
        print("Training net with \"%s\" for max. %i times or until a accuracy of %f and a learn rate of %f" %
              (file, max_amt, epsilon, learn_rate))
        i=0
        err_ = 1
        t = time.time()
        while i < max_amt and float(err_) > epsilon:
            i+=1
            err_ = 0
            o=0
            for o, j in enumerate(train):
                input = j[0]
                target = j[1]
                out = []
                err = []

                out = self.evaluate(input)
                for k in range(0, len(target)):
                    e = out[k]*(1-out[k])*(target[k]-out[k])
                    self.layers[len(self.layers)-1].nodes[k].err = e
                    err.append(e)
                if verbose:
                    print("gerneration:%i, in:%s, target:%s, out:%s, error:%s, total error:%s" %
                     (i, input, target, out, err, sum(err)))
                err_ += abs(sum(err))
                self.back_propergate()
            err_ /= o
            if verbose:
                print("Gen error:%f"%err_)
        t = time.time()-t
        print("Finished training with \"%s\" for %i times and it took %.3f seconds" % (file, i, t))


    def back_propergate(self):
        i = len(self.layers)-2
        while i > 0: #every layer except the last one, startig from the second to last one
            for k, o in enumerate(self.layers[i].nodes): #for every node in the layer
                # j: number of the node
                # o: node object
                e = 0   # error for the node
                for v in self.layers[i+1].nodes: # for every node in the next layer
                    e += v.calc_err(k)
                o.err = e * (o.value * (1-o.value))

                for v in self.layers[i+1].nodes:
                    v.weight_j[k] += self.learn_rate * v.err * o.value

                o.d += self.learn_rate * o.err
            i -= 1
        for i in self.layers[len(self.layers)-1].nodes:
            i.d += self.learn_rate * i.err



    def evaluate(self, *args):
        for i, o in enumerate(self.layers[1].nodes):
            o.value = args[0][i]
        for i in range(2, len(self.layers)):
            self.layers[i].evaluate()

        lst = []
        for o in self.layers[len(self.layers)-1].nodes:
            lst.append(o.value)
        return lst

    def __str__(self):
        return "nodes:%s r:%f" % (self.hl_nodes, self.learn_rate)


class MyNetEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Node.Node):
            return{"bias": o.d, "wji":o.weight_j}
        if isinstance(o, Layer.Layer):
            return o.nodes
        if isinstance(o, Layer.EmptyLayer):
            return
        if isinstance(o, NeuroNet):
            return{"settings":o.hl_nodes, "layers":o.layers}


if __name__ == "__main__":
    n = NeuroNet(2, 3, 1)
    n.train("xor.json", 10000000, False, 0.001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", 10000000, False, 0.0001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", 10000000, False, 0.00001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    """print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", epsilon=0.001, verbose=False)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", epsilon=0.0001, verbose=False)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", epsilon=0.00001, verbose=False)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.export_net("xor_net.json")
    n.train("and.json", epsilon=0.00001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("nand.json", epsilon=0.00001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("or.json", epsilon=0.00001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.train("xor.json", epsilon=0.00001, verbose=False)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.import_net("xor_net.json")
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    """
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.export_net("xor_net.json")
    n.train("or.json", epsilon=0.001)
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    n.import_net('xor_net.json')
    print("[1, 1] equals:%s" % n.evaluate([1, 1]))
    i=1









