import json
from random import random
from math import exp


class Node:
    def __init__(self, last_layer):
        self.d = random()*2-1
        self.weight_j = []
        self.value = 0
        self.last_layer = last_layer
        self.err = 0
        for i in last_layer.nodes:
            self.weight_j.append(random()*2-1)

    def calc_val(self):
        self.value = self.d
        i=0
        for i in range(0, len(self.last_layer.nodes)):
            self.value += self.last_layer.nodes[i].value * self.weight_j[i]
        self.value = 1/(1+exp(-self.value))

    def calc_err(self, index):
        return self.err * self.weight_j[index]

    def __float__(self):
        return self.value

    def __str__(self):
        return self.value

