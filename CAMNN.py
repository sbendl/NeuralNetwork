import tensorflow as tf
import numpy as np
import random
from math import cos, sin, pi, inf
import matplotlib.pyplot as plt

def sample_from_circle(center, radius):
    r = random.uniform(0, radius)
    theta = random.uniform(0, 2*pi)
    return [center[0]+r*cos(theta), center[1]+r*sin(theta)]

points = []
for i in range(1000):
    points.append(sample_from_circle((2,2), .25))
    points.append(sample_from_circle((3,2), .1))
    points.append(sample_from_circle((2,3), .7))
    points.append(sample_from_circle((2,1), .3))


def midpoint(p1, p2):
    return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2

class NeuralGas:
    def __init__(self, epsb=.1, epsn=.005, max_age=25, lam=100, alpha=.2, d=.9):
        self.nodes = set()
        self.edges = set()
        self.epsb = epsb
        self.epsn = epsn
        self.max_age = max_age
        self.lam = lam
        self.alpha = alpha
        self.d = d
        self.num_sigs = 0

        n1 = Node()
        n2 = Node()
        self.add_connection(n1, n2)

    def show(self, pointsin=None):
        plt.subplot().clear()
        plt.subplot().set_title(self.num_sigs)
        plt.subplot().set_title(str(self.num_sigs) + "\n" + str(sum([n.error for n in self.nodes])))
        if pointsin:
            #print(pointsin)
            plt.scatter([point[0] for point in pointsin], [point[1] for point in pointsin], alpha=1, c='r')
        plt.scatter([point[0] for point in points], [point[1] for point in points], alpha=.1)
        for edge in self.edges:
            plt.plot([v.w[0] for v in list(edge.nodes)],[v.w[1] for v in list(edge.nodes)])
        plt.pause(.01   )

    def add_connection(self, a, b):
        e = Edge(a, b)
        a.connect(e)
        b.connect(e)

        if a not in self.nodes:
            self.nodes.add(a)
        if b not in self.nodes:
            self.nodes.add(b)
        if e not in self.edges:
            self.edges.add(e)

    def connected(self, a, b):
        for e in self.edges:
            if a in e.nodes and b in e.nodes:
                return e
        return False

    def find_closest_two(self, signal):
        closest = None
        closest_dist = inf
        next_closest = None
        next_closest_dist = inf
        for node in self.nodes:
            if closest_dist > node.distance_from(signal):
                next_closest_dist = closest_dist
                next_closest = closest
                closest_dist = node.distance_from(signal)
                closest = node
            elif next_closest_dist > node.distance_from(signal):
                next_closest_dist = node.distance_from(signal)
                next_closest = node

        return closest, next_closest



    def train_signal(self, signal):
        #print(len(self.nodes))
        #print(max(self.nodes, key=lambda n:n.error).error)
        self.num_sigs += 1
        s1, s2 = self.find_closest_two(signal)
        s1.age_edges()
        connection = self.connected(s1, s2)
        if connection:
            connection.age = 0
        else:
            self.add_connection(s1, s2)

        s1.calc_error(signal)

        s1.creep_towards(signal, self.epsb)
        for s in s1.neighbors():
            s.creep_towards(signal, self.epsn)

        for e in self.edges.copy():
            if e.age > self.max_age:
                for n in e.nodes:
                    n.disconnect(e)
                self.edges.remove(e)

        self.nodes = {n for n in self.nodes if len(n.edges) > 0}

        if self.num_sigs % self.lam == 0:

            q = max(self.nodes, key=lambda n:n.error)
            f = max(q.neighbors(), key=lambda n:n.error)

            self.show([f.w, q.w])
            r = Node()
            r.w = [.5 * (x + y) for x, y in zip(q.w, f.w)]
            q.error *= self.alpha
            r.error = q.error
            f.error *= self.alpha
            f_to_q = self.connected(f, q)
            f.disconnect(f_to_q)
            q.disconnect(f_to_q)
            self.edges.remove(f_to_q)
            f_to_r = Edge(f, r)
            q_to_r = Edge(q, r)
            self.nodes.add(r)
            f.connect(f_to_r)
            q.connect(q_to_r)
            r.connect(f_to_r)
            r.connect(q_to_r)

            self.edges.add(f_to_r)
            self.edges.add(q_to_r)
            #plt.scatter([f.w[0], r.w[0], q.w[0]], [f.w[1], r.w[1], q.w[1]], c='g')
            #plt.pause(1)
            self.show([f.w, r.w, q.w])


        for n in self.nodes:
            n.error *= self.d


class Node:
    def __init__(self, dim=2):
        self.w = [random.uniform(0, 10) for d in range(dim)]
        self.error = 0
        self.edges = set()

    def neighbors(self):
        return {(e.nodes-{self}).pop() for e in self.edges}

    def connect(self, edge):
        self.edges.add(edge)

    def disconnect(self, edge):
        self.edges.discard(edge)

    def distance_from(self, vec):
        return abs(sum([(x-y)**2 for (x, y) in zip(self.w, vec)])**0.5)

    def age_edges(self):
        for edge in self.edges:
            edge.increment()

    def calc_error(self, vec):
        self.error += self.distance_from(vec)**2

    def creep_towards(self, vec, eps):
        dw = [eps * (s - v) for s, v in zip(vec, self.w)]
        self.w = [x+y for x, y in zip(dw, self.w)]

    def __repr__(self):
        return str(self.w)



class Edge:
    def __init__(self, a, b):
        self.age = 0
        self.nodes = {a, b}

    def increment(self):
        self.age += 1

    def __repr__(self):
        return "(%s --- %s)" % list(self.nodes)

ng = NeuralGas()
plt.ion()
for i in range(10):
    for i, point in enumerate(points):
        ng.train_signal(point)
        if i % 1000 == 0:
            ng.show()




'''
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W*x+b
loss = tf.reduce_sum(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b:%s loss:%s"%(curr_W, curr_b, curr_loss))

'''