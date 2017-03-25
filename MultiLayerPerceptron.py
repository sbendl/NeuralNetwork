import math
import random

class Net:
    def __init__(self, nodesPerLayer, LR):
        self.nodesPerLayer = nodesPerLayer
        self.nodes = []

        for layer in nodesPerLayer:
            layerNodes = []
            for i in range(0, layer):
                layerNodes.append(Node(self))
            layerNodes.append(Node(self, biasNode=True))
            self.nodes.append(layerNodes)

        for node in self.nodes[0]:  # Make the first layer inputs
            node.isInput = True
            node.isHidden = False
        for node in self.nodes[-1]:  # Make the last layer outputs
            node.isOutput = True
            node.isHidden = False

        self.assignNames()

        for layer in range(0, len(nodesPerLayer) - 1):          # Make the connections
            for node in self.nodes[layer]:                      # For each node in current layer
                for child in self.nodes[layer + 1]:             # And each node in the following layer (child)
                    if not child.isBias:                        # If the child is not a bias node
                        connection = Connection(node, child)    # Create a connection between the two
                        connection.weight = random.uniform(0, 1)
                        node.connectionsFrom.append(connection)
                        child.connectionsTo.append(connection)

        self.errTot = 0
        self.LR = float(LR)

    def assignWeights(self, weights):
        for layer in range(0, len(self.nodes) - 1):
            lweight = weights[layer]                            # Sorry this is ugly!!!
            i = 0
            for node in self.nodes[layer]:
                for connection in node.connectionsFrom:
                    connection.weight = lweight[i]
                    i += 1

    def getWeights(self, layer):
        weights = []
        for node in self.nodes[layer]:
            nodeWeights = []
            for connection in node.connectionsFrom:
                nodeWeights.append(connection.weight)

        weights.append(nodeWeights)
        return weights

    def randomizeWeights(self, percent=.1):
        for layer in range(0, len(self.nodes) - 1):
            for node in self.nodes[layer]:
                for connection in node.connectionsFrom:
                    if random.uniform(0, 1) < percent:
                        connection.weight = random.uniform(0, 1)

    def forward(self, inputs):

        for node, input in zip(self.nodes[0], inputs + [1]):     # Run forward pass on all input nodes
            node.forward(input)
            #print(node.nodeName + ": " + str(node.out))

        for layer in self.nodes[1:-1]:                            # Run forward pass on all hidden layers
            for node, input in zip(layer, inputs + [1]):              # Run forward pass on all nodes in layer
                node.forward()
                #print(node.nodeName + ": " + str(node.out))

        for node in self.nodes[-1]:                               # Run forward pass on all output nodes
            node.forward()
            #print(node.nodeName + ": " + str(node.out))

        return [node.out for node in self.nodes[-1]][:-1]       # Return last layer
        #for node in self.nodes[-1]:
        #    print(node.nodeName + ": " + str(node.out))

    def backward(self, targets):
        self.errTot = sum([self.calcErr(target, node.out) for target, node in zip(targets, self.nodes[-1])])
        # self.errTot = self.calcErr(targets[0], self.o1) + self.calcErr(targets[1], self.o2)
        for target, node in zip(targets, self.nodes[-1]):
            node.backward(target)
        for layer in self.nodes[:-1][::-1]:
            for node in layer:
                node.backwardHidden()

    def trainSingle(self, X, y):
        out = self.forward(X)
        self.backward(y)
        return out
        #print(out)

    def train(self, XList, yList):
        totErr = 0
        i = 0
        for x, y in zip(XList, yList):
            self.trainSingle(x, y)
            totErr += self.errTot

        return totErr/len(yList)

    def predict(self, X):
        return self.forward(X)

    def calcErr(self, target, output):
        return .5 * (target - output) ** 2

    def assignNames(self):
        for layerIndex, layer in enumerate(self.nodes):
            for nodeIndex, node in enumerate(layer, 1):
                if node.isBias:
                    nodeIndex = "B"
                if node.isInput:
                    node.nodeName = "I%s" % nodeIndex
                elif node.isHidden:
                    node.nodeName = "H%s.%s" % (layerIndex, nodeIndex)
                else:
                    node.nodeName = "O%s" % nodeIndex

    def printConnections(self):
        self.assignNames()
        for layer in self.nodes:
            for node in layer:
                for connection in node.connectionsFrom:
                    print(connection.inputNode.nodeName + " -> " + connection.outputNode.nodeName + "  \t" + str(
                        connection.weight))

    def printToFile(self, filename):
        with open(filename, 'w') as out:
            self.assignNames()
            for layer in self.nodes:
                for node in layer:
                    for connection in node.connectionsFrom:
                        out.write(connection.inputNode.nodeName + " -> " + connection.outputNode.nodeName + "  \t" + str(
                            connection.weight))


class Connection:
    def __init__(self, parent, child, weight=0):
        self.weight = weight
        self.inputNode = parent
        self.outputNode = child


class Node:
    def __init__(self, net, inputNode=False, outputNode=False, biasNode=False):
        self.isInput = inputNode
        self.isOutput = outputNode
        self.isBias = biasNode
        if not self.isInput and not self.isOutput:
            self.isHidden = True
        else:
            self.isHidden = False
        self.net = net
        self.connectionsTo = []
        self.connectionsFrom = []
        self.out = 0
        self.inTot = 0
        self.errNode = 0
        self.nodeName = "Unassigned"

    def weightedSum(self, inputs):
        self.inTot = sum([weight * value for weight, value in zip(self.weights, inputs)])
        return self.inTot

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.e ** (-x))

    def forward(self, inputs=0):
        if self.isBias:
            self.out = 1
            self.inTot = 1
        elif self.isInput:
            self.out = inputs
            self.inTot = inputs
        else:

            self.inTot = sum([connection.inputNode.out * connection.weight for connection in self.connectionsTo])
            #print(self.nodeName + " (Debug): " + str(self.inTot))
            #totInput = self.weightedSum(inputs)
            self.out = self.sigmoid(self.inTot)
            #self.out = totInput

    def backward(self, target):
        self.errNode = -(target - self.out)
        for connection in self.connectionsTo:
            connection.weight -= self.net.LR * self.errNode * self.out * (1 - self.out) * connection.inputNode.out

    def backwardHidden(self):
        errNode = 0
        for connectionout in self.connectionsFrom:
            o = connectionout.inputNode.out
            errNode += o * o * (1 - o)

        for connectionin in self.connectionsTo:
            connectionin.weight -= self.net.LR * errNode * self.out * (1 - self.out) * (
                connectionin.inputNode.out * connectionin.weight)




if __name__ == '__main__':
    p = Net([2, 2, 2], .5)
    p.assignWeights([[.15, .25, .2, .3, .35, .35], [.4, .5, .45, .55, .6, .6]])
    #p.printConnections()
    print(p.trainSingle([.05, .1], [.01, .99]))
    print(p.errTot)
    print()
    #p.printConnections()
    print(p.trainSingle([.05, .1], [.01, .99]))
    print(p.errTot)

