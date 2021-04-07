print("GeneticNets active...")
# import copy
import random
# from time import sleep
import json
import math


def inSet(strings):
    inputs = {}
    for i in strings:
        inputs[i] = inNode()
    return inputs


def outSet(strings, activation_func="sigmoid"):
    outputs = {}
    for i in strings:
        outputs[i] = outNode(activation_func=activation_func)
    return outputs


def scale(minv, v, maxv, minx=-1, maxx=1):
    # scale to 0 to 1
    return ((v - minv) / (maxv - minv)) * (maxx - minx) + minx


def descale(minv, v, maxv):
    # descale from 0 to 1
    return v * (maxv - minv) + minv


class node:  # the core of the nueral net, a node must have a value
    def __init__(self):
        self.val = 0  # default the node value to 0
        self.total = 0
        self.count = 0
        self.acivation_func = "sigmoid"

    def reset(self):
        self.total = 0
        self.count = 0

    def recieveValue(self, val, weight):  # used to take in a value
        self.total += val * weight
        self.count += abs(weight)

    def calc(self):
        if self.acivation_func == "sigmoid":
            self.val = ((1 / (1 + math.pow(math.e, (-1 * self.total))))  * 2) -1
        elif self.acivation_func == "relu":
            self.val = max(0, self.total)
        elif self.acivation_func == "old":
            self.val = self.total / self.count
        else:
            print("Unknown activation func: ", self.acivation_func)


class inNode(node):  # an input
    def __init__(self):
        super().__init__()
        self.connections = {}  # default to no connections

    def connect(self, cnode):
        self.connections[cnode] = 0  # default each connection strength to 0

    def disconnect(self, dnode):
        try:
            self.connections.pop(dnode)
        except Exception as e:
            print(str(node) + " not found")
            print("Error: ", str(e))

    def activate(self):
        for self.nextNode in self.connections:
            self.nextNode.recieveValue(self.val,
                                       self.connections[self.nextNode])  # push the value through the connection

    def evolve(self, evoRate):
        for self.i in self.connections:
            self.connections[self.i] = customRand(self.connections[self.i], evoRate)


class outNode(node):
    def __init__(self, activation_func="sigmoid"):
        super().__init__()  # these nodes just hold values, so they are kind of dumb
        self.acivation_func = activation_func


class midNode(node):  # pretty much the same as an input node
    def __init__(self, activation_func="sigmoid"):
        super().__init__()
        self.connections = {}
        self.acivation_func = activation_func

    def connect(self, cnode):
        self.connections[cnode] = 0

    def disconnect(self, dnode):
        try:
            self.connections.pop(dnode)
        except Exception as e:
            print(str(node) + " not found")
            print("Error:", str(e))

    def activate(self):
        for self.nextNode in self.connections:
            self.nextNode.recieveValue(self.val,
                                       self.connections[self.nextNode])  # push the value through the connection

    def evolve(self, evoRate):
        for self.i in self.connections:
            self.connections[self.i] = customRand(self.connections[self.i], evoRate)


class net:  # the network itself, contains many nodes
    def __init__(self, inputsRaw, outputsRaw, width, depth, bias=True, activation_func="sigmoid",
                 final_activation_func="sigmoid", neat=False, datafile = None, classifier_output=None):
        self.activation_func = activation_func
        self.final_activation_func = final_activation_func
        self.datafile = datafile

        self.inputs = inSet(inputsRaw)
        self.outputs = outSet(outputsRaw, activation_func=final_activation_func)

        self.expectedInputs = inputsRaw
        self.expectedOutputs = outputsRaw

        self.bias = inNode()

        self.usebias = bias
        self.neat = neat

        if classifier_output is None and len(self.outputs) == 1:
            self.classifier_output = list(self.outputs.keys())[0]
        else:
            self.classifier_output = classifier_output

        self.midnodes = []

        self.out = {}

        # neat stuff
        self.insertlayer = None
        self.newnode = None
        self.rawmidnodeslist = None

        # json stuff
        self.json = None
        self.nodeid = None
        self.nodeindex = None
        self.allmids = None
        self.rowcount = None
        self.connectionjson = None
        self.data = None
        self.connectionid = None
        self.connectionval = None

        for self.i in range(0, width):
            self.midnodestemp = []
            for self.j in range(0, depth):
                self.midnodestemp.append(midNode(activation_func=self.activation_func))

            self.midnodes.append(self.midnodestemp)

        self.width = width
        self.depth = depth

        if width == 0:
            for self.inName in self.inputs:
                self.inputNode = self.inputs[self.inName]
                for self.outName in self.outputs:
                    self.outputNode = self.outputs[self.outName]
                    self.inputNode.connect(self.outputNode)

        else:
            for self.inName in self.inputs:
                self.inputNode = self.inputs[self.inName]
                for self.midNode in self.midnodes[0]:
                    self.inputNode.connect(self.midNode)

            for self.i in range(1, len(self.midnodes)):
                for self.midnode in self.midnodes[self.i - 1]:
                    for self.midnode2 in self.midnodes[self.i]:
                        self.midnode.connect(self.midnode2)

            for self.midnode in self.midnodes[len(self.midnodes) - 1]:
                for self.outputName in self.outputs:
                    self.outputNode = self.outputs[self.outputName]
                    self.midnode.connect(self.outputNode)

        for self.midnodelist in self.midnodes:
            for self.midnode in self.midnodelist:
                self.bias.connect(self.midnode)

    def setNode(self, name, val):  # in
        self.inputs[name].val = val

    def getNode(self, name):  # out
        return self.outputs[name].val

    def receiveInput(self, inputs):
        for self.name in inputs:
            # print(scale(self.expectedInputs[self.name]["min"], inputs[self.name], self.expectedInputs[self.name]["max"]))

            self.setNode(self.name, scale(self.expectedInputs[self.name]["min"], inputs[self.name],
                                          self.expectedInputs[self.name]["max"], minx=-1, maxx=1))


    def getOutput(self):
        self.out = {}
        for self.name in self.expectedOutputs:
                self.out[self.name] = self.getNode(self.name)
        return self.out

    def scale(self, name, val):
        return scale(self.expectedOutputs[name]["min"], val, self.expectedOutputs[name]["max"])

    def process(self):  # eval
        if self.usebias:
            self.bias.activate()

        for self.inName in self.inputs:  # activate net
            self.inputNode = self.inputs[self.inName]
            self.inputNode.activate()

        for self.midnodelist in self.midnodes:
            for self.midnode in self.midnodelist:
                self.midnode.calc()
                self.midnode.activate()

        for self.outName in self.outputs:
            self.outputNode = self.outputs[self.outName]
            self.outputNode.calc()

    def reset(self):
        self.bias.val = -1

        for self.name in self.inputs:  # reset all inputs
            self.inputNode = self.inputs[self.name]
            self.inputNode.reset()

        for self.name in self.outputs:  # reset all outputs
            self.outputNode = self.outputs[self.name]
            self.outputNode.reset()

        for self.midnodelist in self.midnodes:
            for self.midnode in self.midnodelist:
                self.midnode.reset()

    def evolve(self, evoRate):
        # region NEAT
        if self.neat:  # Nuero Evolution of Augmented Topolgies
            if random.randint(0, 3) == 0:
                if random.randint(0, 2) != 0:  # mostly add nodes
                    self.newnode = midNode()
                    length = len(self.midnodes)
                    if random.randint(0,1): #don't add stuff to the end that much
                        length = max(0,length-1)
                    self.insertlayer = random.randint(0, length)  # pick one of the lists, or the end

                    if self.insertlayer != len(self.midnodes) and random.randint(0,
                                                                                 2) == 0:  # 67% chance to do an insert instead
                        self.midnodes.insert(self.insertlayer, [self.newnode])
                    elif self.insertlayer == len(self.midnodes):
                        self.midnodes.append([self.newnode])
                    else:
                        self.midnodes[self.insertlayer].append(self.newnode)

                    # ok, time to connect the node
                    if self.insertlayer == len(self.midnodes) - 1:
                        for self.inName in self.inputs:
                            self.inputNode = self.inputs[self.inName]
                            self.newnode.connect(self.inputNode)  # def to 0, we evolve later anyway

                    else:
                        for self.midnode in self.midnodes[self.insertlayer + 1]:
                            self.newnode.connect(self.midnode)

                else:  # delete midnodes
                    self.rawmidnodeslist = []
                    for self.midnodelist in self.midnodes:
                        for self.midnode in self.midnodelist:
                            self.rawmidnodeslist.append(self.midnode)

                    if len(self.rawmidnodeslist) > 0:
                        self.newnode = self.rawmidnodeslist.pop(random.randint(0, len(self.rawmidnodeslist) - 1))

                        for self.inName in self.inputs:
                            self.inputNode = self.inputs[self.inName]
                            if self.newnode in self.inputNode.connections.keys():
                                self.inputNode.disconnect(self.newnode)

                        for self.midnode in self.rawmidnodeslist:
                            if self.newnode in self.midnode.connections.keys():
                                self.midnode.disconnect(self.newnode)
        # endregion NEAT
        for self.inName in self.inputs:
            self.inputNode = self.inputs[self.inName]
            self.inputNode.evolve(evoRate)

        for self.midnodelist in self.midnodes:
            for self.midnode in self.midnodelist:
                self.midnode.evolve(evoRate)

        if self.usebias:
            self.bias.evolve(evoRate)
        return self

    def getJSON(self, name, ver):
        self.json = {"net_name": name,
                     "net_ver": str(ver),
                     "midwidth": self.width,
                     "nodes": [],
                     "expectedInputs": self.expectedInputs,
                     "expectedOutputs": self.expectedOutputs,
                     "act_func": self.activation_func,
                     "fin_act_func": self.final_activation_func,
                     "use-bias": self.usebias,
                     "use-neat": self.neat,
                     "data-file": self.datafile,
                     "classifier-output": self.classifier_output
                     }

        self.nodeid = 0
        self.nodeindex = {}
        for self.name in self.inputs:
            self.nodeindex[self.inputs[self.name]] = {"layer": "input", "name": self.name, "id": self.nodeid}
            self.nodeid += 1

        self.allmids = []
        self.rowcount = 0
        for self.row in self.midnodes:
            for self.midnode in self.row:
                self.allmids.append(self.midnode)
                self.nodeindex[self.midnode] = {"layer": self.rowcount, "id": self.nodeid}
                self.nodeid += 1
            self.rowcount += 1

        for self.name in self.outputs:
            self.nodeindex[self.outputs[self.name]] = {"layer": "output", "name": self.name, "id": self.nodeid}
            self.nodeid += 1

        self.nodeindex[self.bias] = {"layer": "bias", "id": self.nodeid}

        for self.node in self.nodeindex:
            if self.nodeindex[self.node]["layer"] != "output":

                self.connectionjson = {}
                for self.connection in self.node.connections:
                    self.connectionid = self.nodeindex[self.connection]["id"]
                    self.connectionval = self.node.connections[self.connection]
                    self.connectionjson[self.connectionid] = self.connectionval

                self.nodeindex[self.node]["connections"] = self.connectionjson

            self.json["nodes"].append(self.nodeindex[self.node])

        return self.json

    def save(self, fname, name, ver):
        print("Saving net: " + name + " with version: " + str(ver) + " to file: " + fname)
        self.data = self.getJSON(name, str(ver))

        with open((fname + '.json'), 'w') as fp:
            json.dump(self.data, fp, sort_keys=True, indent=4)


def customRand(cVal, evoRate):
    rate = 1
    for i in range(0, 3):
        if random.randint(0, 2) == 1:
            rate = rate * 1.1
        else:
            break

    rate = rate * evoRate

    newVal = cVal + ((random.random() * 2 - 1) * rate / 10)

    if newVal > 1:
        newVal = 1
    elif newVal < -1:
        newVal = -1

    return newVal


def Random(inputs, outputs, length, width, depth, bias=True, activation_func="relu", final_activation_func="relu", neat=False):
    print("Creating a set of " + str(length) + " random nets")
    netDB = []
    for i in range(0, length):
        newNet = net(inputs, outputs, width, depth, bias=bias, activation_func=activation_func,
                     final_activation_func=final_activation_func, neat=neat)
        newNet.evolve(5)
        netDB.append([newNet, 0])

    return netDB


def loadNet(fname):
    try:
        with open((fname + '.json'), 'r') as fp:
            data = json.load(fp)
        print("Found file with name: " + data["net_name"] + " and ver: " + data["net_ver"])
        return loadNetJSON(data)

    except Exception as e:
        print("Error loading file: ")
        print(e)
        return net({}, {}, 0, 0)


def loadNetJSON(data):
    nodeindex = {}
    newinputs = {}
    newoutputs = {}
    newmids = []

    for i in range(0, int(data["midwidth"])):
        newmids.append([])

    newnet = net({}, {}, 0, 0)
    newnet.activation_func = data["act_func"]
    newnet.final_activation_func = data["fin_act_func"]
    newnet.bias = data["use-bias"]
    newnet.neat = data["use-neat"]
    newnet.datafile = data["data-file"]
    newnet.classifier_output = data["classifier-output"]
    for nodedata in data["nodes"]:
        if nodedata["layer"] == "input":
            newnode = inNode()
            nodeindex[nodedata["id"]] = newnode
            newinputs[nodedata["name"]] = newnode

        elif nodedata["layer"] == "output":
            newnode = outNode(activation_func=newnet.final_activation_func)
            nodeindex[nodedata["id"]] = newnode
            newoutputs[nodedata["name"]] = newnode

        elif nodedata["layer"] == "bias":
            newnode = inNode()
            nodeindex[nodedata["id"]] = newnode
            newnet.bias = newnode

        else:
            newnode = midNode(activation_func=newnet.activation_func)
            nodeindex[nodedata["id"]] = newnode
            newmids[int(nodedata["layer"])].append(newnode)

    newnet.inputs = newinputs
    newnet.outputs = newoutputs
    newnet.midnodes = newmids
    newnet.expectedInputs = data["expectedInputs"]
    newnet.expectedOutputs = data["expectedOutputs"]

    for nodedata in data["nodes"]:
        if nodedata["layer"] != "output":
            Node = nodeindex[nodedata["id"]]
            for connectionnum in nodedata["connections"]:
                Node.connections[nodeindex[int(connectionnum)]] = nodedata["connections"][connectionnum]

    return newnet


def saveNets(netDB, fname, name, ver):
    print("Saving nets: " + name + " with version: " + str(ver) + " to file: " + fname)
    i = 0
    data = {"name": name,
            "ver": str(ver),
            "nets": []}
    for Net in netDB:
        data["nets"].append(Net.getJSON(i, str(ver)))
        i += 1

    with open((fname + '.json'), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


def loadNets(fname):
    # try:
    with open((fname + '.json'), 'r') as fp:
        data = json.load(fp)
    print("Found file with name: " + data["name"] + " and ver: " + data["ver"])

    netDB = []
    for Net in data["nets"]:
        netDB.append([loadNetJSON(Net), 0])

    # except Exception as e:
    # print("Error loading file: ")
    # print(e)
    # return {}

    return netDB