import random
import json
import math
import warnings


def inSet(strings):
    inputs = {}
    for i in strings:
        inputs[i] = InNode()
    return inputs


def outSet(strings, activation_func="sigmoid"):
    outputs = {}
    for i in strings:
        outputs[i] = OutNode(activation_func=activation_func)
    return outputs


def scale(minv, v, maxv, minx=-1, maxx=1):
    # scale to 0 to 1
    return ((v - minv) / (maxv - minv)) * (maxx - minx) + minx


def descale(minv, v, maxv):
    # descale from 0 to 1
    return v * (maxv - minv) + minv


class Node:  # the core of the nueral net, a node must have a value
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
            warnings.warn("Unknown activation func: " + str(self.acivation_func))

class InNode(Node):  # an input
    def __init__(self):
        super().__init__()
        self.connections = {}  # default to no connections

    def connect(self, cnode):
        self.connections[cnode] = 0  # default each connection strength to 0

    def disconnect(self, dnode):
        try:
            self.connections.pop(dnode)
        except Exception as e:
            warnings.warn(str(Node) + " not found. Exception: " + str(e))

    def activate(self):
        for nextNode in self.connections:
            nextNode.recieveValue(self.val, self.connections[nextNode])  # push the value through the connection

    def evolve(self, evoRate):
        for i in self.connections:
            self.connections[i] = customRand(self.connections[i], evoRate)


class OutNode(Node):
    def __init__(self, activation_func="sigmoid"):
        super().__init__()  # these nodes just hold values, so they are kind of dumb
        self.acivation_func = activation_func


class MidNode(Node):  # pretty much the same as an input node
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
            warnings.warn(str(Node) + " not found. Exception: " + str(e))

    def activate(self):
        for nextNode in self.connections:
            nextNode.recieveValue(self.val, self.connections[nextNode])  # push the value through the connection

    def evolve(self, evoRate):
        for i in self.connections:
            self.connections[i] = customRand(self.connections[i], evoRate)


class Net:  # the network itself, contains many nodes
    def __init__(self, inputsRaw, outputsRaw, width, depth, bias=True, activation_func="sigmoid",
                 final_activation_func="sigmoid", neat=False, datafile = None, classifier_output=None):
        self.activation_func = activation_func
        self.final_activation_func = final_activation_func
        self.datafile = datafile

        self.inputs = inSet(inputsRaw)
        self.outputs = outSet(outputsRaw, activation_func=final_activation_func)

        self.expectedInputs = inputsRaw
        self.expectedOutputs = outputsRaw

        self.bias = InNode()

        self.usebias = bias
        self.neat = neat

        if classifier_output is None and len(self.outputs) == 1:
            self.classifier_output = list(self.outputs.keys())[0]
        else:
            self.classifier_output = classifier_output

        self.midnodes = []

        self.out = {}

        for i in range(0, width):
            midnodestemp = []
            for j in range(0, depth):
                midnodestemp.append(MidNode(activation_func=self.activation_func))

            self.midnodes.append(midnodestemp)

        self.width = width
        self.depth = depth

        if width == 0:
            for inName in self.inputs:
                inputNode = self.inputs[inName]
                for outName in self.outputs:
                    outputNode = self.outputs[outName]
                    inputNode.connect(outputNode)

        else:
            for inName in self.inputs:
                inputNode = self.inputs[inName]
                for midNode in self.midnodes[0]:
                    inputNode.connect(midNode)

            for i in range(1, len(self.midnodes)):
                for midnode in self.midnodes[i - 1]:
                    for midnode2 in self.midnodes[i]:
                        midnode.connect(midnode2)

            for midnode in self.midnodes[len(self.midnodes) - 1]:
                for outputName in self.outputs:
                    outputNode = self.outputs[outputName]
                    midnode.connect(outputNode)

        for midnodelist in self.midnodes:
            for midnode in midnodelist:
                self.bias.connect(midnode)

    def setNode(self, name, val):  # in
        self.inputs[name].val = val

    def getNode(self, name):  # out
        return self.outputs[name].val

    def receiveInput(self, inputs):
        for name in inputs:
            self.setNode(name, scale(self.expectedInputs[name]["min"], inputs[name],
                                     self.expectedInputs[name]["max"], minx=-1, maxx=1))


    def getOutput(self):
        out = {}
        for name in self.expectedOutputs:
                out[name] = self.getNode(name)
        return out

    def scale(self, name, val):
        return scale(self.expectedOutputs[name]["min"], val, self.expectedOutputs[name]["max"])

    def process(self):  # eval
        if self.usebias:
            self.bias.activate()

        for inName in self.inputs:  # activate net
            inputNode = self.inputs[inName]
            inputNode.activate()

        for midnodelist in self.midnodes:
            for midnode in midnodelist:
                midnode.calc()
                midnode.activate()

        for outName in self.outputs:
            outputNode = self.outputs[outName]
            outputNode.calc()

    def reset(self):
        self.bias.val = -1

        for name in self.inputs:  # reset all inputs
            inputNode = self.inputs[name]
            inputNode.reset()

        for name in self.outputs:  # reset all outputs
            outputNode = self.outputs[name]
            outputNode.reset()

        for midnodelist in self.midnodes:
            for midnode in midnodelist:
                midnode.reset()

    def evolve(self, evoRate):
        # region NEAT
        if self.neat:  # Nuero Evolution of Augmented Topolgies
            if random.randint(0, 3) == 0:
                if random.randint(0, 2) != 0:  # mostly add nodes
                    newnode = MidNode()
                    length = len(self.midnodes)
                    if random.randint(0,1): #don't add stuff to the end that much
                        length = max(0,length-1)
                    insertlayer = random.randint(0, length)  # pick one of the lists, or the end

                    # 67% chance to do an insert instead
                    if insertlayer != len(self.midnodes) and random.randint(0, 2) == 0:
                        self.midnodes.insert(insertlayer, [newnode])
                    elif insertlayer == len(self.midnodes):
                        self.midnodes.append([newnode])
                    else:
                        self.midnodes[insertlayer].append(newnode)

                    # ok, time to connect the node
                    if insertlayer == len(self.midnodes) - 1:
                        for inName in self.inputs:
                            inputNode = self.inputs[inName]
                            newnode.connect(inputNode)  # def to 0, we evolve later anyway

                    else:
                        for midnode in self.midnodes[insertlayer + 1]:
                            newnode.connect(midnode)

                else:  # delete midnodes
                    rawmidnodeslist = []
                    for midnodelist in self.midnodes:
                        for midnode in midnodelist:
                            rawmidnodeslist.append(midnode)

                    if len(rawmidnodeslist) > 0:
                        newnode = rawmidnodeslist.pop(random.randint(0, len(rawmidnodeslist) - 1))

                        for inName in self.inputs:
                            inputNode = self.inputs[inName]
                            if newnode in inputNode.connections.keys():
                                inputNode.disconnect(newnode)

                        for midnode in rawmidnodeslist:
                            if newnode in midnode.connections.keys():
                                midnode.disconnect(newnode)
        # endregion NEAT
        for inName in self.inputs:
            inputNode = self.inputs[inName]
            inputNode.evolve(evoRate)

        for midnodelist in self.midnodes:
            for midnode in midnodelist:
                midnode.evolve(evoRate)

        if self.usebias:
            self.bias.evolve(evoRate)
        return self

    def getJSON(self, name, ver):
        outjson = {"net_name": name,
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
                   "classifier-output": self.classifier_output}

        nodeid = 0
        nodeindex = {}
        for name in self.inputs:
            nodeindex[self.inputs[name]] = {"layer": "input", "name": name, "id": nodeid}
            nodeid += 1

        allmids = []
        rowcount = 0
        for row in self.midnodes:
            for midnode in row:
                allmids.append(midnode)
                nodeindex[midnode] = {"layer": rowcount, "id": nodeid}
                nodeid += 1
            rowcount += 1

        for name in self.outputs:
            nodeindex[self.outputs[name]] = {"layer": "output", "name": name, "id": nodeid}
            nodeid += 1

        nodeindex[self.bias] = {"layer": "bias", "id": nodeid}

        for node in nodeindex:
            if nodeindex[node]["layer"] != "output":

                connectionjson = {}
                for connection in node.connections:
                    connectionid = nodeindex[connection]["id"]
                    connectionval = node.connections[connection]
                    connectionjson[connectionid] = connectionval

                nodeindex[node]["connections"] = connectionjson

            outjson["nodes"].append(nodeindex[node])

        return outjson

    def save(self, fname, name, ver, log=True):
        if log:
            print("Saving net: " + name + " with version: " + str(ver) + " to file: " + fname)
        data = self.getJSON(name, str(ver))

        with open((fname + '.json'), 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)


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


def Random(inputs, outputs, length, width, depth, bias=True, activation_func="relu", final_activation_func="relu", neat=False, log=True):
    if log:
        print("Creating a set of " + str(length) + " random nets")
    netDB = []
    for i in range(0, length):
        newNet = Net(inputs, outputs, width, depth, bias=bias, activation_func=activation_func,
                     final_activation_func=final_activation_func, neat=neat)
        newNet.evolve(5)
        netDB.append([newNet, 0])

    return netDB


def loadNet(fname, log=True):
    try:
        with open((fname + '.json'), 'r') as fp:
            data = json.load(fp)
        if log:
            print("Found file with name: " + data["net_name"] + " and ver: " + data["net_ver"])
        return loadNetJSON(data)

    except Exception as e:
        warnings.warn("Error loading file: " + str(e))
        return Net({}, {}, 0, 0)


def loadNetJSON(data):
    nodeindex = {}
    newinputs = {}
    newoutputs = {}
    newmids = []

    for i in range(0, int(data["midwidth"])):
        newmids.append([])

    newnet = Net({}, {}, 0, 0)
    newnet.activation_func = data["act_func"]
    newnet.final_activation_func = data["fin_act_func"]
    newnet.usebias = data["use-bias"]
    newnet.neat = data["use-neat"]
    newnet.datafile = data["data-file"]
    newnet.classifier_output = data["classifier-output"]
    for nodedata in data["nodes"]:
        if nodedata["layer"] == "input":
            newnode = InNode()
            nodeindex[nodedata["id"]] = newnode
            newinputs[nodedata["name"]] = newnode

        elif nodedata["layer"] == "output":
            newnode = OutNode(activation_func=newnet.final_activation_func)
            nodeindex[nodedata["id"]] = newnode
            newoutputs[nodedata["name"]] = newnode

        elif nodedata["layer"] == "bias":
            newnode = InNode()
            nodeindex[nodedata["id"]] = newnode
            newnet.bias = newnode

        else:
            newnode = MidNode(activation_func=newnet.activation_func)
            nodeindex[nodedata["id"]] = newnode
            newmids[int(nodedata["layer"])].append(newnode)

    newnet.inputs = newinputs
    newnet.outputs = newoutputs
    newnet.midnodes = newmids
    newnet.expectedInputs = data["expectedInputs"]
    newnet.expectedOutputs = data["expectedOutputs"]

    for nodedata in data["nodes"]:
        if nodedata["layer"] != "output":
            node = nodeindex[nodedata["id"]]
            for connectionnum in nodedata["connections"]:
                node.connections[nodeindex[int(connectionnum)]] = nodedata["connections"][connectionnum]

    return newnet


def saveNets(netDB, fname, name, ver, log=True):
    if log:
        print("Saving nets: " + name + " with version: " + str(ver) + " to file: " + fname)
    i = 0
    data = {"name": name,
            "ver": str(ver),
            "nets": []}
    for net in netDB:
        data["nets"].append(net.getJSON(i, str(ver)))
        i += 1

    with open((fname + '.json'), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


def loadNets(fname, log=False):
    with open((fname + '.json'), 'r') as fp:
        data = json.load(fp)

    if log:
        print("Found file with name: " + data["name"] + " and ver: " + data["ver"])

    netDB = []
    for net in data["nets"]:
        netDB.append([loadNetJSON(net), 0])
    return netDB