from MachineLearning import GeneticNets as gn
from MachineLearning import GeneticEvolution as ge
from MachineLearning import NetRender as nr

#config
fname = "scripts/files/titanic.json"
generations = 10
populationSize = 10
trainSize = 400
testSize = 100
evoRate = 4

#specific config - midnodes
midDepth = 10
midWidth = 2

screen = nr.screen()
rSettings = nr.stdSettings(screen)
rSettings["settings"]["vdis"] = 15

dataset, trainset, testset, metadata = ge.loadDataset(fname, testSize)

NetDB = gn.Random(metadata.inputs, metadata.outputs, 20, 2, 5) #creates the neural nets

bestNet = [None, 0]

import time
startTime = time.time()

for genCount in range(0, generations):
    evoRate = evoRate * 0.95
    NetDB, best, bestscore, truescore = ge.Test(NetDB, dataset, trainset, trainSize, testset, renderSettings = rSettings)
    if truescore > bestNet[1]:
        bestNet[0] = best
    NetDB = ge.evolve(NetDB, evoRate)
    screen.bestNet(best, bestscore, truescore)

gn.saveNets([bestNet[0]], "scripts/files/testnet", "Nets from file: " + fname, 2)
nr.stop()

print("Execution time: ", time.time() - startTime)
