from MachineLearning import GeneticNets as gn
from MachineLearning import GeneticEvolution as ge
from MachineLearning import NetRender as nr

def test(fname = "venv/tests/files/titanic.json"):
    print("Running test script v2")
    #config
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

    NetDB = gn.Random(metadata.inputs, metadata.outputs, populationSize, midWidth, midDepth) #creates the neural nets

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

    gn.saveNets([bestNet[0]], "testnet", "Nets from file: " + fname, 2)
    nr.stop()

    print("Execution time: ", time.time() - startTime)

if __name__ == "__main__":
    test('files/titanic.json')
