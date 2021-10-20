try:
    import pygame
except ImportError:
    raise Exception("Pygame is required to use net renderer.")
from time import sleep


def stdSettings(display):  # get default settings for nueral net display
    return {"func": display.showNet, "score-text": 'auto', "settings": {"minCut": 0.1, "time": 1, "vdis": 20}}


def stop():
    pygame.quit()


def text_objects(text, font):
    textSurface = font.render(text, True, (255, 0, 0))
    return textSurface, textSurface.get_rect()


class screen:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Neural Net - Genetic Algorithm')
        self.bestnet = None
        self.bestscore = 0
        self.truescore = 0

    def bestNet(self, BestNet, BestScore, TrueScore):
        self.bestnet = BestNet
        self.bestscore = round(BestScore, 3)
        self.truescore = round(TrueScore, 3)

    def message_display(self, text, pos):
        largeText = pygame.font.Font('freesansbold.ttf', 15)
        TextSurf, TextRect = text_objects(str(text), largeText)
        TextRect.center = pos
        self.display.blit(TextSurf, TextRect)

    def showNet(self, net, minCut, time, score, vdis=50, hdis=150):
        counter = 0
        while counter != time:
            counter += 1
            self.display.fill((173, 216, 230))

            y = 50
            x = 100
            nodeCords = {}

            for name in net.inputs:
                inputNode = net.inputs[name]
                nodeCords[inputNode] = (x, y)
                self.message_display(name, (x - 50, y))
                y += vdis

            maxy = y

            for row in net.midnodes:
                x += hdis
                y = 50
                for midnode in row:
                    nodeCords[midnode] = (x, y)
                    y += round(vdis * (len(net.inputs) / len(row)))

            maxy = max(maxy, y)
            x += hdis
            y = 50

            for name in net.outputs:
                outputNode = net.outputs[name]
                nodeCords[outputNode] = (x, y)
                self.message_display(name, (x + 50, y))
                y += vdis

            maxy = max(maxy, y)
            self.message_display("Score: " + str(round(score, 3)) + "%", (100, maxy + 25))

            y = maxy + 100
            x = 100

            if self.bestnet is not None:

                for name in self.bestnet.inputs:
                    inputNode = self.bestnet.inputs[name]
                    nodeCords[inputNode] = (x, y)
                    self.message_display(name, (x - 50, y))
                    y += vdis

                maxy2 = y

                for row in self.bestnet.midnodes:
                    x += hdis
                    y = maxy + 100
                    for midnode in row:
                        nodeCords[midnode] = (x, y)
                        y += round(vdis * (len(self.bestnet.inputs) / len(row)))

                maxy2 = max(maxy2, y)

                x += hdis
                y = maxy + 100
                for name in self.bestnet.outputs:
                    outputNode = self.bestnet.outputs[name]
                    nodeCords[outputNode] = (x, y)
                    self.message_display(name, (x + 50, y))
                    y += vdis

                maxy2 = max(maxy2, y)

                self.message_display("Score: " + str(self.bestscore) + "%  True Score: " + str(self.truescore) + "%",
                                     (200, maxy2 + 25))

            name = list(net.outputs.keys())[0]
            outNodeType = type(net.outputs[name])

            for startNode, startLoc in nodeCords.items():
                if type(startNode) != outNodeType:
                    for endNode, value in startNode.connections.items():
                        endLoc = nodeCords[endNode]

                        if abs(value) > minCut:
                            if value > 0:
                                color = (0, 0, 0)
                            else:
                                color = (255, 255, 255)

                            pygame.draw.line(self.display, color, startLoc, endLoc, abs(round(value * 10)))

            for node, loc in nodeCords.items():
                pygame.draw.circle(self.display, (255, 0, 0), loc, 10)

            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    return
            sleep(0.03)
