import PythonExtended.Graphing as graph
import PythonExtended.Math as m
import warnings
import MachineLearning.GeneticEvolution as ge

class netclassification:
    def __init__(self, x, y, z, xdif, ydif, zdif):
        self.x = x
        self.y = y
        self.z = z
        self.xdif = xdif
        self.ydif = ydif
        self.zdif = zdif

    def getclose(self,x,y):
        for i in range(0, len(self.x)):
            if (abs(self.x[i] - x) < self.xdif and
                abs(self.y[i] - y) < self.ydif):
                return self.z[i]
        warnings.warn("NO POINT FOUND FOR", x, y)

class graphpoint:
    def __init__(self, x, y, z, color="blue", rawjson = None):
        self.x = x
        self.y = y
        self.z = z

        self.color = color
        self.posicount = z
        self.count = 1
        if rawjson is None:
            rawjson = {}
        self.rawjson = [rawjson]

#endregion
#region data management functions
#order of output colors
colors = [[(255,0,0), 'red'],
          [(0,128,0), 'green'],
          [(255,255,0), 'yellow'],
          [(0,0,0), 'black']]

def clump(points, xdif, ydif, zdif, percent=False):
    outpoints = []
    for pointa in points:
        found = False
        for pointb in outpoints:
            if (abs(pointa.x - pointb.x) < xdif and
                abs(pointa.y - pointb.y) < ydif and
                (abs(pointa.z - pointb.z) < zdif or percent)):
                found = True
                pointb.count += 1
                pointb.posicount += pointa.z
                pointb.rawjson.append(pointa.rawjson[0])
        if not found:
            outpoints.append(graphpoint(pointa.x, pointa.y, pointa.z, pointa.color, rawjson=pointa.rawjson[0]))
    return outpoints

def colorize(points, colormode, percents, zmax, zmin, classification=None, net=None, testmode="SimplePosi"):
    for point in points:
        if percents:
            point.z = (point.posicount / point.count) * 100
        if colormode == "none":
            point.color = "blue"
        elif colormode == "data-file":
            if percents:
                point.color = (1 - m.scale(0, point.z, 100, 0, 1),
                               0.5 - m.scale(0, point.z, 100, 0, 0.5),
                               m.scale(0, point.z, 100, 0, 1))
            else:
                if percents:
                    point.color = (1 - m.scale(0, point.z, 100, 0, 1),
                                   0.5 - m.scale(0, point.z, 100, 0, 0.5),
                                   m.scale(0, point.z, 100, 0, 1))
                else:
                    if m.unitscale(zmin, point.z, zmax) > 0:
                        point.color = "blue"
                    else:
                        point.color = "orange"
        elif colormode == "score":
            netscore = classification.getclose(point.x, point.y)
            cutoff = (zmax - zmin)*0.5 + zmin
            if ((netscore > cutoff and point.z > cutoff) or
                (netscore <= cutoff and point.z <= cutoff)):
                point.color = "blue"
            else:
                point.color = "orange"
        elif colormode == "net-score":
            correct = ge.Test_Obj(net, point.rawjson, testmode) / len(point.rawjson)
            point.color = (1 - correct, (1 - correct)*0.5, correct)
        elif colormode == "var-error":
            netscore = classification.getclose(point.x, point.y)
            cutoff = (zmax - zmin) * 0.5 + zmin
            correct = ge.Test_Obj(net, point.rawjson, testmode) / len(point.rawjson)
            if correct > 0.5 and not ((netscore > cutoff and point.z > cutoff) or
                                      (netscore <= cutoff and point.z <= cutoff)):
                point.color = "orange"
            else:
                point.color = "blue"
        else:
            warnings.warn("Color mode not found: " + str(colormode))

#region GraphingFuncs
def GraphNet(net, xaxis, yaxis, fig, title=None):
    zaxis = ' & '.join(net.outputs)
    xs = []
    ys = []
    zs = []
    outcolors = []
    count = 0
    for name in net.outputs:
        xvals = []
        yvals = []
        zvals = []
        outcolors.append(colors[count][1])
        count += 1
        x = -1
        while x <= 1:
            y = -1
            while y <= 1:
                xvals.append(x)
                yvals.append(y)

                net.reset()
                for inName in net.inputs:
                    if inName == xaxis:
                        net.setNode(inName, x)
                    elif inName == yaxis:
                        net.setNode(inName, y)
                    else:
                        net.setNode(inName, 0)
                net.process()
                zvals.append(net.getNode(name))
                y += 0.1
            x += 0.1
        xs.append(xvals)
        ys.append(yvals)
        zs.append(zvals)
    if title is None:
        title = xaxis + " by " + yaxis
    graph.multiGraph3D(xs, ys, zs, outcolors, xaxis, yaxis, zaxis, title, plt=fig)

def GraphNetOutputs(net, xaxis, yaxis, outputs, fig, title=None):
    zaxis = ' & '.join(net.outputs)
    xs = []
    ys = []
    zs = []
    outcolors = []
    count = 0
    for name in outputs:
        xvals = []
        yvals = []
        zvals = []
        outcolors.append(colors[count][1])
        count += 1
        x = -1
        while x <= 1:
            y = -1
            while y <= 1:
                xvals.append(x)
                yvals.append(y)

                net.reset()
                for inName in net.inputs:
                    if inName == xaxis:
                        net.setNode(inName, x)
                    elif inName == yaxis:
                        net.setNode(inName, y)
                    else:
                        net.setNode(inName, 0)
                net.process()
                zvals.append(net.getNode(name))
                y += 0.1
            x += 0.1
        xs.append(xvals)
        ys.append(yvals)
        zs.append(zvals)
    if title is None:
        title = xaxis + " by " + yaxis
    graph.multiGraph3D(xs, ys, zs, outcolors, xaxis, yaxis, zaxis, title, plt=fig)


def GraphNetData(net, data, xaxis, yaxis, zaxis, defvalue, useclump, usepercents, colormode, fig, title=None):
    #region handle net
    netxvals = []
    netyvals = []
    netzvals = []

    xmin = data["inputs"][xaxis]["min"]
    xmax = data["inputs"][xaxis]["max"]
    ymin = data["inputs"][yaxis]["min"]
    ymax = data["inputs"][yaxis]["max"]
    zmin = data["outputs"][zaxis]["min"]
    zmax = data["outputs"][zaxis]["max"]

    x = -1
    while x <= 1:
        y = -1
        while y <= 1:
            netxvals.append(m.scale(-1, x, 1, xmin, xmax))
            netyvals.append(m.scale(-1, y, 1, ymin, ymax))
            net.reset()
            for inName in net.inputs:
                if inName == xaxis:
                    net.setNode(inName, x)
                elif inName == yaxis:
                    net.setNode(inName, y)
                else:
                    net.setNode(inName, defvalue)
            net.process()
            if usepercents:
                netzvals.append(m.scale(-1, net.getNode(zaxis), 1, 0, 100))
            else:
                netzvals.append(m.scale(-1, net.getNode(zaxis), 1, zmin, zmax))

            y += 0.1
        x += 0.1
    #endregion
    #region handle data
    datapoints = []
    for item in data["data"]:
        datapoints.append(graphpoint(item[xaxis], item[yaxis], item[zaxis], rawjson=item))

    if useclump or usepercents:
        datapoints = clump(datapoints, (xmax-xmin)/20, (ymax-ymin)/20, (zmax-zmin)/20, usepercents)

    classification = netclassification(netxvals, netyvals, netzvals, (xmax-xmin)/15, (ymax-ymin)/15, (zmax-zmin)/15)
    colorize(datapoints, colormode, usepercents, zmax, zmin, net=net, classification=classification)

    dataxvals = []
    datayvals = []
    datazvals = []
    datacolorvals = []
    datasizevals = []
    for point in datapoints:
        dataxvals.append(point.x)
        datayvals.append(point.y)
        datazvals.append(point.z)
        datacolorvals.append(point.color)
        datasizevals.append(point.count)

    size = None
    if useclump or usepercents:
        size = datasizevals

    if usepercents:
        zaxis += "%"

    #endregion
    if title is None:
        title = xaxis + " by " + yaxis
    graph.Graph3D(netxvals,netyvals,netzvals,"red", xaxis, yaxis, zaxis, title, plt=fig)
    graph.Graph3D(dataxvals, datayvals, datazvals, datacolorvals, xaxis, yaxis, zaxis, title, plt=fig, s=size)