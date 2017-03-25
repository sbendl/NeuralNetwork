import MultiLayerPerceptron

if __name__ == '__main__':
    trainingX = []
    trainingy = []
    test = []
    with open("optdigits_training.csv") as t:
        for line in t:
            l = line.split(",")
            lineInts = []
            for num in l:
                lineInts.append(int(num))
            trainingX.append(lineInts[:-1])
            y = [0 for i in range(0, 10)]
            y[lineInts[-1]] = 1
            trainingy.append(y)

    testX = []
    testy = []
    test = []
    with open("optdigits_training.csv") as t:
        for line in t:
            l = line.split(",")
            lineInts = []
            for num in l:
                lineInts.append(int(num))
            testX.append(lineInts[:-1])
            y = [0 for i in range(0, 10)]
            y[lineInts[-1]] = 1
            testy.append(y)

    nn = MultiLayerPerceptron.Net([64, 30, 10], 1.0)

    #Visualize.drawNN(nn)

    # for x in range(1, 100):
    #     nn.train(trainingX, trainingy)
    #     print(nn.errTot)
    err = 100
    i = 0
    errArr = [5 for x in range(0, 100)]
    while err > .55:
        err = nn.train(trainingX, trainingy)
        errArr[i % 100] = err
        nn.LR = sum(errArr) / len(errArr) * 2
        i = i + 1
        print(str(i) + ", " + str(nn.LR) + ": " + str(err))
        if i % 10 == 0:
            print("randomizing...")
            nn.randomizeWeights(percent=.01)
    #Visualize.drawNN(nn)

    score = 0
    for x, t in zip(testX, testy):
        out = nn.predict(x)
        max = 0
        maxindex = 0
        for i,o in enumerate(out):
            if o > max:
                max = o
                maxindex = i
        print(t.index(1), maxindex)
        if t.index(1) == maxindex:
            score += 1

    print(score)
    print(nn.predict(testX[0]))
