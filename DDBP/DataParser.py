import re
import numpy

def Parse(input):
    # mapping for chars in value string
    dict = {
        '0' : 0, 
        '1' : 1, 
        '2' : 2, 
        '3' : 3, 
        '4' : 4, 
        '5' : 5, 
        '6' : 6, 
        '7' : 7, 
        '8' : 8, 
        '9' : 9, 
        'A' : 10, 
        'B' : 11, 
        'C' : 12, 
        'D' : 13, 
    }
    t = input.split(":");
    data = [];
    outputs = [];
    vals = t[0].split(" ");    

    for c in t[1]:
        arr = numpy.repeat(0, 14);
        if c=="\n":
            continue
        arr[dict[c]] = 1;
        outputs.append(arr);

    players = numpy.concatenate((processPlayer(vals[0]), processPlayer(vals[1]), processPlayer(vals[2]), processPlayer(vals[3])));
    for suit in range(0, 5):
        for vista in range(0,4):
            suit_arr = numpy.repeat(0, 5);
            vista_arr = numpy.repeat(0, 4);
            suit_arr[suit] = 1;
            vista_arr[vista] = 1;
            data.append(numpy.concatenate((suit_arr, vista_arr, players)));
    return (data, outputs);

    

def processPlayer(cards):
    c = cards.split(".");
    p = [];

    # bit mapping for char values
    dict =     {
        'A' : 0,
        'K' : 1,
        'Q' : 2,
        'J' : 3,
        'T' : 4,
        '9' : 5,
        '8' : 6,
        '7' : 7,
        '6' : 8,
        '6' : 8,
        '5' : 9,
        '4' : 10,
        '3' : 11,
        '2' : 12,
    }
    
    for t in c:
        v = numpy.repeat(0, 13);
        for i in t:
             v[dict[i]] = 1;

        p.append(v);
    # suits: Spades, Hearts, Diamonds, Clubs
    # contracts: None, Spades, earts, Diamonds, Clubs
    # east, north, west, south
    return numpy.concatenate(tuple(p));

def ReadFile(path, linesCount):
    dataSet = []
    outputsSet = []
    lineNumber = 1
    with open(path, "r") as file:
        for line in file:
            if lineNumber > linesCount:
                break
            print(line)
            data, outputs = Parse(line);
            dataSet.append(data)
            outputsSet.append(outputs);
            lineNumber = lineNumber + 1
    return (dataSet, outputsSet)
