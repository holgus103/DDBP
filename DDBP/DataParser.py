import re
import numpy

def Parse(input):
    vals = input.split(" ");    
    # TODO: Parse desired output
    return numpy.concatenate((processPlayer(vals[0]), processPlayer(vals[1]), processPlayer(vals[2]), processPlayer(vals[3])));

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
    return numpy.concatenate(tuple(p));
