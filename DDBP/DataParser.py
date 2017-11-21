import re
import numpy

def Parse(input):
    vals = re.findall("\w+", input);
    # TODO: Parse desired output
    return numpy.concatenate((processPlayer(vals[0:3]), processPlayer(vals[4:7]), processPlayer(vals[8:11]), processPlayer(vals[12:15])));

def processPlayer(cards):
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
    
    for t in cards:
        v = numpy.repeat(0, 13);
        for i in t:
             v[dict[i]] = 1;

        p.append(v);
    return numpy.concatenate(tuple(p));
