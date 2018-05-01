

import data_parser as dp;
import numpy as np;

TEST_TRUMP = False
TRAIN_TRUMP = False
TEST_NO_TRUMP = False
TRAIN_NO_TRUMP = True
BATCHES = 1
PARTITION = 1
SET_SIZE = 2

def testPlayerRotations():
    (data, outputs) = dp.read_file("./data/sol100000.txt", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);
    for i in range(0, 4):
        #print("i: {0}".format(i))
        c = data[6][0][(i*52):((i+1)*52)]      
        start = (i + 2) % 4;
        end = (i + 2) % 4 + 1
        if(not np.array_equal(data[6][1][start*52:end*52], c)):
            print("FAILURE i: {0}".format(i))
            return False;
    return True;

line = "T5.K4.652.A98542 K6.QJT976.QT7.Q6 432.A.AKJ93.JT73 AQJ987.8532.84.K:65658888888843433232"
    # 'A' : 0,
    # 'K' : 1,
    # 'Q' : 2,
    # 'J' : 3,
    # 'T' : 4,
    # '9' : 5,
    # '8' : 6,
    # '7' : 7,
    # '6' : 8,
    # '6' : 8,
    # '5' : 9,
    # '4' : 10,
    # '3' : 11,
    # '2' : 12,

nt_0 = [0,0,0,0,0,0,1,0,0,0,0,0,0,0];
nt_1 = [0,0,0,0,0,1,0,0,0,0,0,0,0,0];
nt_2 = [0,0,0,0,0,0,1,0,0,0,0,0,0,0];
nt_3 = [0,0,0,0,0,1,0,0,0,0,0,0,0,0];
# T5.K4.652.A98542
p1_spades =     [0,0,0,0,1,0,0,0,0,1,0,0,0]; # T5
p1_hearts =     [0,1,0,0,0,0,0,0,0,0,1,0,0]; # K4
p1_diamonds =   [0,0,0,0,0,0,0,0,1,1,0,0,1]; # 652
p1_clubs =      [1,0,0,0,0,1,1,0,0,1,1,0,1]; # A98542
# K6.QJT976.QT7.Q6
p2_spades =     [0,1,0,0,0,0,0,0,1,0,0,0,0]; # K6
p2_hearts =     [0,0,1,1,1,1,0,1,1,0,0,0,0]; # QJT976
p2_diamonds =   [0,0,1,0,1,0,0,1,0,0,0,0,0]; # QT7
p2_clubs =      [0,0,1,0,0,0,0,0,1,0,0,0,0]; # Q6
# 432.A.AKJ93.JT73
p3_spades =     [0,0,0,0,0,0,0,0,0,0,1,1,1]; # 432
p3_hearts =     [1,0,0,0,0,0,0,0,0,0,0,0,0]; # A
p3_diamonds =   [1,1,0,1,0,1,0,0,0,0,0,1,0]; # AKJ93
p3_clubs =      [0,0,0,1,1,0,0,1,0,0,0,1,0]; # JT73
# AQJ987.8532.84.K
p4_spades =     [1,0,1,1,0,1,1,1,0,0,0,0,0]; # AQJ987
p4_hearts =     [0,0,0,0,0,0,1,0,0,1,0,1,1]; # 8532 
p4_diamonds =   [0,0,0,0,0,0,1,0,0,0,1,0,0]; # 84
p4_clubs =      [0,1,0,0,0,0,0,0,0,0,0,0,0]; # K

def testParseHand():
    deal = line.split(':')[0];
    players = deal.split(' ');
    print(np.array_equal(dp.process_player(players[0]), p1_spades + p1_hearts + p1_diamonds + p1_clubs))
    print(np.array_equal(dp.process_player(players[1]), p2_spades + p2_hearts + p2_diamonds + p2_clubs))
    print(np.array_equal(dp.process_player(players[2]), p3_spades + p3_hearts + p3_diamonds + p3_clubs))
    print(np.array_equal(dp.process_player(players[3]), p4_spades + p4_hearts + p4_diamonds + p4_clubs)) 
    
def array_assert(arr1, arr2):
    l1 = len(arr1);
    l2 = len(arr2);
    if(l1 != l2):
        print("Arrays don't have the same length");
        return False;
    for i in range(0, l1):
        if(arr1[i] != arr2[i]):
            print("{0} element doesn't match".format(i));
            return False
    return True;

def parseNoTrump():
    print("parseNoTrump:");
    data = {}
    dp.parse(line, data, True, False);
    p1 = p1_spades + p1_hearts + p1_diamonds + p1_clubs
    p2 = p2_spades + p2_hearts + p2_diamonds + p2_clubs
    p3 = p3_spades + p3_hearts + p3_diamonds + p3_clubs
    p4 = p4_spades + p4_hearts + p4_diamonds + p4_clubs
    out = [nt_0, nt_2];
    inputs = [p1 + p2 + p3 + p4, p3 + p4 + p1 + p2] 
    for i in range(0,2):
        print(array_assert(data[6][i], inputs[i]))

def parseColors():
    print("printColors:")
    data = {}
    dp.parse(line, data, False, True);
    p1 = [\
            p1_spades + p1_hearts + p1_diamonds + p1_clubs, \
            p1_hearts + p1_spades + p1_diamonds + p1_clubs, \
            p1_diamonds + p1_hearts + p1_spades + p1_clubs, \
            p1_clubs + p1_hearts + p1_diamonds + p1_spades\
        ];
    p2 = [\
            p2_spades + p2_hearts + p2_diamonds + p2_clubs, \
            p2_hearts + p2_spades + p2_diamonds + p2_clubs, \
            p2_diamonds + p2_hearts + p2_spades + p2_clubs, \
            p2_clubs + p2_hearts + p2_diamonds + p2_spades\
        ];
    p3 = [\
            p3_spades + p3_hearts + p3_diamonds + p3_clubs, \
            p3_hearts + p3_spades + p3_diamonds + p3_clubs, \
            p3_diamonds + p3_hearts + p3_spades + p3_clubs, \
            p3_clubs + p3_hearts + p3_diamonds + p3_spades\
        ]; 
    p4 = [\
            p4_spades + p4_hearts + p4_diamonds + p4_clubs, \
            p4_hearts + p4_spades + p4_diamonds + p4_clubs, \
            p4_diamonds + p4_hearts + p4_spades + p4_clubs, \
            p4_clubs + p4_hearts + p4_diamonds + p4_spades\
        ];    
    hands = [\
    # spades 
                p1[0] + p2[0] + p3[0] + p4[0],\
                p3[0] + p4[0] + p1[0] + p2[0],\

    # hearts
                p1[1] + p2[1] + p3[1] + p4[1],\
                p3[1] + p4[1] + p1[1] + p2[1],\
    # diamonds 
                p1[2] + p2[2] + p3[2] + p4[2],\
                p3[2] + p4[2] + p1[2] + p2[2],\
    # clubs
                p1[3] + p2[3] + p3[3] + p4[3],\
                p3[3] + p4[3] + p1[3] + p2[3],\
    ]     
    indexes = [8,8,8,8,4,4,3,3]
    for i in range(0, len(data)):
        print(array_assert(hands[i], data[indexes[i]][i]));

print(testPlayerRotations());
testParseHand();
parseNoTrump();
parseColors();
