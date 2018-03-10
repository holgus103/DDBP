

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
    (data, outputs, test_data, test_outputs) = dp.read_file("./DDBP/data/library", SET_SIZE, True, TRAIN_NO_TRUMP, TRAIN_TRUMP, TEST_NO_TRUMP, TEST_TRUMP, PARTITION);
    for i in range(0, 4):
        print("i: {0}".format(i))
        c = data[0][(i*52):((i+1)*52)]
        for j in range(1, 4):            
            start = (i - j + 4) % 4;
            end = (i - j + 4) % 4 + 1
            if(not np.array_equal(data[j][start*52:end*52], c)):
                print("FAILURE i: {0} j: {1}".format(i,j))
                return False;
    return True;

print(testPlayerRotations());
        
