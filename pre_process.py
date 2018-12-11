import numpy as np
from random import shuffle
from feed_forward import *
from pca import *
from random_features import *
from isomap import *
from nmf import *
import csv
import sys
import math

"Script fpr parsing the dataset into features and labels," \
"calling dimensionality reduction algorithms, and then training" \
"on a feed forward neural network with backprop"

def parse(file):
    training_data = []
    dataset = []
    with open(file, 'r', encoding='utf') as features:
        csv_input = csv.reader(features, delimiter=",")
        for row in csv_input:
            dataset.append(row)
    for instance in dataset:
        inst = []
        for i in instance:
            inst.append(float(i)) #convert to float
        training_data.append(inst)
    return training_data

def call_feed_forward(inputArray, expectedOutputArray, index):
    hidden_layer_amount = 1
    hidden_nodes_amount = 50
    output_nodes_amount = 1
    epocs = 10    # set low since convergence is fast
    activation_type = "s"  # set "s" for sig;

    #Initialize the Feed Forward Neural Net
    feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)

    feedforward.initialize()
    train_x = inputArray[0: int(len(inputArray) * 0.8)]
    test_x = inputArray[(int(len(inputArray) * 0.8)-1): -1]
    train_y = expectedOutputArray[0: int(len(inputArray) * 0.8)]
    test_y = expectedOutputArray[(int(len(inputArray) * 0.8) - 1): -1]

    for j in range(epocs):  #training stage
        sum_error = 0
        for i in range(len(train_x)):
            output = feedforward.feed_forward(i)
            sum_error += (train_y[i][0] - output)
            feedforward.backprop(i)
        print('>epoch=%d, average training error=%.3f' % (j, sum_error/len(train_x)))
        if abs(sum_error ** 2) < 0.1:
            break

    test_sum_error = 0
    num_correct = 0
    num_ones = 0
    predicted_outputs = []
    expected_outputs = []
    squared_errors = []
    for k in range(len(test_x)):  #testing stage
        if test_y[k][0] == 1:
            num_ones += 1
        test_output = feedforward.feed_forward(k)
        predicted_outputs.append(test_output[0])
        test_sum_error += (test_y[k][0] - test_output)
        expected_outputs.append(test_y[k][0])
    print('Average testing error=%.3f' % (test_sum_error/len(test_x)))
    print("Total testing error=%.3f" % (test_sum_error))
    print('Squared Error=%.3f' % (test_sum_error ** 2))
    squared_errors.append(test_sum_error ** 2)

    #the following code snippet predicts the first 500 frames of the video on a trained NN
    test_x = parse("./data/Subset_X.csv")
    test_y = parse("./data/Subset_Y.csv")
    test_x = decompose(test_x, 7)
    one_output = []
    for y in test_y:
        one_output.append([y[index]])
    print("Outputs on label:", item)
    predicted_outputs = []
    expected_outputs = []
    test_sum_error = 0
    for k in range(len(test_x)):
        test_output = feedforward.feed_forward(k)
        predicted_outputs.append(test_output[0])
        test_sum_error += (one_output[k][0] - test_output)
        expected_outputs.append(one_output[k][0])
        print('Average testing error=%.3f' % (test_sum_error / len(test_x)))
        print("Total testing error=%.3f" % (test_sum_error))
        print('Squared Error=%.3f' % (test_sum_error ** 2))
    print('Predicted outputs=', predicted_outputs)
    print('Expected outputs=', expected_outputs)
    return squared_errors


print("nmf ",7)
orig_stdout = sys.stdout
f = open('nmf_50_out.txt', 'w')
f = open('test.txt', 'w')
sys.stdout = f
features = parse('./data/X_Data_4.csv')
#reduced_x = decompose(features, 50)         #pca
#reduced_x = select_random(features, 50)   #random subset of values
#reduced_x = embed(features, 50)           #isomap
reduced_x = decompose(features, 7)        #nmf
labels = parse('./data/Y_Data_4_1.csv')
squared_error = []
squared_errors = []
counter = 0
for item in range(len(labels[0])):
    one_output = []
    for y in labels:
        one_output.append([y[item]])
    print("Outputs on label:", item)
    squared_error = call_feed_forward(reduced_x, one_output, counter)   #trains on a FFN one label at a time
    counter += 1
    squared_errors.append(squared_error)
average_squared_error = np.sum(squared_errors)/len(squared_errors)
print("Average Squared Error: ", average_squared_error)  #prints out the average squared error across all 12 rgb labels
print("Root Average Squared Error: ", math.sqrt(average_squared_error))
sys.stdout = orig_stdout
f.close()
