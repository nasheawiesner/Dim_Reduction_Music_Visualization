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

def call_feed_forward(inputArray, expectedOutputArray):
    hidden_layer_amount = 1
    hidden_nodes_amount = 50
    output_nodes_amount = 1
    epocs = 10
    activation_type = "s"  # set "s" for sig;

    feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)

    feedforward.initialize()
    train_x = inputArray[0: int(len(inputArray) * 0.8)]
    test_x = inputArray[(int(len(inputArray) * 0.8)-1): -1]
    train_y = expectedOutputArray[0: int(len(inputArray) * 0.8)]
    test_y = expectedOutputArray[(int(len(inputArray) * 0.8) - 1): -1]

    for j in range(epocs):
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
    for k in range(len(test_x)):
        if test_y[k][0] == 1:
            num_ones += 1
        test_output = feedforward.feed_forward(k)
        predicted_outputs.append(test_output[0])
        test_sum_error += (test_y[k][0] - test_output)
        expected_outputs.append(test_y[k][0])
        if test_output < 0.5:
            test_output = 0.0
        else:
            test_output = 1.0
        if test_output == test_y[k][0]:
            num_correct += 1
    #accuracy = num_correct/len(test_x)
    #print("Accuracy: ", (accuracy * 100))
    print('Average testing error=%.3f' % (test_sum_error/len(test_x)))
    print("Total testing error=%.3f" % (test_sum_error))
    print('Squared Error=%.3f' % (test_sum_error ** 2))
    squared_errors.append(test_sum_error ** 2)
    #print('Predicted outputs=', predicted_outputs)
    #print('Expected outputs=', expected_outputs)
    return squared_errors
print("nmf ",50)
orig_stdout = sys.stdout
f = open('nmf_50_out.txt', 'w')
#f = open('test.txt', 'w')
sys.stdout = f
features = parse('./data/X_Data_4.csv')
#reduced_x = decompose(features, 50)         #pca
#reduced_x = select_random(features, 50)   #random subset of values
#reduced_x = embed(features, 50)           #isomap
reduced_x = decompose(features, 50)        #nmf
labels = parse('./data/Y_Data_4_1.csv')
squared_error = []
squared_errors = []
for item in range(len(labels[0])):
    one_output = []
    for y in labels:
        one_output.append([y[item]])
    print("Outputs on label:", item)
    squared_error = call_feed_forward(reduced_x, one_output)
    squared_errors.append(squared_error)
average_squared_error = np.sum(squared_errors)/len(squared_errors)
print("Average Squared Error: ", average_squared_error)
print("Root Average Squared Error: ", math.sqrt(average_squared_error))
sys.stdout = orig_stdout
f.close()
