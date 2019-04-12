# Testing the NeuralNetwork Class

import NeuralNetwork
import matplotlib.pyplot
import numpy

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = NeuralNetwork.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
with open("test_data/mnist_train.csv", 'r') as training_data_file:
    training_data_list = training_data_file.readlines()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

# got through all records in the training data set
for e in range(epochs):
    for record in training_data_list:
        # split tje record by the ',' commas
        all_values = record.split(',')

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # create the target output values (all 0.01, except the desired label wich is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# test the neural network

#scorecard for how well the network performs, initially empty
scorecard = []

with open("test_data/mnist_test.csv", 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

# got through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')

    # correct answer is the first value
    correct_label = int(all_values[0])
    print(correct_label, "correct_label   ", end='')

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # query the network
    outputs = n.query(inputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "networks answer")

    # append correct or incorrect to list
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)



