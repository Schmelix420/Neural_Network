import numpy
import scipy.special

# neural network class defenition
class neuralNetwork:

    # Initialisierung- die Anzahl der Knoten für Eingabeschicht, verdeckte Schicht und Ausgabeschicht festlegen
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        # link weight matrices
        # wih = W_input_hidden (hidden_nodes x input_nodes = Eine Matrix mit den Gewichten für die Verknüpfungen
        # zwischen der Eingabeschicht und der versteckten Schicht.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        # who = W_hidden_output (output_nodes x hidden_nodes = Eine Matrix für die Verknüpfungen zwischen der
        # versteckten Schicht und der Ausgabeschicht
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # Trainieren - die Gewichte anhand von Trainingsbeispielen verfeinern, d.h. das Netz anlernen
    def train(self, inputs_list, target_list):
        # convert input list into 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate the signals to the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate the signals into the final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the ( target - actual )
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by wights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # delta_weights_jk = alpha * E_k * O_k(1 - O_k) * O_j.T
        # mit alpha = lernrate
        # O_j transponierte Matrix der Ausgänge von der vorherigen Schicht

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))



    # Abfragen -  eine Antwort von den Ausgabeknoten für eine gegebene Eingangsbelegung abgreifen
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # X_hidden = W_input_hidden * I
        # O_hidden = sigmoid( X_hidden )
        # X_output = W_hidden_output * O_hidden
        # O_output = sigmoid( X_output )

        #calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate the signals into the final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
