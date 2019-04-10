import numpy

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
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0,5), (self.hnodes, self.inodes))

        # who = W_hidden_output (output_nodes x hidden_nodes = Eine Matrix für die Verknüpfungen zwischen der
        # versteckten Schicht und der Ausgabeschicht
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0,5), (self.onodes, self.hnodes))

    # Trainieren - die Gewichte anhand von Trainingsbeispielen verfeinern, d.h. das Netz anlernen
    def train(self):
        pass

    # Abfragen -  eine Antwort von den Ausgabeknoten für eine gegebene Eingangsbelegung abgreifen
    def query(self):
        pass
