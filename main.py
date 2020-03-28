import matplotlib.pyplot
import numpy
import scipy.special as sp
class neuralNetwork:
    def __init__(self, inputNode, hiddenNode, outputNode, learningRate):
        print("Initializare neural network")
        self.inputNode = inputNode
        self.hiddenNode = hiddenNode
        self.outputNode = outputNode
        self.learningRate = learningRate

        self.win = numpy.random.normal(0.0, pow(self.hiddenNode, -0.5), (self.hiddenNode, self.inputNode))
        self.who = numpy.random.normal(0.0, pow(self.outputNode, -0.5), (self.outputNode, self.hiddenNode))

        self.activation_function = lambda x: sp.expit(x)


    def train(self, inputs_list, targets_list):
        # print("Train neural nerwork")

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.win, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        error_outputs = (targets - final_outputs)
        error_hidden = numpy.dot(self.who.T, error_outputs)

        self.who += self.learningRate * numpy.dot((error_outputs * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.win += self.learningRate * numpy.dot((error_hidden * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, input_list):
        # print("Query")
        inputs = numpy.array(input_list, ndmin=2).T
        # print("2D input: ", inputs)
        # print("self win : ", self.win)

        hidden_inputs = numpy.dot(self.win, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def printW(self):
        print(self.win)
        print(self.who)

inputnode = 784
hiddennode = 200
outputnode = 10

learningRate = 0.1
repeat_training = 5

retea = neuralNetwork(inputnode, hiddennode, outputnode, learningRate)

data_file = open('mnist_dataset/mnist_train.csv', 'r')
training_data_list = data_file.readlines()
data_file.close()


for e in range(repeat_training):
    for record in training_data_list:
        all_values = record.split(',')
        tinputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        ttargets = numpy.zeros(outputnode) + 0.01
        ttargets[int(all_values[0])] = 0.99

        retea.train(tinputs, ttargets)

test_data_file = open('mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


score_board = []
percent = []
incorect = []

for test in test_data_list:
    results = {}
    all_test_value = test.split(',')
    test_inputs = (numpy.asfarray(all_test_value[1:]) / 255 * 0.99) + 0.01
    list_result = retea.query(test_inputs)
    # print(list_result)
    hi = max(list_result)[0]
    # print(hi)

    results['correct'] = int(all_test_value[0])
    results['guess'] = numpy.argmax(list_result)
    results['percent'] = hi * 100

    if results['correct'] != results['guess']:
        percent.append(0)
    else:
        percent.append(1)

    score_board.append(results)

# for result in score_board:
#     print(result)
#
# print('\n')
# print('#'*100)
# print('\n')
print(percent)

suma = 0
for pr in percent:
    suma += pr

print(f'Suma = {suma}')
print(f'Performanta = {(suma/len(percent)) * 100}')

# print(all_values[0])
# print(retea.query(qinputs))


# image_array = numpy.asfarray( all_values [1:]).reshape((28,28))
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# matplotlib.pyplot.show()

print("")
# retea.printW();