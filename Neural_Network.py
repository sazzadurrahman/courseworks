
# coding: utf-8

# ### Neural Network Model

# In[16]:


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import division


# ### Down load the data sets from these links 
# 
# #### A training set 
# http://www.pjreddie.com/media/files/mnist_train.csv
# 
# #### A testing set 
# http://www.pjreddie.com/media/files/mnist_test.csv
# 
# #####  Datasets and notebook are in the same folder
# 

# In[17]:


# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes1 ,hiddennodes2, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2      
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))      
        self.wh2o = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden1 layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden1 layer
        hidden1_outputs = self.activation_function(hidden1_inputs)
        
        # calculate signals into hidden2 layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from hidden2 layer
        hidden2_outputs = self.activation_function(hidden2_inputs)        
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden2_errors = numpy.dot(self.wh2o.T, output_errors) 
        
        
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)
        
        # update the weights for the links between the hidden and output layers
        self.wh2o += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))
        
        # update the weights for the links between the hidden and output layers
        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), numpy.transpose(hidden1_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)
  
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[18]:


# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()[0:1000]
training_data_file.close()


# In[19]:


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes1 = 100
hidden_nodes2 = 100
output_nodes = 10

# learning rate
learning_rate = 0.05

# create instance of neural network
neural_network_model = neuralNetwork(input_nodes, hidden_nodes1, hidden_nodes2 ,output_nodes, learning_rate)


# In[20]:


# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        neural_network_model.train(inputs, targets)
        pass
    pass


# In[21]:


# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()[0:1000]
test_data_file.close()


# In[22]:


# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    #print(all_values[0])
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = neural_network_model.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass


# In[23]:


# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

