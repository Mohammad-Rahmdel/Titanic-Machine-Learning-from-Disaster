import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignoring warnings




def normalizer(x):
    m = x.shape[0]
    x = x - np.mean(x)
    x = (x * m) / (np.sum(x ** 2))
    return x

import pandas as pd
data = pd.read_csv("./datasets/train.csv")
# print(data.head())
m = data.shape[0] #891
y = np.array(data.loc[:,'Survived'])
x1 = np.array(data.loc[:,'Pclass'])
x2 = np.array(data.loc[:,'Sex'])
x3 = np.array(data.loc[:,'Age'])
x4 = np.array(data.loc[:,'SibSp'])
x5 = np.array(data.loc[:,'Parch'])
x6 = np.array(data.loc[:,'Fare'])


x1 = normalizer(x1)
x4 = normalizer(x4)
x5 = normalizer(x5)
x6 = normalizer(x6)

x2[x2 == 'male'] = 1
x2[x2 == 'female'] = -1
x2 = normalizer(x2)

#solving unknown age problem
#replacing NANs with mean of ages
f = np.isnan(x3)
cnt = 0
for i in f:
    if i == True:
        cnt += 1
x3 = np.nan_to_num(x3)
mean = np.sum(x3) / (m - cnt)
x3[x3 == 0] = mean
x3 = normalizer(x3)


x = np.vstack((x1, x2, x3, x4, x5, x6))
print(x.shape)












# y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
# y = tf.constant(39, name='y')                    # Define y. Set to 39

# loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

# init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
#                                                  # the loss variable will be initialized and ready to be computed
# with tf.Session() as session:                    # Create a session and print the output
#     session.run(init)                            # Initializes the variables
#     print("loss = " + str(session.run(loss)))    # Prints the loss





# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a,b)
# with tf.Session() as session:
#     print("C = " + str(session.run(c)))
    
    
#     x = tf.placeholder(tf.int64, name = 'x')  #A placeholder is simply a variable that you will assign data to only later, 
#     #when running the session. We say that you feed data to these placeholders when running the session.
#     print(session.run(2 * x, feed_dict = {x: 3}))





# Exercise 1.1 - Linear function
def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    result = tf.add(tf.matmul(W, X), b)

    with tf.Session() as session:
        result = session.run(result)
    
    return result

# print(linear_function())



# Exercise 1.2 - Computing the sigmoid
def sigmoid(z):
    """
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    x = tf.placeholder(tf.float32, name = 'x')
    results = tf.sigmoid(x)
    with tf.Session() as session:
        results = session.run(results, feed_dict={x: z})
    
    return results

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(12) = " + str(sigmoid(12)))



# Exercise 1.3 - Computing the Cost
def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost 
    """
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    J = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = y)

    with tf.Session() as session:
        cost = session.run(J, feed_dict={z: logits, y: labels})

    
    return cost

# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# cost = cost(logits, np.array([0,0,1,1]))
# print ("cost = " + str(cost))




def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(shape=[n_x, None] ,dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[n_y, None] ,dtype=tf.float32, name='Y')
    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
    W1 : [25, 12288], b1 : [25, 1], W2 : [12, 25], b2 : [12, 1], W3 : [6, 12], b3 : [6, 1]

    Returns:
    parameters 
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    # using Xavier Initialization for weights
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X) , b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1) , b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2) , b3)

    return Z3


# tf.reset_default_graph()



# Exercise 2.4 Compute cost
def compute_cost(Z3, Y):
    """
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)   # shape (number of examples, num_classes)
    labels = tf.transpose(Y)    # shape (number of examples, num_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    # tf.reduce_mean basically does the summation over the examples

    return cost


# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))




# Exercise 2.6 - Building the model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train (input size = 12288, number of training examples = 1080)
    Y_train (output size = 6, number of training examples = 1080)
    X_test (12288, 120)
    Y_test (6, 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                    
    
    m = Y_train.shape[1]
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    X, Y = create_placeholders(n_x, n_y)
    costs = [] 

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer() # Initialize all the variables

    with tf.Session() as session:
        session.run(init)
        for i in range(num_epochs):
            seed = seed + 1
            epoch_cost = 0.
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            n_minibatches = np.floor(m / minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / n_minibatches

            
            if print_cost == True:
                if i % 100 == 0:
                    print ("Cost after epoch %i: %f" % (i, epoch_cost))
                if i % 5 == 0:
                    costs.append(epoch_cost)
            
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()



        parameters = session.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters
    

# parameters = model(X_train, Y_train, X_test, Y_test)
""" results : 
    Train Accuracy: 0.9990741
    Test Accuracy: 0.725
"""



