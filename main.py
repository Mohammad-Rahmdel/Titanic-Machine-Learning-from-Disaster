import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework import ops
from utils import train_preprocessing, test_preprocessing, normalizer, random_mini_batches, ReLu, sigmoid
from Analysis import preprocessing
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignoring warnings

from featue_engineering import preprocessed_data


# from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)




def trainFunction(X_train, Y_train, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, n_l = 1, lambd=0.0, keep_prob=1):
    # n_l = number of hidden layer
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                    
    
    m = Y_train.shape[1]
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    X = tf.placeholder(shape=[n_x, None] ,dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[n_y, None] ,dtype=tf.float32, name='Y')
    
    costs = [] 

    tf.set_random_seed(1)

    # define n1 to n(n_l - 1) here
    n = {}
    n["0"] = n_x
    n["1"] = n_x * 4
    n["2"] = n_x * 8
    n["3"] = n_x * 8
    n["4"] = n_x * 8
    n["5"] = n_x * 4

    
    parameters = {}
    for i in range(1, n_l):
        parameters["b" + str(i)] = tf.get_variable("b" + str(i), [n[str(i)], 1], initializer = tf.zeros_initializer())
    parameters["b" + str(n_l)] = tf.get_variable("b" + str(n_l), [1, 1], initializer = tf.zeros_initializer())


    # using Xavier Initialization for weights
    for i in range(1, n_l):
        parameters["W" + str(i)] = tf.get_variable("W" + str(i), [n[str(i)], n[str(i-1)]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    parameters["W" + str(n_l)] = tf.get_variable("W" + str(n_l), [1, n[str(n_l - 1)]], initializer = tf.zeros_initializer())

    A = {}
    Z = {}
    A["0"] = X
    for i in range(1, n_l):
        Z[str(i)] = tf.add(tf.matmul(parameters["W" + str(i)], A[str(i - 1)]) , parameters["b" + str(i)])
        A[str(i)] = tf.nn.relu(Z[str(i)])   
        if keep_prob != 1 :
            rate = (keep_prob / 2) + i * 0.1
            A[str(i)] = tf.nn.dropout(A[str(i)], keep_prob=rate)

    Z[str(n_l)] = tf.add(tf.matmul(parameters["W" + str(n_l)], A[str(n_l - 1)]) , parameters["b" + str(n_l)])

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Z[str(n_l)],  labels = Y)
    for i in range(1, n_l + 1):
        cost += lambd * tf.nn.l2_loss(parameters["W" + str(i)])
    cost = tf.reduce_mean(cost)


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
                # epoch_cost += minibatch_cost / minibatch_cost.shape[1]

            if print_cost == True:
                if i % 100 == 0:
                    print ("Cost after epoch %i: %f" % (i, epoch_cost))
                if i % 5 == 0:
                    costs.append(epoch_cost)
      
            
        
        # plot the cost
        if print_cost == True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters = session.run(parameters)
        print ("Parameters have been trained!")

    return parameters



def train_prediction(parameters, X, Y_train, n_l = 1):
    m = Y_train.shape[1]
    A = {}
    Z = {}
    A["0"] = X
    for i in range(1, n_l):
        Z[str(i)] = np.dot(parameters["W" + str(i)], A[str(i - 1)]) + parameters["b" + str(i)]  
        A[str(i)] = ReLu(Z[str(i)])   
    Z[str(n_l)] = np.dot(parameters["W" + str(n_l)], A[str(n_l - 1)]) + parameters["b" + str(n_l)]  
    
    Y = sigmoid(Z[str(n_l)])[0]
    Y = np.around(Y)
    Y_train = Y_train[0]
    Y_train = Y_train.astype(int)

    g = np.subtract(Y, Y_train)
    g = abs(g)
    print("number of wrong predictions from 891 samples = " + str(np.sum(g)))
    print("Train Accuracy = " + str(1 - (np.sum(g)) / m))



# # USING pre_processing()
# train = train_preprocessing()
# X_train = np.transpose(train.iloc[:,1:].values)   # Convert pandas dataframe to numpy arraylist
# Y_train = np.transpose(train.iloc[:,0:1].values)
# m = Y_train.shape[1]
# # print(X_train.shape) # (13, 889)
# # print(Y_train.shape) # (1, 889)

# # USING processing()
# X_train, Y_train, X_test, _ = preprocessing()
# X_train = np.transpose(X_train.iloc[:,:].values)
# X_test = np.transpose(X_test.iloc[:,:].values)
# Y_train = Y_train.values
# Y_train = Y_train.reshape(Y_train.shape+(1,))
# Y_train = np.transpose(Y_train)
# m = Y_train.shape[1]

# # USING preprocessed_data()
X_train, Y_train, X_test, _ = preprocessed_data()

X_train = np.transpose(X_train.values)
X_test = np.transpose(X_test.values)
Y_train = Y_train.values
Y_train = Y_train.reshape(Y_train.shape+(1,))
Y_train = np.transpose(Y_train)
m = Y_train.shape[1]


n_l = 6
# TODO TODO TODO TODO TODO TODO TODO RUN MODEL HERE TODO TODO TODO TODO TODO TODO TODO TODO TODO
parameters = trainFunction(X_train, Y_train, 0.0006, 1000, m, True, n_l, 0, 1)
# TODO TODO TODO TODO TODO TODO TODO PREDICT MODEL HERE TODO TODO TODO TODO TODO TODO TODO TODO TODO
train_prediction(parameters, X_train, Y_train, n_l)




def output(X_test, parameters):

    W = {}
    b = {}
    size = int (len(parameters) / 2 ) + 1
    n_l = size - 1
    for i in range(1, size):
        W[str(i)] = parameters["W" + str(i)]
        b[str(i)] = parameters["b" + str(i)]

    A = {}
    Z = {}
    A["0"] = X_test
    for i in range(1, n_l):
        Z[str(i)] = np.dot(parameters["W" + str(i)], A[str(i - 1)]) + parameters["b" + str(i)]  
        A[str(i)] = ReLu(Z[str(i)])   
    Z[str(n_l)] = np.dot(parameters["W" + str(n_l)], A[str(n_l - 1)]) + parameters["b" + str(n_l)] 

    Y_hat = sigmoid(Z[str(n_l)])[0]
    Y_hat = np.around((Y_hat))

    # print(Y_hat.shape)
    # print(Y_hat)
    data = pd.read_csv("./datasets/answer.csv")
    Y = np.array(data.loc[:,'survived']) 
    Y = Y[891:]

    g = np.subtract(Y, Y_hat)
    g = abs(g)
    print("number of wrong predictions from 418 samples = " + str(np.sum(g)))
    m = 418
    print("Test Accuracy = " + str(1 - (np.sum(g)) / m))


    # data = pd.read_csv("./datasets/test.csv")
    # id = np.array(data.loc[:,'PassengerId'])
    # output = np.stack((id, Y_hat))
    # # output = np.transpose(output)
    # output = output.astype(int)
    # # print(output)
    # df = pd.DataFrame({"PassengerId" : output[0], "Survived" : output[1]})
    # df.to_csv("foo.csv", index=False)
    

# test = test_preprocessing()
# X_test = np.transpose(test.values)

output(X_test, parameters)




