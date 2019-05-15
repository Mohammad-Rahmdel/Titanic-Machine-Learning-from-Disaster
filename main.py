import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignoring warnings



def normalizer(x):
    m = x.shape[0]
    x = x - np.mean(x)
    x = (x * m) / (np.sum(x ** 2))
    return x

def ReLu(x):
    return np.maximum(0, x)

def sigmoid(z):
    x = tf.placeholder(tf.float32, name = 'x')
    results = tf.sigmoid(x)
    with tf.Session() as session:
        results = session.run(results, feed_dict={x: z})
    
    return results


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X of shape (input size, m)
    Y of shape (1, m)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def trainFunction(X_train, Y_train, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, n_l=1):
    # n_l = number of hidden layer

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                    
    
    m = Y_train.shape[1]
    print(m)
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    X = tf.placeholder(shape=[n_x, None] ,dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[n_y, None] ,dtype=tf.float32, name='Y')
    
    costs = [] 

    # parameters = initialize_parameters()
    tf.set_random_seed(1)

    # define n1 to n(n_l - 1) here
    n = {}
    n["0"] = n_x
    n["1"] = 32
    n["2"] = 64
    n["3"] = 128
    n["4"] = 64
    n["5"] = 32

    
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
    Z[str(n_l)] = tf.add(tf.matmul(parameters["W" + str(n_l)], A[str(n_l - 1)]) , parameters["b" + str(n_l)])

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z[str(n_l)],  labels = Y))


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate / 80).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
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
        correct_prediction = tf.equal(tf.argmax(Z[str(n_l)]), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters






import pandas as pd

def data_manipulation(dataset):
    data = pd.read_csv("./datasets/" + dataset + ".csv")
    # data = data[0:864]
    # print(data.shape)
    # print(data.head())
    m = data.shape[0] #891
    if dataset == "train":
        y = np.array([data.loc[:,'Survived']])

    x1 = np.array(data.loc[:,'Pclass'])
    x2 = np.array(data.loc[:,'Sex'])
    x3 = np.array(data.loc[:,'Age'])  # has NAN
    x4 = np.array(data.loc[:,'SibSp'])
    x5 = np.array(data.loc[:,'Parch'])
    x6 = np.array(data.loc[:,'Fare'])

    if dataset == "test":
        f = pd.isnull(x6)
        c = []
        for i in range(len(f)) :
            if f[i] == True:
                c.append(i)
        for i in c:
            x6[i] = 0

    x7 = np.array(data.loc[:,'Name'])
    x8 = np.array(data.loc[:,'Embarked'])  # has NAN


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


    x8[x8 == 'S'] = 1
    x8[x8 == 'C'] = 2
    x8[x8 == 'Q'] = 3

    f = pd.isnull(x8)
    c = []
    for i in range(len(f)) :
        if f[i] == True:
            c.append(i)
    for i in c:
        x8[i] = 0
    mean = np.sum(x8) / (m - len(c))
    x8[x8 == 0] = mean
    x8 = normalizer(x8)





    for i in range(len(x7)):
        if "Mr." in x7[i]:
            x7[i] = 1
        elif "Mrs." in x7[i]:
            x7[i] = 2
        elif "Miss." in x7[i]:
            x7[i] = 3
        elif "Master." in x7[i]:
            x7[i] = 4
        elif "Dr." in x7[i]:
            x7[i] = 5
        else :
            # print(x7[i])
            x7[i] = 6

    x7 = normalizer(x7)

    #x7 = Name{
    # Mr.
    # Mrs.
    # Miss.
    # Master.
    # Dr.
    # Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")    #1
    # Ms. Encarnacion                                         #1
    # Sir. Cosmo Edmund ("Mr Morgan")                         #1
    # Don.                                                    #1
    # Rev.                                                    #6
    # Mme.                                                    #1
    # Ms.                                                     #1
    # Major.                                                  #2
    # Gordon.                                                 #2
    # Mlle.                                                   #2
    # Col.                                                    #2
    # Capt.                                                   #1
    # Countess.                                               #1
    # Jonkheer.                                               #1
    # }


    # x = np.vstack((x1, x2, x3, x4, x5, x6))
    x = np.vstack((x1, x2, x3, x4, x5, x6, x7, x8))
    # print(x.shape)
    if dataset == "train" :
        return x, y
    elif dataset == "test" :
        return x
    else :
        print("WRONG INPUT!")

x, y = data_manipulation("train")


X_train = x
Y_train = y
m = Y_train.shape[1]
# print(X_train.shape)
# print(Y_train.shape)


# data = pd.read_csv("./datasets/test.csv")
# print(data.shape)


n_l = 6
# TODO TODO TODO TODO TODO TODO TODO RUN MODEL HERE TODO TODO TODO TODO TODO TODO TODO TODO TODO
parameters = trainFunction(X_train, Y_train, 0.001, 10000, m, True, n_l)



def train_prediction(parameters, X, Y_train, n_l = 1):
   
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
    # print(Y[0:5])
    # print(Y_train[0:5])

    g = np.subtract(Y, Y_train)
    g = abs(g)
    print(np.sum(g))
    print("Train Accuracy = " + str(1 - (np.sum(g)) / m))


# TODO TODO TODO TODO TODO TODO TODO PREDICT MODEL HERE TODO TODO TODO TODO TODO TODO TODO TODO TODO
train_prediction(parameters, x, Y_train, n_l)


def output(parameters):

    X_test = data_manipulation("test") 
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


    data = pd.read_csv("./datasets/test.csv")
    id = np.array(data.loc[:,'PassengerId'])
    output = np.stack((id, Y_hat))
    # output = np.transpose(output)
    output = output.astype(int)
    # print(output)
    df = pd.DataFrame({"PassengerId" : output[0], "Survived" : output[1]})
    df.to_csv("foo.csv", index=False)
    # np.savetxt("foo.csv", output, delimiter=",", header="PassengerId,B", comments="")
    

output(parameters)



""" Results :
8 features
m 0.05 1000
80%

6 features
79%

4layer
n_x = 8
83%

4layer
n_x = 8
learning rate = 0.008
90%

6layers
n_x = 8
learning rate = 0.008
train accuracy = 96%
test accuracy = 72%
"""
