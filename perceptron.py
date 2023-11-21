#-------------------------------------------------------------------------
# AUTHOR: Rebecca Glatts
# FILENAME: perceptron.py
# SPECIFICATION: Tests the different hyperparameters of perceptrons and mlpclassifer models by iterating
# through them
# FOR: CS 4210- Assignment #4
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]
models = ["Perceptron", "MLPclassifier"]
df = pd.read_csv('C:\\Users\\Rebecca\\Downloads\\optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('C:\\Users\\Rebecca\\Downloads\\optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0
highest_mlpclassifier_accuracy = 0

for learning_rate in n: #iterates over n

    for shuffle_value in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for model in models: #iterates over the algorithms
            
            #Create a Neural Network classifier
            if model == "Perceptron":
               clf = Perceptron(eta0 = learning_rate, shuffle = shuffle_value, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init = learning_rate, hidden_layer_sizes = 1,
                                    shuffle = shuffle_value, max_iter=1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            correct = 0
            total = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    correct += 1
                total += 1


            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            accuracy = correct/total
            if Perceptron and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy}, Parameters: learning rate={learning_rate}, shuffle={shuffle_value}")
            if MLPClassifier and accuracy > highest_mlpclassifier_accuracy:
                highest_mlpclassifier_accuracy = accuracy
                print(f"Highest MLPClassifier accuracy so far: {accuracy}, Parameters: learning rate={learning_rate}, shuffle={shuffle_value}")











