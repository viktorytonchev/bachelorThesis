# -*- coding: utf-8 -*-

"""

Created on Tue May 10 09:39:39 2022

 

@author: TonchevVY

"""
 

import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from numpy import std
from numpy import dstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

window_size = 50 # Time steps for the windows when splitting the data. 1 time step is 0.02 seconds or 20 miliseconds, since the sensor recorded data at 50Hz
overlap = 0.5 # Overlap of the windows


# Generate an array with the names of the pickle files
def file_names(numbers):
    files = []
    for x in numbers:
        files.append("annotations_0000" + str(x) + ".pickle")
    return files


# Open all the pickle files
def load_pickles(names):
    result = []
    for x in names:
        infile = open(x, 'rb')
        result.append(pickle.load(infile))
        infile.close()
    return result


#Generate arrays with the names of the csv files containing the data from the sensors
def sensor_names(numbers):
    acc_names = []
    gyro_names = []
    for x in numbers:
        acc_names.append("0000" + str(x) + "_Accelerometer_50Hz.csv")
        gyro_names.append("0000" + str(x) + "_Gyroscope_50Hz.csv")
    return acc_names, gyro_names


# Load the data from the csv files with the censor data
def load_sensor_files(acc_names, gyro_names):
    result_acc = []
    result_gyro = []
    for x in acc_names:
        result_acc.append(pd.read_csv(x))
    for x in gyro_names:
        result_gyro.append(pd.read_csv(x))
    return result_acc, result_gyro


# The names of some of the annotations are inconsistent, so fix the dictionary keys
def fix_keys(synced_dict):
    if "Callibration" in synced_dict.keys():
        synced_dict["Calibration"] = synced_dict.pop("Callibration")
    if "Random" in synced_dict.keys():
        synced_dict["Random_movement"] = synced_dict.pop("Random")
    if "Cucumber_(fork_-__stab)" in synced_dict.keys():
        synced_dict["Cucumber_(fork_stab)"] = synced_dict.pop("Cucumber_(fork_-__stab)")


# Round the annotations to 2 decimals. This is done since the sensor records data at 50Hz, so the data is in intervals of 0.02 seconds, which means 2 decimal precision is all we need.
def round_and_synchronize(dictionaries, cal_sensor):
    result = []
    counter = 0
    for x in dictionaries:
        if "default" in x.keys():
            x.pop("default")
        cal_ann = x["Calibration"][0][0]
        diff = round(cal_ann - cal_sensor[counter], 2)
        counter = counter + 1
        synced_dict = dict()
        for key in x:
            synced_dict[key] = x[key] - diff
        rounded_synced_dict = dict()
        for key in synced_dict:
            rounded_synced_dict[key] = np.round(synced_dict[key], 2)
        rounded_synced_dict["Leftover"][0][0] = 0
        result.append(rounded_synced_dict)
    return result


# For some sessions the gyrocsope or accelerometer records data for a few timesteps more, so they need to be truncated and made even
def even_out(acc, gyro):
    size_acc = acc.shape[0]
    size_gyro = gyro.shape[0]
    diff = abs(size_acc - size_gyro)
    if size_acc < size_gyro:
        while diff > 0:
            gyro.drop(gyro.index[-1], axis = 0, inplace = True)
            diff = diff - 1
    else:
        while diff > 0:
            acc.drop(acc.index[-1], axis = 0, inplace = True)
            diff = diff - 1


# Convert the annotation names to categories to be used for the algorithm
def annotation_to_category_just_eating(key):
    if key == "Leftover":
        return 0
    elif key == "Calibration":
        return 0
    elif key == "Yoghurt_(spoon)":
        return 1
    elif key == "Random_movement":
        return 0
    elif key == "Cereal_(spoon)":
        return 1
    elif key == "Cheek_scratch":
        return 0
    elif key == "Piece_of_bread_with_hummus_on_it_(hand)":
        return 1
    elif key == "Chin_scratch":
        return 0
    elif key == "Croissant_(hand)":
        return 1
    elif key == "Back_of_head_scratch":
        return 0
    elif key == "Grapes_(hand)":
        return 1
    elif key == "Corn_(fork_scoop)":
        return 1
    elif key == "Cucumber_(fork_stab)":
        return 1
    else:
        print("Unrecognized key: " + key)
        return 13


# Split the data into windows of 1s (or to however much it is set) and assign ground truth to them
def finalize(x, y, z, ground, steps):
    
    start = 0
    result_x = []
    result_y = []
    result_z = []
    result_ground = []
    counter = 0
    for a in range(steps):
        window_x = []
        window_y = []
        window_z = []
        flag = False
        for key in ground.keys():
            if not flag and not key == "Leftover":
                for values in ground[key]:
                    if start*0.02 >= values[0] and (start + window_size)*0.02 <= values[1]:
                        result_ground.append(annotation_to_category_just_eating(key))
                        flag = True
        if not flag and counter<5:
            result_ground.append(annotation_to_category_just_eating("Leftover"))
        if flag or (not flag and counter<5):
            for b in range(start, start + window_size):
                window_x.append(x[b])
                window_y.append(y[b])
                window_z.append(z[b])
            result_x.append(window_x)
            result_y.append(window_y)
            result_z.append(window_z)
            counter = counter + 1
        start = start + int(window_size*(1 - overlap))
    return result_x, result_y, result_z, result_ground


# The times at which calibration starts. These were all extracted manually by making plots of the data and looking for the 5 bumps made by the tapping of the sensor during the experiment
numbers = [0, 2, 3, 4, 5, 6, 8, 9]
files = file_names(numbers)
dictionaries = load_pickles(files)
acc_names, gyro_names = sensor_names(numbers)
data_acc, data_gyro = load_sensor_files(acc_names, gyro_names)


for x in range(len(numbers)):
    even_out(data_acc[x], data_gyro[x])


for x in dictionaries:
    fix_keys(x)


cal_sensor = [2.3, 3.76, 3.94, 4.12, 3.64, 5.04, 5.3, 4.5]
synced_dict = round_and_synchronize(dictionaries, cal_sensor)


data_acc_x = []
data_acc_y = []
data_acc_z = []


data_gyro_x = []
data_gyro_y = []
data_gyro_z = []


for x in range(len(data_acc)):
    data_acc_x.append(data_acc[x]["x-axis (g)"].to_numpy())
    data_acc_y.append(data_acc[x]["y-axis (g)"].to_numpy())
    data_acc_z.append(data_acc[x]["z-axis (g)"].to_numpy())

    
for x in range(len(data_gyro)):
    data_gyro_x.append(data_gyro[x]["x-axis (deg/s)"].to_numpy())
    data_gyro_y.append(data_gyro[x]["y-axis (deg/s)"].to_numpy())
    data_gyro_z.append(data_gyro[x]["z-axis (deg/s)"].to_numpy())


final_x_acc = []
final_y_acc = []
final_z_acc = []
final_x_gyro = []
final_y_gyro = []
final_z_gyro = []
final_ground = []
for x in range(len(synced_dict)):
    steps = math.floor(len(data_acc_x[x])/(window_size*(1 - overlap)) - 1)
    temp_x, temp_y, temp_z, temp_ground = finalize(data_acc_x[x], data_acc_y[x], data_acc_z[x], synced_dict[x], steps)
    final_x_acc.append(np.array(temp_x))
    final_y_acc.append(np.array(temp_y))
    final_z_acc.append(np.array(temp_z))
    steps = math.floor(len(data_gyro[x])/(window_size*(1 - overlap)) - 1)
    temp_x, temp_y, temp_z, temp_ground1 = finalize(data_gyro_x[x], data_gyro_y[x], data_gyro_z[x], synced_dict[x], steps)
    final_x_gyro.append(np.array(temp_x))
    final_y_gyro.append(np.array(temp_y))
    final_z_gyro.append(np.array(temp_z))
    final_ground.append(np.array(temp_ground))


# This is the function that splits the data into train and test sets.
def train_test_data_variations(train, test):
    
    
    final_train_x = []
    final_train_y = []
    final_train_z = []
    final_test_x = []
    final_test_y = []
    final_test_z = []
    final_train_x_gyro = []
    final_train_y_gyro = []
    final_train_z_gyro = []
    final_test_x_gyro = []
    final_test_y_gyro = []
    final_test_z_gyro = []
    final_train_ground = []
    final_test_ground = []
    
    
    # Make the final arrays with the training data
    for x in train:
        for y in final_x_acc[x]:
            final_train_x.append(y)
    
    for x in train:
        for y in final_y_acc[x]:
            final_train_y.append(y)
    
    for x in train:
        for y in final_z_acc[x]:
            final_train_z.append(y)
    
    for x in train:
        for y in final_x_gyro[x]:
            final_train_x_gyro.append(y)
    
    for x in train:
        for y in final_y_gyro[x]:
            final_train_y_gyro.append(y)
    
    for x in train:
        for y in final_z_gyro[x]:
            final_train_z_gyro.append(y)
    
    for x in train:
        for y in final_ground[x]:
            final_train_ground.append(y)
    
    
    # Make the final arrays with the testing data
    for y in final_x_acc[test]:
        final_test_x.append(y)
    
    for y in final_y_acc[test]:
        final_test_y.append(y)
    
    for y in final_z_acc[test]:
        final_test_z.append(y)
    
    for y in final_x_gyro[test]:
        final_test_x_gyro.append(y)
    
    for y in final_y_gyro[test]:
        final_test_y_gyro.append(y)
    
    for y in final_z_gyro[test]:
        final_test_z_gyro.append(y)
    
    for y in final_ground[test]:
        final_test_ground.append(y)
    
    
    # Convert to numpy
    final_train = list()
    final_train.append(np.array(final_train_x))
    final_train.append(np.array(final_train_y))
    final_train.append(np.array(final_train_z))
    final_train.append(np.array(final_train_x_gyro))
    final_train.append(np.array(final_train_y_gyro))
    final_train.append(np.array(final_train_z_gyro))
    final_train_ground = np.array(final_train_ground)

    # Convert to numpy
    final_test = list()
    final_test.append(np.array(final_test_x))
    final_test.append(np.array(final_test_y))
    final_test.append(np.array(final_test_z))
    final_test.append(np.array(final_test_x_gyro))
    final_test.append(np.array(final_test_y_gyro))
    final_test.append(np.array(final_test_z_gyro))
    final_test_ground = np.array(final_test_ground)
    
    final_train = dstack(final_train)
    final_test = dstack(final_test)
    
    return final_train, final_train_ground, final_test, final_test_ground


# Load the dataset, returns train and test X and y elements
def load_dataset(train, test, prefix=''):
	# load all train
	trainX, trainy = train_test_data_variations(train, test)[0], train_test_data_variations(train, test)[1]
	# load all test
	testX, testy = train_test_data_variations(train, test)[2], train_test_data_variations(train, test)[3]
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 500, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_ints = [y.argmax() for y in trainy]
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_ints), y=y_ints)
    class_weights_dict = dict(enumerate(class_weights))
    # fit model
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight=class_weights_dict)
    # predict probabilities for test set
    yhat_probs = model.predict(testX, verbose=0)
    # predict crisp classes for test set
    yhat_classes = yhat_probs.argmax(axis=1)
    
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    testy = np.argmax(testy, axis=1)
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes, average='weighted')
    print('F1 score: %f' % f1)
    
    # confusion matrix
    matrix = confusion_matrix(testy, yhat_classes)
    # print(matrix)
    
    return f1, matrix, accuracy


# summarize scores    
def summarise_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    return m, s
    

# Run an experiment
def run_experiment(train, test, repeats=1):
	# load data
	trainX, trainy, testX, testy = load_dataset(train, test, "C:/Users/vikic/OneDrive - University of Twente/Twente/Research Project/")
	# repeat experiment
	return evaluate(trainX, trainy, testX, testy)


# Test the model in x repetitions.
def test(train, test, repeats = 5):
    
    print("Running test over " + str(repeats) + " repetitions of the LSTM algorithm." )
    
    scores = list()
    accuracies = list()
    matrices = list()
    
    for a in range(repeats):
        
        scores_temp = list()
        accuracies_temp = list()
    
        for x in range(len(train)):
            f1, matrix, accuracy = run_experiment(train[x], test[x])
            scores_temp.append(f1)
            matrices.append(matrix)
            accuracies_temp.append(accuracy)
        
        m, s = summarise_results(scores_temp)
        scores.append(m)
        print('Average F1 over the different train/test distributions: %.3f (+/-%.3f)' % (m, s))
        m, s = summarise_results(accuracies_temp)
        accuracies.append(m)
        print('Average accuracy over the different train/test distributions: %.3f (+/-%.3f)' % (m, s))
        
    print("Average F1 score over " + str(repeats) + " repetitions:  %.3f (+/-%.3f)" % summarise_results(scores))
    print("Average accuracy over " + str(repeats) + " repetitions:  %.3f (+/-%.3f)" % summarise_results(accuracies))
    
    counter = 100
    #Display confusion matrices
    for matrix in matrices:
        ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')

        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class')

        # Ticket labels
        ax.xaxis.set_ticklabels(['Other', 'Eating'])
        ax.yaxis.set_ticklabels(['Other', 'Eating'])

        # Display the visualization of the Confusion Matrix.
        plt.savefig("Confusion matrix" + str(counter) + ".jpg")
        plt.show()
        
        counter = counter + 1


train_set = []
test_set = [0, 1, 2, 3, 4, 5, 6, 7]
for x in range(len(test_set)):
    train_temp = [0, 1, 2, 3, 4, 5, 6, 7]
    train_temp.remove(test_set[x])
    train_set.append(train_temp)


#Run the test
test(train_set, test_set)
