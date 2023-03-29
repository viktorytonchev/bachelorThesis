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
import random
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
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical

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
        if not flag and counter < 5:
            result_ground.append(annotation_to_category_just_eating("Leftover"))
        if flag or (not flag and counter < 5):
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


# This is for the file names
numbers = [0, 2, 3, 4, 5, 6, 8, 9]
files = file_names(numbers)
dictionaries = load_pickles(files)
acc_names, gyro_names = sensor_names(numbers)
data_acc, data_gyro = load_sensor_files(acc_names, gyro_names)


for x in range(len(numbers)):
    even_out(data_acc[x], data_gyro[x])


for x in dictionaries:
    fix_keys(x)


# The times at which calibration starts. These were all extracted manually by making plots of the data and looking for the 5 bumps made by the tapping of the sensor during the experiment
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
def train_test_data_variations():
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
    train = []
    test = []
    serial_numbers = []
    
    
    # Assign serial numbers to the data to help with preventing leakage
    for x in range(len(final_ground)):
        serial_counter = 0
        temp_serial = []
        for y in final_ground[x]:
            temp_serial.append(serial_counter)
            serial_counter = serial_counter + 1
        serial_numbers.append(temp_serial)
        
    
    for x in range(len(final_ground)):
        
        
        #Keep counters of how much train/test data has been assigned from each category for each person. This is to make sure there is a minimal number of gestures to test on.
        counters = [0]*8
        for y in range(len(final_ground[x])):
            counters[final_ground[x][y]] = counters[final_ground[x][y]] + 1
        train_counters = []
        test_counters = []
        for y in counters:
            train_counters.append(int(y*0.8))
            test_counters.append(int(y*0.2))
        
        
        # Shuffle all of the data plus the serial numbers in the same manner
        shuffled_x_acc, shuffled_y_acc, shuffled_z_acc, shuffled_x_gyro, shuffled_y_gyro, shuffled_z_gyro, shuffled_ground, shuffled_serial_numbers = shuffle(np.array(final_x_acc[x]), np.array(final_y_acc[x]), np.array(final_z_acc[x]), np.array(final_x_gyro[x]), np.array(final_y_gyro[x]), np.array(final_z_gyro[x]), np.array(final_ground[x]), np.array(serial_numbers[x]))
        
        
        temp_train  = []
        temp_test = []
        
        
        # Check if the next or previous window is assigned and assign the current one in the same way. This is to minimize leakage. If not assign randomly with 80% chance for train and 20% for test.
        for y in range(len(shuffled_ground)):
            if (((shuffled_serial_numbers[y] + 1) in temp_test) or ((shuffled_serial_numbers[y] - 1) in temp_test)) and test_counters[shuffled_ground[y]] > 0:
                temp_test.append(shuffled_serial_numbers[y])
                test_counters[shuffled_ground[x]] = counters[shuffled_ground[x]] - 1
            elif (((shuffled_serial_numbers[y] + 1) in temp_train) or ((shuffled_serial_numbers[y] - 1) in temp_train)) and train_counters[shuffled_ground[y]] > 0:
                temp_train.append(shuffled_serial_numbers[y])
                train_counters[shuffled_ground[y]] = train_counters[shuffled_ground[y]] - 1
            elif random.random() < 0.8 and train_counters[shuffled_ground[y]] > 0:
                temp_train.append(shuffled_serial_numbers[y])
                train_counters[shuffled_ground[y]] = train_counters[shuffled_ground[y]] - 1
            else:
                temp_test.append(shuffled_serial_numbers[y])
                test_counters[shuffled_ground[y]] = test_counters[shuffled_ground[y]] - 1
        
        train.append(temp_train)
        test.append(temp_test)
    
    
    # Make the final arrays with the training data
    for x in range(len(train)):
        for y in train[x]:
            final_train_x.append(final_x_acc[x][y])
    for x in range(len(train)):
        for y in train[x]:
            final_train_y.append(final_y_acc[x][y])
    for x in range(len(train)):
        for y in train[x]:
            final_train_z.append(final_z_acc[x][y])     
    for x in range(len(train)):
        for y in train[x]:
            final_train_x_gyro.append(final_x_gyro[x][y])
    for x in range(len(train)):
        for y in train[x]:
            final_train_y_gyro.append(final_y_gyro[x][y])
    for x in range(len(train)):
        for y in train[x]:
            final_train_z_gyro.append(final_z_gyro[x][y])       
    for x in range(len(train)):
        for y in train[x]:
            final_train_ground.append(final_ground[x][y])       
    
    
    # Make the final arrays with the testing data
    for x in range(len(test)):
        for y in test[x]:
            final_test_x.append(final_x_acc[x][y])
    for x in range(len(test)):
        for y in test[x]:
            final_test_y.append(final_y_acc[x][y])
    for x in range(len(test)):
        for y in test[x]:
            final_test_z.append(final_z_acc[x][y])     
    for x in range(len(test)):
        for y in test[x]:
            final_test_x_gyro.append(final_x_gyro[x][y])
    for x in range(len(test)):
        for y in test[x]:
            final_test_y_gyro.append(final_y_gyro[x][y])
    for x in range(len(test)):
        for y in test[x]:
            final_test_z_gyro.append(final_z_gyro[x][y]) 
    for x in range(len(test)):
        for y in test[x]:
            final_test_ground.append(final_ground[x][y]) 
    
    
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
def load_dataset():
	# Load all training and testing data
	trainX, trainy, testX, testy = train_test_data_variations()
	# One hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)    
	return trainX, trainy, testX, testy


# Fit and evaluate a model
def evaluate(trainX, trainy, testX, testy):
    # Parameters for the model
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # Define model
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Get a class weight dictionary
    y_ints = [y.argmax() for y in trainy]
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_ints),y=y_ints)
    class_weights_dict = dict(enumerate(class_weights))
    # Fit model
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight=class_weights_dict)
    # Predict probabilities for test set
    yhat_probs = model.predict(testX, verbose=0)
    # Predict crisp classes for test set
    yhat_classes = yhat_probs.argmax(axis=1)
    
    # Reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    testy = np.argmax(testy, axis=1)
    
    # Accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    # F1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes, average='weighted')
    # confusion matrix
    matrix = confusion_matrix(testy, yhat_classes)
    
    return f1, matrix, accuracy


# summarize scores    
def summarise_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    return m, s
    

# Run an experiment
def run_experiment():
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	return evaluate(trainX, trainy, testX, testy)


# Test the model in x repetitions.
def test(repeats = 5):
    
    print("Running test over " + str(repeats) + " repetitions of the LSTM algorithm." )
    
    scores = list()
    accuracies = list()
    matrices = list()
    
    for a in range(repeats):
    
        f1, matrix, accuracy = run_experiment()
        scores.append(f1)
        matrices.append(matrix)
        accuracies.append(accuracy)
        
        print('F1 score: %.3f' % f1)
        print('Accuracy: %.3f' % accuracy)
        
    print("Average F1 score over " + str(repeats) + " repetitions:  %.3f (+/-%.3f)" % summarise_results(scores))
    print("Average accuracy over " + str(repeats) + " repetitions:  %.3f (+/-%.3f)" % summarise_results(accuracies))
    
    
    counter = 100
    # Display confusion matrices
    for matrix in matrices:
        ax = sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')

        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class');

        # Ticket labels
        ax.xaxis.set_ticklabels(['Other', 'Eating'])
        ax.yaxis.set_ticklabels(['Other', 'Eating'])

        # Display the visualization of the Confusion Matrix.
        plt.savefig("Confusion matrix" + str(counter) + ".jpg")
        plt.show()
        
        counter = counter + 1


# Run the test
test()

