import math
import numpy as np
import matplotlib.pyplot as plt
import create_trainable_set
import laspy
import os
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow import keras
import pickle
import csv
from scipy.signal import lfilter

"""
These are used to label the data. The first file is the first file in the dataset, and the fourth file is the fourth file in the dataset.
The change times are the times at which the labels change, and the labels are the labels at those times.
The first file is 255 seconds long, and the fourth file is 531 seconds long.

The reason that files 2 and 3 are not included is because they both only contain one label (4) for the entire file. This is not useful for training.
"""
first_file_change_times = [0, 54, 77, 83, 93, 101, 112, 120, 131, 140, 155, 175, 189, 199, 216, 222, 240]
first_file_labels = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 3, 5, 3, 5, 3, 5]
first_file_len = 255

fourth_file_change_times = [7, 22, 31, 37, 48, 51, 54, 64, 71, 82, 138, 184, 195, 204, 213, 233, 243, 271, 278, 282,
                            358, 371, 387, 413, 448, 450, 458, 478, 491, 492, 495, 496]
fourth_file_labels = [3, 4, 4, 3, 4, 2, 3, 4, 2, 3, 4, 3, 4, 3, 2, 3, 2, 3, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0]
fourth_file_len = 531


def get_data_from_directory(directory, return_scan_names=False):
    """
    This function takes in a directory and returns the lists of data and labels for the data in that directory.
    :param directory: the directory to get the data from
    :return: (data_list, data_labels) where data_list is a list of scans (each scan contains lists of points and their
                properties), and data_labels is a list of labels for each scan
    """

    # create the label dictionaries
    first_label_lookup = {}
    fourth_label_lookup = {}
    for i in range(max(first_file_len + 1, fourth_file_len + 1)):
        for j in range(len(first_file_change_times)):
            if first_file_change_times[j] > i:
                first_label_lookup[i] = first_file_labels[j-1]
                break
            elif j == len(first_file_change_times) - 1:
                first_label_lookup[i] = first_file_labels[j]
                break
        for j in range(len(fourth_file_change_times)):
            if fourth_file_change_times[j] > i:
                fourth_label_lookup[i] = fourth_file_labels[j-1]
                break
            elif j == len(fourth_file_change_times) - 1:
                fourth_label_lookup[i] = fourth_file_labels[j]
                break

    # create the list and corresponding labels of our data
    data_list = []
    data_labels = []
    scan_names = []
    for file in create_trainable_set.get_all_files_in_directory(directory=directory):
        las = laspy.read(file)

        points = [las.x, las.y, las.z, las.intensity]
        data_list.append(points)
        scan_names.append(file)
        if 'second' in file or 'third' in file:
            data_labels.append(4) # 4 is the label for all .las files in the second and third scans
        elif 'first' in file:
            basename = os.path.basename(file)
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            data_labels.append(first_label_lookup[file_no])
        elif 'fourth' in file:
            basename = os.path.basename(file)
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            data_labels.append(fourth_label_lookup[file_no])
        else:
            print("ERROR: Cannot label file: " + file)

    if return_scan_names:
        return data_list, data_labels, scan_names
    return data_list, data_labels


def voxelize_and_normalize_data(data_list, num_voxels_per_dimension):
    """
    This function takes in a list of scans and returns a list of voxelized and normalized scans.
    Note that in order to pass this into tensorflow functions, you will need to cast this to a numpy array.

    :param data_list: a list of scans, each scan being a list of x values, a list of y values, a list of z values,
                        and a list of intensity values
    :param num_voxels_per_dimension: the number of voxels per dimension to divide the scans into
                                        (i.e. 32 means 32x32x32 voxels)
    :return: a list of voxelized scans, each scan being a 3D array of the number of points in each voxel
    """

    voxelized_list = []
    for i in tqdm(range(len(data_list))):
        scan = voxelize_and_normalize_scan(data_list[i], num_voxels_per_dimension)
        voxelized_list.append(scan)
    return voxelized_list


def voxelize_and_normalize_data_with_intensity(data_list, num_voxels_per_dimension):
    """
    This function takes in a list of scans and returns a list of voxelized and normalized scans.
    Note that in order to pass this into tensorflow functions, you will need to cast this to a numpy array.

    This function differs from voxelize_and_normalize_data in that it also utilizes the intensity values of the points.
    The average intensity value of the points in each voxel is stored in the voxelized scan.
    Because of this, the voxelized scans will be 4D arrays instead of 3D arrays.
    However, you can think of them as 3D arrays, where each voxel simply has two properties:
        the number of points in the voxel and the average intensity value of the points in the voxel.

    :param data_list: a list of scans, each scan being a list of x values, a list of y values, a list of z values,
                        and a list of intensity values
    :param num_voxels_per_dimension: the number of voxels per dimension to divide the scans into
                                        (i.e. 32 means 32x32x32 voxels)
    :return: a list of voxelized scans, each scan being a 3D array of the number of points in each voxel
    """
    voxelized_list = []
    for i in tqdm(range(len(data_list))):
        scan = voxelize_and_normalize_scan_with_intensity(data_list[i], num_voxels_per_dimension)
        voxelized_list.append(scan)
    return voxelized_list


def voxelize_and_normalize_scan(scan, num_voxels_per_dimension):
    """
    This function takes in a scan and returns a voxelized and normalized scan.
    :param scan: a list of... a list of x values, a list of y values, a list of z values, and a list of intensity values
    :param num_voxels_per_dimension: the number of voxels per dimension to divide the scans into
                                        (i.e. 32 means 32x32x32 voxels)
    :return: the voxelized scan, which is a 3D array of the number of points in each voxel
    """

    x_vals = scan[0]
    y_vals = scan[1]
    z_vals = scan[2]
    intensity_vals = scan[3]
    min_x = min(x_vals)
    max_x = max(x_vals)
    min_y = min(y_vals)
    max_y = max(y_vals)
    min_z = min(z_vals)
    max_z = max(z_vals)
    min_intensity = min(intensity_vals)
    max_intensity = max(intensity_vals)

    voxel_dict = {}

    for i in range(len(x_vals)):
        this_x = x_vals[i]
        this_y = y_vals[i]
        this_z = z_vals[i]
        this_intensity = intensity_vals[i]

        # find the voxel that this point belongs to
        x_voxel = int((this_x - min_x) / (max_x - min_x) * (num_voxels_per_dimension - 1))
        y_voxel = int((this_y - min_y) / (max_y - min_y) * (num_voxels_per_dimension - 1))
        z_voxel = int((this_z - min_z) / (max_z - min_z) * (num_voxels_per_dimension - 1))
        if voxel_dict.get((x_voxel, y_voxel, z_voxel)) is None:
            voxel_dict[(x_voxel, y_voxel, z_voxel)] = [(this_intensity - min_intensity) / (max_intensity - min_intensity)]
        else:
            voxel_dict[(x_voxel, y_voxel, z_voxel)].append((this_intensity - min_intensity) / (max_intensity - min_intensity))

    max_density = 0
    min_density = math.inf
    for key in voxel_dict.keys():
        density = len(voxel_dict[key])
        if density > max_density:
            max_density = density
        if density < min_density:
            min_density = density

    voxelized_scan = np.zeros((num_voxels_per_dimension, num_voxels_per_dimension, num_voxels_per_dimension))

    for key in voxel_dict.keys():
        # for now each voxel will only contain the density of points in that voxel
        voxelized_scan[key[0]][key[1]][key[2]] = float(len(voxel_dict[key]) - min_density) / (max_density - min_density)

    return voxelized_scan


def voxelize_and_normalize_scan_with_intensity(scan, num_voxels_per_dimension):
    """
    This function takes in a scan and returns a voxelized and normalized scan.
    Note that this function differs from voxelize_and_normalize_scan in that it also utilizes the intensity values of the points.
    The resulting voxelized scan will be a 4D array instead of a 3D array.
    However, you can think of it as a 3D array, where each voxel simply has two properties:
        the number of points in the voxel and the average intensity value of the points in the voxel.

    :param scan: a list of... a list of x values, a list of y values, a list of z values, and a list of intensity values
    :param num_voxels_per_dimension: the number of voxels per dimension to divide the scans into
                                        (i.e. 32 means 32x32x32 voxels)
    :return: the voxelized scan, which is a 3D array of the number of points in each voxel
    """
    x_vals = scan[0]
    y_vals = scan[1]
    z_vals = scan[2]
    intensity_vals = scan[3]
    min_x = min(x_vals)
    max_x = max(x_vals)
    min_y = min(y_vals)
    max_y = max(y_vals)
    min_z = min(z_vals)
    max_z = max(z_vals)
    min_intensity = min(intensity_vals)
    max_intensity = max(intensity_vals)

    voxel_dict = {}

    for i in range(len(x_vals)):
        this_x = x_vals[i]
        this_y = y_vals[i]
        this_z = z_vals[i]
        this_intensity = intensity_vals[i]

        # find the voxel that this point belongs to
        x_voxel = int((this_x - min_x) / (max_x - min_x) * (num_voxels_per_dimension - 1))
        y_voxel = int((this_y - min_y) / (max_y - min_y) * (num_voxels_per_dimension - 1))
        z_voxel = int((this_z - min_z) / (max_z - min_z) * (num_voxels_per_dimension - 1))
        if voxel_dict.get((x_voxel, y_voxel, z_voxel)) is None:
            voxel_dict[(x_voxel, y_voxel, z_voxel)] = [(this_intensity - min_intensity) / (max_intensity - min_intensity)]
        else:
            voxel_dict[(x_voxel, y_voxel, z_voxel)].append((this_intensity - min_intensity) / (max_intensity - min_intensity))

    max_density = 0
    min_density = math.inf
    for key in voxel_dict.keys():
        density = len(voxel_dict[key])
        if density > max_density:
            max_density = density
        if density < min_density:
            min_density = density

    voxelized_scan = np.zeros((num_voxels_per_dimension, num_voxels_per_dimension, num_voxels_per_dimension, 2))

    for key in voxel_dict.keys():
        # each voxel will contain an ordered pair of the density of points in that voxel and the average intensity of those points
        voxelized_scan[key[0]][key[1]][key[2]][:] = np.array(float(len(voxel_dict[key]) - min_density) / (max_density - min_density), np.mean(voxel_dict[key])) #TODO: make sure this really works

    return voxelized_scan


def normalize_labels(label_list):
    """
    This function takes in a list of labels and returns a list of normalized labels.
    :param label_list: the list of labels to normalize
    :return: the list of normalized labels
    """
    biggest = max(label_list)
    for i in range(len(label_list)):
        label_list[i] = float(label_list[i]) / (biggest + 0.0001) #TODO: find some way to do this better
    return label_list


def plot_voxelized_scan(scan_to_plot):
    """
    This function takes in a voxelized scan and plots it, allowing you to visualize the scan the way the computer sees it.
    Currently, it only supports 3D scans and cannot utilize the intensity values of the points.
    Instead, it simply colors each voxel based on the number of points in that voxel.

    :param scan_to_plot: the voxelized scan to plot
    :return: nothing
    """
    colors = np.empty(scan_to_plot.shape, dtype=object)
    Blues = plt.get_cmap('Blues')
    for i in range(scan_to_plot.shape[0]):
        for j in range(scan_to_plot.shape[1]):
            for k in range(scan_to_plot.shape[2]):
                colors[i][j][k] = Blues(scan_to_plot[i][j][k])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(scan_to_plot, facecolors=colors, edgecolor='k')
    plt.show()


def one_hot_labels(label_list):
    """
    This function takes in a list of labels and returns a list of one-hot labels.
    Note that as of yet, this function only supports labels that are 0, 1, 2, 3, 4, or 5.

    :param label_list: the list of labels to one-hot
    :return: the list of one-hot labels
    """
    one_hot_list = []
    for i in range(len(label_list)):
        if label_list[i] == 0:
            one_hot_list.append((1, 0, 0, 0, 0, 0))
        elif label_list[i] == 1:
            one_hot_list.append((0, 1, 0, 0, 0, 0))
        elif label_list[i] == 2:
            one_hot_list.append((0, 0, 1, 0, 0, 0))
        elif label_list[i] == 3:
            one_hot_list.append((0, 0, 0, 1, 0, 0))
        elif label_list[i] == 4:
            one_hot_list.append((0, 0, 0, 0, 1, 0))
        elif label_list[i] == 5:
            one_hot_list.append((0, 0, 0, 0, 0, 1))
    return one_hot_list


def one_hot_to_int(one_hot_list):
    new_list = []
    for current_tuple in one_hot_list:
        if current_tuple == (1, 0, 0, 0, 0, 0):
            new_list.append(0)
        elif current_tuple == (0, 1, 0, 0, 0, 0):
            new_list.append(1)
        elif current_tuple == (0, 0, 1, 0, 0, 0):
            new_list.append(2)
        elif current_tuple == (0, 0, 0, 1, 0, 0):
            new_list.append(3)
        elif current_tuple == (0, 0, 0, 0, 1, 0):
            new_list.append(4)
        elif current_tuple == (0, 0, 0, 0, 0, 1):
            new_list.append(5)
        else:
            print("Error: invalid one-hot tuple")
    return new_list


def create_new_classifier(num_voxels_per_dimension, train_directory=None, train_data=None, train_labels=None):
    """
    This function creates a new classifier and trains it on the data provided.

    :param num_voxels_per_dimension: the number of voxels per dimension to use
    :param train_directory: the directory containing the training data
                                (if not provided, train_data and train_labels must be provided)
    :param train_data: a 3D numpy array containing the training data
    :param train_labels: a 1D numpy array containing the training labels
    :return: the trained classifier
    """

    if train_data is None or train_labels is None:
        if train_directory is None:
            print("Error: either provide training data and labels or a directory containing training data")
            return
        else:
            print("Loading training data...")
            # Load the data
            train_data, train_labels = get_data_from_directory(train_directory)
            train_labels = np.array(one_hot_labels(train_labels))
            print("Voxelizing and normalizing data...")
            train_data = np.array(voxelize_and_normalize_data(train_data, num_voxels_per_dimension))

    print("Creating model...")
    # Create the model
    input_shape = (num_voxels_per_dimension, num_voxels_per_dimension, num_voxels_per_dimension, 1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])

    print("Compiling model...")
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    print("Training model...")
    # Train the model
    batch_size = 4
    model.fit(train_data, epochs=10, y=train_labels, batch_size=batch_size)

    print("Done!")
    return model


def create_new_classifier_with_intensity(num_voxels_per_dimension, train_directory=None, train_data=None, train_labels=None):
    """
    This function creates a new classifier and trains it on the data provided.


    :param num_voxels_per_dimension: the number of voxels per dimension to use
    :param train_directory: the directory containing the training data
                                (if not provided, train_data and train_labels must be provided)
    :param train_data: a 3D numpy array containing the training data
                        (if not provided, train_directory must be provided)
    :param train_labels: a 1D numpy array containing the training labels
                            (if not provided, train_directory must be provided)
    :return: the trained classifier
    """

    if train_data is None or train_labels is None:
        if train_directory is None:
            print("Error: either provide training data and labels or a directory containing training data")
            return
        else:
            print("Loading training data...")
            # Load the data
            train_data, train_labels = get_data_from_directory(train_directory)
            train_labels = np.array(one_hot_labels(train_labels))
            print("Voxelizing and normalizing data...")
            train_data = np.array(voxelize_and_normalize_data_with_intensity(train_data, num_voxels_per_dimension))

    print("Creating model...")
    # Create the model
    input_shape = (num_voxels_per_dimension, num_voxels_per_dimension, num_voxels_per_dimension, 2)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])

    print("Compiling model...")
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    print("Training model...")
    # Train the model
    batch_size = 4
    model.fit(train_data, epochs=10, y=train_labels, batch_size=batch_size)

    print("Done!")
    return model


def load_model(path_to_model):
    """
    This function loads a model from a file.
    :param path_to_model: the path to the model file
    :return: the loaded model
    """
    model = tf.keras.models.load_model(path_to_model)
    return model


def test_model(model, num_voxels_per_dimension, test_directory=None, test_data=None, test_labels=None):
    """
    This function tests a model on the data provided.
    Predictions (what the model guessed) are saved to predictions.csv in the current directory.
    Realities (the correct labels) are saved to realities.csv in the current directory.

    :param model: the model to test
    :param num_voxels_per_dimension: the number of voxels per dimension to use
                                        (it is very important that this is the same as used when training the model)
    :param test_directory: the directory containing the test data
    :param test_data: a 3D numpy array containing the test data
    :param test_labels: a 1D numpy array containing the test labels
    :return: Nothing
    """
    if test_data is None or test_labels is None:
        if test_directory is None:
            print("Error: either provide test data and labels or a directory containing test data")
            return
        else:
            print("Loading test data...")
            # Load the data
            test_data, test_labels = get_data_from_directory(test_directory)
            test_labels = np.array(one_hot_labels(test_labels))
            print("Voxelizing and normalizing data...")
            test_data = np.array(voxelize_and_normalize_data(test_data, num_voxels_per_dimension))

    # Make predictions
    predictions = model.predict(test_data, batch_size=4)

    # Save the predictions
    np.savetxt('predictions.csv', np.argmax(predictions, axis=1), delimiter=',', fmt='%d')
    np.savetxt('realities.csv', np.argmax(test_labels, axis=1), delimiter=',', fmt='%d')

    print("Done!")


def test_model_with_intensity(model, num_voxels_per_dimension, test_directory=None, test_data=None, test_labels=None):
    """
        This function tests a model on the data provided.
        Predictions (what the model guessed) are saved to predictions.csv in the current directory.
        Realities (the correct labels) are saved to realities.csv in the current directory.

        Note that this function differs from test_model in that it uses the intensity of the voxels.
        The provided model must have been trained with intensity, otherwise there will be a dimension mismatch.

        :param model: the model to test
        :param num_voxels_per_dimension: the number of voxels per dimension to use
                                            (it is very important that this is the same as used when training the model)
        :param test_directory: the directory containing the test data
        :param test_data: a 4D numpy array containing the test data
        :param test_labels: a 1D numpy array containing the test labels
        :return: Nothing
    """

    if test_data is None or test_labels is None:
        if test_directory is None:
            print("Error: either provide test data and labels or a directory containing test data")
            return
        else:
            print("Loading test data...")
            # Load the data
            test_data, test_labels = get_data_from_directory(test_directory)
            print("Voxelizing and normalizing data...")
            test_data = np.array(voxelize_and_normalize_data_with_intensity(test_data, num_voxels_per_dimension))

    # Make predictions
    predictions = model.predict(test_data, batch_size=4)

    # Save the predictions
    np.savetxt('predictions.csv', np.argmax(predictions, axis=1), delimiter=',', fmt='%d')
    np.savetxt('realities.csv', np.argmax(test_labels, axis=1), delimiter=',', fmt='%d')

    print("Done!")


def test_model_with_intensity_and_scan_name(model, test_data):
    predictions = model.predict(test_data, batch_size=4)
    return predictions


def separate_by_scan_name(data, scan_names):
    first_data = []
    first_times = []
    second_data = []
    second_times = []
    third_data = []
    third_times = []
    fourth_data = []
    fourth_times = []
    for i in range(len(scan_names)):
        if "first" in scan_names[i]:
            first_data.append(data[i])
            basename = os.path.basename(scan_names[i])
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            first_times = np.append(first_times, file_no)
        elif "second" in scan_names[i]:
            second_data.append(data[i])
            basename = os.path.basename(scan_names[i])
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            second_times = np.append(second_times, file_no)
        elif "third" in scan_names[i]:
            third_data.append(data[i])
            basename = os.path.basename(scan_names[i])
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            third_times = np.append(third_times, file_no)
        elif "fourth" in scan_names[i]:
            fourth_data.append(data[i])
            basename = os.path.basename(scan_names[i])
            file_no = int(basename[basename.find('_') + 1:basename.find('.')])
            fourth_times = np.append(fourth_times, file_no)
        else:
            print("Error: scan name not recognized")

    return first_data, first_times, second_data, second_times, third_data, third_times, fourth_data, fourth_times


def plot_prediction_vs_reality_vs_yolo(prediction_list, prediction_time_labels, yolo_list, yolo_time_labels, reality_list, reality_time_labels):

    # Sort the lists by time
    prediction_time_labels, prediction_list = zip(*sorted(zip(prediction_time_labels, prediction_list)))
    reality_time_labels, reality_list = zip(*sorted(zip(reality_time_labels, reality_list)))
    yolo_time_labels, yolo_list = zip(*sorted(zip(yolo_time_labels, yolo_list)))


    # Create a new figure
    plt.figure()

    # Plot yolo vs time as a green line first
    # since YOLO data is frame by frame, we'll smooth it out a bit
    n = 450  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    filtered_yolo_list = lfilter(b, a, yolo_list)
    plt.plot(yolo_time_labels, yolo_list, color='green', label='yolo')

    # Plot prediction vs time as a blue line
    plt.plot(prediction_time_labels, prediction_list, color='blue', label='LiDAR-Based Prediction')

    # Plot reality vs time as a red line
    plt.plot(reality_time_labels, reality_list, color='red', label='reality')

    # Set the x-axis label
    plt.xlabel('time')

    # Set the y-axis label
    plt.ylabel('number of people')

    # Set the plot title
    plt.title('Line Graph of prediction and reality over time')

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    plt.show()


def evenly_split_data(data, labels):
    """
    This function splits the data and labels into two sets, one with even indices and one with odd indices.
    :param data: a list of data
    :param labels: a list of labels
    :return: (even_data, even_labels, odd_data, odd_labels)
    """

    even_data = []
    even_labels = []
    odd_data = []
    odd_labels = []
    for i in range(len(data)):
        if i % 2 == 0:
            even_data.append(data[i])
            even_labels.append(labels[i])
        else:
            odd_data.append(data[i])
            odd_labels.append(labels[i])
    return even_data, even_labels, odd_data, odd_labels


def randomly_split_data(data, labels, percent_test, scan_names=None):
    """
    This function randomly splits the data and labels into two sets, one for training and one for testing.

    :param data: the data to split
    :param labels: the labels to split
    :param percent_test: the percent of the data to be placed in the test set
    :return: (train_data, train_labels, test_data, test_labels)
    """

    test_data = []
    test_labels = []
    train_data = []
    train_labels = []
    if scan_names is not None:
        test_scan_names = []
        train_scan_names = []
    for i in range(len(data)):
        if random.random() < percent_test:
            test_data.append(data[i])
            test_labels.append(labels[i])
            if scan_names is not None:
                test_scan_names.append(scan_names[i])
        else:
            train_data.append(data[i])
            train_labels.append(labels[i])
            if scan_names is not None:
                train_scan_names.append(scan_names[i])
    if scan_names is not None:
        return train_data, train_labels, test_data, test_labels, train_scan_names, test_scan_names
    return train_data, train_labels, test_data, test_labels


def save_processed_data_and_labels(directory_of_data):
    all_data, all_labels, scan_names = get_data_from_directory(directory=directory_of_data, return_scan_names=True)
    all_data = voxelize_and_normalize_data_with_intensity(all_data, 10)
    all_labels = one_hot_labels(all_labels)

    with open("alldata.pkl", "wb") as f:
        pickle.dump(all_data, f)
    with open("alllabels.pkl", "wb") as f:
        pickle.dump(all_labels, f)
    with open("scan_names.pkl", "wb") as f:
        pickle.dump(scan_names, f)


def load_processed_data_and_labels_from_file(return_scan_names=False):
    with open("alldata.pkl", "rb") as f:
        all_data = pickle.load(f)
    with open("alllabels.pkl", "rb") as f:
        all_labels = pickle.load(f)
    with open("scan_names.pkl", "rb") as f:
        scan_names = pickle.load(f)
    
    if return_scan_names:
        return all_data, all_labels, scan_names
    return all_data, all_labels


def load_yolo_results_from_csv(filename):
    vid1_results = []
    vid1_times = []
    vid2_results = []
    vid2_times = []
    vid3_results = []
    vid3_times = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader.__next__() # skip header
        lineno = 0
        for line in reader:
            if line[0] != '':
                vid1_results.append(int(line[0]))
                vid1_times.append(lineno / 30) # 30 fps
            if line[1] != '':
                vid2_results.append(int(line[1]))
                vid2_times.append(lineno / 30) # 30 fps
            if line[2] != '':
                vid3_results.append(int(line[2]))
                vid3_times.append(lineno / 30) # 30 fps
            lineno += 1
    
    return vid1_results, vid1_times, vid2_results, vid2_times, vid3_results, vid3_times


def downsample_to_given_times(times, data, new_times):
    """
    This function takes a list of times and a list of data and returns a new list of data that is downsampled to the given new_times
    """
    new_data = []
    for i in range(len(new_times)):
        j = 0
        while j < len(data) - 1 and times[j] < new_times[i]:
            j += 1
        new_data.append(data[j])
    return new_data 

#save_processed_data_and_labels('trainable_set')

all_data, all_labels, scan_names = load_processed_data_and_labels_from_file(return_scan_names=True)

train_data, train_labels, test_data, test_labels, train_scan_names, test_scan_names = randomly_split_data(all_data, all_labels, 0.2, scan_names=scan_names)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

my_classifier = create_new_classifier_with_intensity(10, train_data=train_data, train_labels=train_labels)

#my_classifier.save('new_ten_dim_random_split_with_intensity.h5')

#my_classifier = load_model('ten_dim_random_split_with_intensity.h5')

#test_model_with_intensity(my_classifier, 10, test_data=test_data, test_labels=test_labels, test_scan_names=test_scan_names)
predictions = test_model_with_intensity_and_scan_name(my_classifier, test_data=test_data)
predictions = np.argmax(predictions, axis=1)
first_predictions, first_p_times, second_predictions, second_p_times, third_predictions, third_p_times, fourth_predictions, fourth_p_times = separate_by_scan_name(predictions, test_scan_names)

first_truths, first_times, second_truths, second_times, third_truths, third_times, fourth_truths, fourth_times = separate_by_scan_name(all_labels, scan_names)
fourth_truths = one_hot_to_int(fourth_truths)
first_truths = one_hot_to_int(first_truths)
second_truths = one_hot_to_int(second_truths)

# yolo video 1 corresponds to first lidar scan, yolo video 3 corresponds to fourth lidar scan
yolo_1, yolo_1_times, yolo_2, yolo_2_times, yolo_3, yolo_3_times = load_yolo_results_from_csv('yolo_labels/yolo_labels.csv')

plot_prediction_vs_reality_vs_yolo(first_predictions, first_p_times, yolo_1, yolo_1_times, first_truths, first_times)
#plot_prediction_vs_reality_vs_yolo(second_predictions, second_p_times, yolo_2, yolo_2_times, second_truths, second_times)
plot_prediction_vs_reality_vs_yolo(fourth_predictions, fourth_p_times, yolo_3, yolo_3_times, fourth_truths, fourth_times)

# downsample to only show YOLO and reality predictions at the moment that a NN prediction is made
new_yolo_1 = downsample_to_given_times(yolo_1_times, yolo_1, first_p_times)
new_yolo_3 = downsample_to_given_times(yolo_3_times, yolo_3, fourth_p_times)
new_first_truths = downsample_to_given_times(first_times, first_truths, first_p_times)
new_fourth_truths = downsample_to_given_times(fourth_times, fourth_truths, fourth_p_times)

plot_prediction_vs_reality_vs_yolo(first_predictions, first_p_times, new_yolo_1, first_p_times, new_first_truths, first_p_times)
plot_prediction_vs_reality_vs_yolo(fourth_predictions, fourth_p_times, new_yolo_3, fourth_p_times, new_fourth_truths, fourth_p_times)