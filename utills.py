from random import seed, shuffle, randint, uniform
import numpy as np
from math import tanh


def generate_data_sets(digits):
    training_set = []
    training_set_labels = []

    test_set = []
    test_set_labels = []

    challenge_set = []
    challenge_set_labels = []

    # keeps track of the number of each digit in the training set
    training_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    # keeps track of the number of each digit in the test set
    test_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }
    # keeps track of the number of each digit in the challenge set
    challenge_set_count = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0
    }

    with open("data/MNISTnumImages5000_balanced.txt") as f1, open("data/MNISTnumLabels5000_balanced.txt") as f2:
        images = f1.readlines()
        labels = f2.readlines()

        # collect 400 images of each of the specified digits for the training set, 100 for the test set, and 100 for the challenge set
        for i in range(len(labels)):
            label = labels[i].strip()
            # training set
            if (label in digits) and training_set_count[label] < 400:
                training_set.append(images[i])
                training_set_labels.append(labels[i])
                training_set_count[label] += 1
            # test set
            elif (label in digits) and test_set_count[label] < 100:
                test_set.append(images[i])
                test_set_labels.append(labels[i])
                test_set_count[label] += 1
            # challenge set
            elif (label not in digits) and challenge_set_count[label] < 100:
                challenge_set.append(images[i])
                challenge_set_labels.append(labels[i])
                challenge_set_count[label] += 1

    # shuffle the training set and its labels in the same order
    random_seed = randint(1, 10)
    seed(random_seed)
    shuffle(training_set)
    seed(random_seed)
    shuffle(training_set_labels)

    # write out the sets and their labels to text files
    with open("output/training_set.txt", "w") as f, open("output/training_set_labels.txt", "w") as f2, open("output/test_set.txt", "w") as f3, open("output/test_set_labels.txt", "w") as f4, open("output/challenge_set.txt", "w") as f5, open("output/challenge_set_labels.txt", "w") as f6:
        for image in training_set:
            f.write(image)
        for label in training_set_labels:
            f2.write(label)

        for image in test_set:
            f3.write(image)
        for label in test_set_labels:
            f4.write(label)

        for image in challenge_set:
            f5.write(image)
        for label in challenge_set_labels:
            f6.write(label)


def initialize_weights(layers):
    weights = []
    for l in range(len(layers) - 1):
        layer_weights = []
        for _ in range(layers[l + 1]):
            layer_weights.append(
                [uniform(-1, 1) for _ in range(layers[l] + 1)])  # add 1 for the bias input
        weights.append(layer_weights)
    return weights


def calculate_net_input(weights, inputs):
    return np.dot(weights, inputs)


def activation_function(net_input):
    return tanh(net_input)


def get_metrics(output, labels):
    correct_predictions = 0
    for i in range(len(output)):
        prediction = output[i].index(max(output[i]))
        true = int(labels[i].strip())
        if prediction == true:
            correct_predictions += 1
    accuracy = correct_predictions / len(output)
    error_fraction = 1 - accuracy
    return error_fraction


def feed_forward(inputs, weights, layers):
    outputs = []
    for inp in inputs:
        points = [float(point) for point in inp.split("\t")]
        for l in range(len(layers)):
            output = []
            for n in range(layers[l]):
                weights_current = weights[l][n]
                net_input = calculate_net_input(
                    weights_current, [1] + points)  # add a bias input of 1
                output.append(activation_function(net_input))
            points = output
        outputs.append(points)
    return outputs


def simulate_back_propogation(layers, learning_rate, epochs):
    num_inputs = 784
    weights_untrained = initialize_weights([num_inputs] + layers)

    # read in the test set and its labels
    test_set = []
    test_set_labels = []
    with open("output/test_set.txt") as f, open("output/test_set_labels.txt") as f2:
        test_set = [line.strip() for line in f.readlines()]
        test_set_labels = [line.strip() for line in f2.readlines()]

    # get the output of the untrained network on the test set
    output_untrained_test_set = feed_forward(
        test_set, weights_untrained, layers)

    # get the error fraction of the untrained network on the test set
    error_fraction_untrained_test = get_metrics(
        output_untrained_test_set, test_set_labels)
    # write the error fraction to a file
    with open("output/error_fraction_untrained_test_set.txt", "w") as f:
        f.write(str(error_fraction_untrained_test))

    # read in the training set and its labels
    training_set = []
    training_set_labels = []
    with open("output/training_set.txt") as f, open("output/training_set_labels.txt") as f2:
        training_set = [line.strip() for line in f.readlines()]
        training_set_labels = [line.strip() for line in f2.readlines()]

    # train the network
    # for epoch in range(epochs):
    #     outputs = []
    #     for i in range(len(training_set)):
    #         inp = training_set[i]
    #         # convert the input to a list of floats and add a bias input of 1
    #         points = [1] + [float(point)
    #                         for point in inp.split("\t")]
    #         # calculate the output of the network
    #         for l in range(len(layers)):
    #             output = []
    #             for n in range(layers[l]):
    #                 weights_current = weights_untrained[l][n]
    #                 net_input = calculate_net_input(weights_current, points)
    #                 output.append(activation_function(net_input))
    #             points = output
    #         outputs.append(points)

    #         # back propogate
    #         deltas = []
        # calculate the delta of the output layer
