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


def activation_function_derivative(net_input):
    return 1 - tanh(net_input) ** 2


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

    # write the untrained weights to a file
    with open("output/weights_untrained.txt", "w") as f:
        for layer in weights_untrained:
            for n in range(len(layer)):
                f.write("\t".join([str(weight) for weight in layer[n]]) + "\n")

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

    # define operating parameters
    L = -0.75
    H = 0.75
    # train the network
    print("Training the network...")
    weights_trained = np.array(weights_untrained.copy(), dtype=object)
    error_fractions = []
    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        network_outputs = []
        for i in range(len(training_set)):
            inp = training_set[i]
            # convert the input to a list of floats
            points = [float(point)
                      for point in inp.split("\t")]
            derivatives = []
            outputs = []
            # calculate the output of the network
            for l in range(len(layers)):
                output_layer = []
                derivatives_layer = []
                for n in range(layers[l]):
                    weights_current = weights_trained[l][n]
                    net_input = calculate_net_input(
                        weights_current, [1] + points)  # add a bias input of 1
                    output_layer.append(activation_function(net_input))
                    derivatives_layer.append(
                        activation_function_derivative(net_input))
                points = output_layer
                outputs.append(output_layer)
                derivatives.append(derivatives_layer)
            # since points after the loop is the output of the output layer, it is now the output of the network, add it to the outputs list
            network_outputs.append(points)

            # back propogate
            deltas = np.zeros(len(layers), dtype=object)
            output = np.array(
                [1 if points[j] >= H and j == int(training_set_labels[i]) else -1 if points[j] <= L and int(training_set_labels[i]) != j else points[j] for j in range(len(points))])
            label = np.full(len(output), -1)  # Initialize label array with -1
            # Set the index from training_set_labels[i] to 1
            label[int(training_set_labels[i])] = 1
            # calculate the delta for the output layer
            deltas[-1] = np.multiply(
                derivatives[-1], np.subtract(label, output))
            # calculate the deltas for the hidden layers
            for l in range(len(layers) - 2, -1, -1):
                error_terms = np.zeros(layers[l], dtype=object)
                # skip the bias weight
                for n in range(1, layers[l]+1):
                    error_term = 0
                    for j in range(layers[l+1]):
                        error_term += weights_trained[l +
                                                      1][j][n] * deltas[l+1][j]
                    error_terms[n-1] = error_term
                deltas[l] = np.multiply(derivatives[l], error_terms)

            # update the weights
            inputs = [float(point) for point in inp.split("\t")] + outputs[:-1]
            for l in range(len(layers)):
                for n in range(layers[l]):
                    # update the bias weight
                    weights_trained[l][n][0] += learning_rate * deltas[l][n]
                    # skip the bias weight
                    for w in range(1, len(weights_trained[l][n])):
                        weights_trained[l][n][w] += learning_rate * \
                            deltas[l][n] * float(inputs[w-1])

        # get the error fraction of the network in training on the training set
        error_fraction_training = get_metrics(
            network_outputs, training_set_labels)
        print("Error fraction on training set:", error_fraction_training)
        error_fractions.append(error_fraction_training)

    # write the error fractions to a file
    with open("output/error_fractions_training.txt", "w") as f:
        for error in error_fractions:
            f.write(str(error) + "\n")
    print("Training complete.")

    # write the trained weights to a file
    with open("output/weights_trained.txt", "w") as f:
        for layer in weights_trained:
            for n in range(len(layer)):
                f.write("\t".join([str(weight) for weight in layer[n]]) + "\n")
