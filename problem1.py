from utills import generate_data_sets, simulate_back_propogation, feed_forward
from os import makedirs
import matplotlib.pyplot as plt
import numpy as np


makedirs("output", exist_ok=True)
makedirs("output/problem1", exist_ok=True)

generate_data_sets(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

learning_rate = 0.01
epochs = 150
layers = [125, 10]

# simulate_back_propogation(layers, learning_rate, epochs)

# load the weights and data sets
weights = []
weight_layers = [784] + layers
for l in range(len(weight_layers) - 1):
    layer_weights = []
    for _ in range(weight_layers[l + 1]):
        layer_weights.append([0 for _ in range(weight_layers[l] + 1)])
    weights.append(layer_weights)
training_set = []
training_set_labels = []
print("Loading weights and data sets...")
with open("output/problem1/weights_trained.txt", "r") as f1, open("output/training_set.txt", "r") as f2, open("output/training_set_labels.txt", "r") as f3, open("output/test_set.txt", "r") as f4, open("output/test_set_labels.txt", "r") as f5:
    for i in range(len(weight_layers) - 1):
        for j in range(weight_layers[i + 1]):
            weights[i][j] = [float(weight.strip())
                             for weight in f1.readline().split("\t")]
    training_set = [line.strip() for line in f2.readlines()]
    training_set_labels = [line.strip() for line in f3.readlines()]
    test_set = [line.strip() for line in f4.readlines()]
    test_set_labels = [line.strip() for line in f5.readlines()]

# get the training set output and create the confusion matrix
print("Creating confusion matrix for training set...")
training_set_output = feed_forward(training_set, weights, layers)
confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
for i in range(len(training_set_output)):
    output = training_set_output[i].index(max(training_set_output[i]))
    confusion_matrix[int(training_set_labels[i])][output] += 1

# plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
# ... and label them with the respective list entries
ax.set_xticklabels([str(i) for i in range(10)])
ax.set_yticklabels([str(i) for i in range(10)])

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, confusion_matrix[i][j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion Matrix Training Set")
fig.tight_layout()
plt.savefig("output/problem1/confusion_matrix_training_set.png")
plt.show()

# get the test set output and create the confusion matrix
print("Creating confusion matrix for test set...")
test_set_output = feed_forward(test_set, weights, layers)
confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
for i in range(len(test_set_output)):
    output = test_set_output[i].index(max(test_set_output[i]))
    confusion_matrix[int(test_set_labels[i])][output] += 1

# plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
# ... and label them with the respective list entries
ax.set_xticklabels([str(i) for i in range(10)])
ax.set_yticklabels([str(i) for i in range(10)])

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, confusion_matrix[i][j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion Matrix Test Set")
fig.tight_layout()
plt.savefig("output/problem1/confusion_matrix_test_set.png")
plt.show()

error_fractions_training = []
error_fractions_test = []
with open("output/problem1/error_fractions_training.txt", "r") as f1, open("output/problem1/error_fractions_training_test_set.txt", "r") as f2:
    lines = f1.readlines()
    for i in range(len(lines)):
        if (i+1) % 10 == 0:
            error_fractions_training.append(float(lines[i].strip()))
        lines[i] = lines[i].strip()
    error_fractions_test = [float(line.strip()) for line in f2.readlines()]

# plot the time series of error fractions
# Assuming error_fractions_training and error_fractions_test have length 15
epochs = range(10, 151, 10)  # epochs 10, 20, ..., 150

fig, ax = plt.subplots()

# Plotting the data with the corresponding epochs
ax.plot(epochs, error_fractions_training, label="Training Set", marker='o')
ax.plot(epochs, error_fractions_test, label="Test Set", marker='o')

# Set x-ticks to be 10, 20, ..., 150
ax.set_xticks(epochs)

ax.set_xlabel("Epochs")
ax.set_ylabel("Error Fraction")
ax.set_title("Error Fraction vs Epochs")
ax.legend()

plt.savefig("output/problem1/error_fractions_time_series.png")
plt.show()
