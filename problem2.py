from utills import generate_data_sets, simulate_reconstruction_learning
from os import makedirs
import matplotlib.pyplot as plt
from random import sample
import numpy as np


makedirs("output", exist_ok=True)
makedirs("output/problem2", exist_ok=True)

generate_data_sets(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

learning_rate = 0.01
epochs = 15
layers = [125, 784]

# simulate_reconstruction_learning(layers, learning_rate, epochs)

mean_reconstruction_errors_training = []
mean_reconstruction_errors_test = []
standard_deviations_test = []
mean_reconstruction_error_training = 0
mean_reconstruction_error_test = 0
with open("output/problem2/mean_reconstruction_errors_training_set.txt") as f1, open("output/problem2/mean_reconstruction_errors_test_set.txt") as f2, open("output/problem2/standard_deviations_test_set.txt") as f3, open("output/problem2/mean_reconstruction_error_training_set.txt") as f4, open("output/problem2/mean_reconstruction_error_test_set.txt") as f5:
    mean_reconstruction_errors_training = [
        float(err) for err in f1.readlines()]
    mean_reconstruction_errors_test = [float(err) for err in f2.readlines()]
    standard_deviations_test = [float(err) for err in f3.readlines()]
    lines = f4.readlines()
    mean_reconstruction_error_training = float(lines[0])
    lines = f5.readlines()
    mean_reconstruction_error_test = float(lines[0])

# plot mean reconstruction errors for training and test set as bar chart
fig, ax = plt.subplots()
ax.bar(["Training Set", "Test Set"], [mean_reconstruction_error_training,
                                      mean_reconstruction_error_test], color=["blue", "orange"])
ax.set_xlabel("Data Set")
ax.set_ylabel("Mean Reconstruction Error")
ax.set_title("Mean Reconstruction Error for Training and Test Set")
plt.savefig("output/problem2/mean_reconstruction_error_bar_chart.png")
plt.show()

# plot mean reconstruction errors and standard deviations for each digit in the test set as a table of 10 rows and 2 columns
fig, ax = plt.subplots()
table_data = []
for i in range(10):
    table_data.append([mean_reconstruction_errors_test[i],
                      standard_deviations_test[i]])
ax.table(cellText=table_data, colLabels=["Mean Reconstruction Error", "Standard Deviation"], rowLabels=[
         str(i) for i in range(10)], loc="center")
ax.axis("off")

plt.savefig("output/problem2/mean_reconstruction_error_table.png")
plt.show()

# plot the time series of mean reconstruction errors for training set
fig, ax = plt.subplots()
ax.plot(mean_reconstruction_errors_training)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Reconstruction Error")
ax.set_title("Mean Reconstruction Error vs Epoch for Training Set")
plt.savefig("output/problem2/mean_reconstruction_error_training.png")
plt.show()

# read in trained weights for problem 2
weights2 = []
with open("output/problem2/weights_trained.txt") as f:
    lines = f.readlines()
    for line in lines:
        weights2.append([float(w) for w in line.strip().split("\t")][1:])

hidden_neuron_weights2 = weights2[:125]
sample_hidden_neuron_weights2 = sample(hidden_neuron_weights2, 20)

# read in trained weights for problem 1
weights1 = []
with open("output/problem1/weights_trained.txt") as f:
    lines = f.readlines()
    for line in lines:
        weights1.append([float(w) for w in line.strip().split("\t")][1:])
hidden_neuron_weights1 = weights1[:125]
sample_hidden_neuron_weights1 = sample(hidden_neuron_weights1, 20)

# normalize the trained weights for problem 1 between 0 and 1, reshape them to 28x28 and plot as 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(sample_hidden_neuron_weights1):
        arr = np.array(sample_hidden_neuron_weights1[i])
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        pixel_matrix = arr.reshape((28, 28))
        pixel_matrix = np.rot90(pixel_matrix, k=-1)
        pixel_matrix = np.fliplr(pixel_matrix)
        ax.imshow(pixel_matrix, cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')

fig.suptitle(
    "Plot of Trained Weights for 20 Random Hidden Neurons in Problem 1")

plt.tight_layout()
plt.savefig("output/problem2/sample_hidden_neuron_weights1.png")
plt.show()

# normalize the trained weights for problem 2 between 0 and 1, reshape them to 28x28 and plot as 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(sample_hidden_neuron_weights2):
        arr = np.array(sample_hidden_neuron_weights2[i])
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        pixel_matrix = arr.reshape((28, 28))
        pixel_matrix = np.rot90(pixel_matrix, k=-1)
        pixel_matrix = np.fliplr(pixel_matrix)
        ax.imshow(pixel_matrix, cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')

fig.suptitle(
    "Plot of Trained Weights for 20 Random Hidden Neurons in Problem 2")

plt.tight_layout()
plt.savefig("output/problem2/sample_hidden_neuron_weights2.png")
plt.show()

# read in test set and test set output
test_set = []
with open("output/test_set.txt") as f, open("output/problem2/output_trained_test_set.txt") as f2:
    lines = f.readlines()
    for line in lines:
        test_set.append([float(pixel) for pixel in line.strip().split("\t")])
    test_set_output = []
    lines = f2.readlines()

# plot 8 random samples from the test set and their reconstructions
test_set_output = []
for line in lines:
    test_set_output.append([float(pixel)
                           for pixel in line.strip().split("\t")])

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
axes = axes.flatten()

random_indices = sample(range(len(test_set)), 8)

for i, idx in enumerate(random_indices):
    # Plot original image
    original_image = np.array(test_set[idx]).reshape((28, 28))
    axes[i].imshow(original_image, cmap='gray')
    axes[i].axis('off')
    # Plot reconstructed image
    reconstructed_image = np.array(test_set_output[idx]).reshape((28, 28))
    axes[i + 8].imshow(reconstructed_image, cmap='gray')
    axes[i + 8].axis('off')

fig.suptitle("Random Samples from Test Set and Their Reconstructions")
plt.tight_layout()
plt.savefig("output/problem2/random_samples_reconstructions.png")
plt.show()
