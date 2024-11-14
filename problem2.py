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
with open("output/problem2/mean_reconstruction_errors_training.txt") as f1, open("output/problem2/mean_reconstruction_errors_test_set.txt") as f2, open("output/problem2/standard_deviations_test_set.txt") as f3:
    mean_reconstruction_errors_training = [
        float(err) for err in f1.readlines()]
    mean_reconstruction_errors_test = [float(err) for err in f2.readlines()]
    standard_deviations_test = [float(err) for err in f3.readlines()]


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
fig, axes = plt.subplots(8, 2, figsize=(8, 16))
axes = axes.flatten()

for i in range(8):
    index = sample(range(len(test_set)), 1)[0]
    test_image = np.array(test_set[index])
    test_image = test_image.reshape((28, 28))
    test_image = np.rot90(test_image, k=-1)
    test_image = np.fliplr(test_image)
    axes[2*i].imshow(test_image, cmap='gray')
    axes[2*i].axis('off')

    output_image = np.array([float(pixel)
                            for pixel in lines[index].strip().split("\t")])
    output_image = output_image.reshape((28, 28))
    output_image = np.rot90(output_image, k=-1)
    output_image = np.fliplr(output_image)
    axes[2*i+1].imshow(output_image, cmap='gray')
    axes[2*i+1].axis('off')

plt.tight_layout()
plt.savefig("output/problem2/sample_reconstructions.png")
plt.show()
