from utills import generate_data_sets, simulate_reconstruction_learning
from os import makedirs
import matplotlib.pyplot as plt
from random import sample
import numpy as np


makedirs("output", exist_ok=True)
makedirs("output/problem2", exist_ok=True)

generate_data_sets(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

learning_rate = 0.01
epochs = 10
layers = [125, 784]

simulate_reconstruction_learning(layers, learning_rate, epochs)

mean_reconstruction_errors_training = []
mean_reconstruction_errors_test = []
standard_deviations_test = []
with open("output/problem2/mean_reconstruction_errors_training.txt") as f1, open("output/problem2/reconstruction_errors_test_set.txt") as f2, open("output/problem2/standard_deviations_test_set.txt") as f3:
    mean_reconstruction_errors_training = [
        float(err) for err in f1.readlines()]
    mean_reconstruction_errors_test = [float(err) for err in f2.readlines()]
    standard_deviations_test = [float(err) for err in f3.readlines()]

mean_reconstruction_error_training = mean_reconstruction_errors_training[-1]
mean_reconstruction_error_test = 0
for err in mean_reconstruction_errors_test:
    mean_reconstruction_error_test += err * 400
mean_reconstruction_error_test /= 4000

# plot as 2 bards, side by side
fig, ax = plt.subplots()
bar_width = 0.35
index = [1, 2]
rects1 = ax.bar(index[0], mean_reconstruction_error_training,
                bar_width, label='Training')
rects2 = ax.bar(index[1], mean_reconstruction_error_test,
                bar_width, label='Test')

ax.set_xlabel('Data Set')
ax.set_ylabel('Mean Reconstruction Error')
ax.set_title('Mean Reconstruction Error for Training and Test Data Sets')
ax.set_xticks(index)
ax.set_xticklabels(['Training', 'Test'])
ax.legend()

fig.tight_layout()
plt.savefig("output/problem2/mean_reconstruction_error.png")
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

# plot the time series of mean reconstruction errors in normal range for training set
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

# normalize the weights between 0 and 255, reshape them to 28x28 and plot as 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(sample_hidden_neuron_weights1):
        arr = np.array(sample_hidden_neuron_weights1[i])
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255
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

# normalize the weights between 0 and 255, reshape them to 28x28 and plot as 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(sample_hidden_neuron_weights2):
        arr = np.array(sample_hidden_neuron_weights2[i])
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255
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

test_set = []
test_set_output = []
with open("output/test_set.txt", "r") as f1, open("output/problem2/output_trained_test_set.txt", "r") as f2:
    test_set = f1.readlines()
    test_set = [list(map(float, line.strip().split("\t")))
                for line in test_set]
    test_set_output = f2.readlines()
    test_set_output = [list(map(float, line.strip().split("\t"))
                            ) for line in test_set_output]

# plot 8 random samples from the test set and their reconstructions
sample_indices = sample(range(1000), 8)
sample_test_set = [test_set[i] for i in sample_indices]
sample_test_set_output = [test_set_output[i] for i in sample_indices]

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
axes = axes.flatten()

for i in range(8):
    # Plot original test set image
    arr = np.array(sample_test_set[i])
    pixel_matrix = arr.reshape((28, 28))
    pixel_matrix = np.rot90(pixel_matrix, k=-1)
    pixel_matrix = np.fliplr(pixel_matrix)
    axes[i].imshow(pixel_matrix, cmap='gray')
    axes[i].axis('off')

    # Plot reconstructed test set image
    arr = np.array(sample_test_set_output[i])
    pixel_matrix = arr.reshape((28, 28))
    pixel_matrix = np.rot90(pixel_matrix, k=-1)
    pixel_matrix = np.fliplr(pixel_matrix)
    axes[i + 8].imshow(pixel_matrix, cmap='gray')
    axes[i + 8].axis('off')

plt.tight_layout()
plt.savefig("output/problem2/sample_test_set_comparison.png")
plt.show()
