from utills import generate_data_sets, simulate_reconstruction_learning
from os import makedirs


makedirs("output", exist_ok=True)
makedirs("output/problem2", exist_ok=True)

generate_data_sets(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

learning_rate = 0.01
epochs = 5
layers = [125, 784]

simulate_reconstruction_learning(layers, learning_rate, epochs)
