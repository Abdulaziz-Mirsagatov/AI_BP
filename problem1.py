from utills import generate_data_sets, simulate_back_propogation
from os import makedirs

makedirs("output", exist_ok=True)

generate_data_sets(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

learning_rate = 0.01
epochs = 50
layers = [100, 10]

simulate_back_propogation(layers, learning_rate, epochs)
