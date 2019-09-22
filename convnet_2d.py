import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression


# TODO: Define/tune the layers
    # TODO: Check padding
    # TODO: Check regularization
    # TODO: Check activation function
    # TODO: Check learning rate
    # TODO: Check flatten

# CONFIGURATION 1  - sentdex tutorial inspired
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, output_size, activation='softmax')
    # Regression layer is used for backpropagation
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir="log")

    return model

'''
#CONFIGURATION 2 - 1st own model
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    convnet = conv_2d(convnet, nb_filter=10, filter_size=2, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=360, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=180, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model
