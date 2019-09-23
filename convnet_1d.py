import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# TODO: Define/tune the layers
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):
    # TODO: Check padding
    # TODO: Check regularization
    # TODO: Check activation function
    # TODO: Check learning rate

    convnet = input_data(shape=[None, input_size], name='input')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, 3, 2, activation='relu')
    convnet = max_pool_1d(convnet, 3)

    convnet = conv_1d(convnet, 6, 2, activation='relu')
    convnet = max_pool_1d(convnet, 3)

    convnet = fully_connected(convnet, 9, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir="log")

    return model