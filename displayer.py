"""
Displaying the filters
Arash Saber Tehrani
"""

import numpy as np
import tflearn
import matplotlib.pyplot as plt
#   ---------------------------------------
def filter_displayer(model, layer, padding=1):
    """
    The function displays the filters of layer
    :param model: tflearn obj, DNN model of tflearn
    :param layer: string or tflearn obj., the layer whose weights 
    we want to display
    :param padding: The number of pixels between each two filters
    :return: imshow the purput image
    """
    if isinstance(layer, str):
        vars = tflearn.get_layer_variables_by_name(layer)
        variable = vars[0]
    else:
        variable = layer.W
    filters = model.get_weights(variable)

    # n is the number of convolutions per filter
    n = filters.shape[2] * filters.shape[3]/2
    # Ensure the output image is rectangle with width twice as
    # big as height
    # and compute number of tiles per row (nc) and per column (nr)
    nr = int(np.ceil(np.sqrt(n)))
    nc = 2*nr
    # Assuming that the filters are square
    filter_size = filters.shape[0]
    # Size of the output image with padding
    img_w = nc * (filter_size + padding) - padding
    img_h = nr * (filter_size + padding) - padding
    # Initialize the output image
    filter_img = np.zeros((img_h, img_w))

    # Normalize image to 0-1
    fmin = filters.min()
    fmax = filters.max()
    filters = (filters - fmin) / (fmax - fmin)

    # Starting the tiles
    filter_x = 0
    filter_y = 0
    for r in range(filters.shape[3]):
        for c in range(filters.shape[2]):
            if filter_x == nc:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    filter_img[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = \
                        filters[i, j, c, r]
            filter_x += 1

    # Plot figure
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.imshow(filter_img, cmap='gray', interpolation='nearest')
    plt.show()
#   ---------------------------------------
if __name__ == '__main__':
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    LR = 1e-3
    height, width = 50, 50

    net = input_data(shape=[None, height, width, 1], name='input')
    #net = batch_normalization(net, scope='bN1')

    net = conv_2d(net, 32, 5, activation='relu', scope='conv1',
                  bias=True,
                  weights_init=tflearn.initializations.xavier(uniform=False),
                  bias_init=tflearn.initializations.xavier(uniform=False))
    net = max_pool_2d(net, 5, 2, name='maxP1')
    net = dropout(net, 0.8, name='drop1')


    net = conv_2d(net, 64, 5, activation='relu', scope='conv2',
                  bias=True,
                  weights_init=tflearn.initializations.xavier(uniform=False),
                  bias_init=tflearn.initializations.xavier(uniform=False))
    net = max_pool_2d(net, 5, name='maxP2')
    net = dropout(net, 0.8, name='drop2')

    #net = batch_normalization(net,scope='bN2')
    net = conv_2d(net, 128, 5, activation='relu', scope='conv3',
                  bias=True,
                  weights_init=tflearn.initializations.xavier(uniform=False),
                  bias_init=tflearn.initializations.xavier(uniform=False))
    net = max_pool_2d(net, 5, name='maxP3')
    net = dropout(net, 0.8, name='drop3')

    net = fully_connected(net, 1024, activation='relu', scope='fc1',
                  bias=True,
                  weights_init=tflearn.initializations.xavier(uniform=False),
                  bias_init=tflearn.initializations.xavier(uniform=False))
    net = dropout(net, 0.8, name='drop4')

    net = fully_connected(net, 2, activation='softmax', scope='output')
    net = regression(net, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(net, tensorboard_dir='logs', tensorboard_verbose=3)
    #   ---------------------------------------
    model_name = 'DogCatClassifier-{}.model'.format(LR)
    model.load(model_name+'.tflearn')

    filter_displayer(model, layer='conv1', padding=1)


