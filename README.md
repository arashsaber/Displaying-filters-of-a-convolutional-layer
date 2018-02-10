# Displaying Filters of a convolutional layer
We want to display the filters of convolutional layer as tiles beside each other in one photo. Such photos are quite helpful in understanding CNN's.

We assume the NN model is developed by tflearn, which simplifies accessing the weights of each layer easy. The complete code can be found in

We use the following function for the task:

    def filter_displayer(model, layer, padding=1, normalize=True):
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
        if nc*(nr-1) > n:
            nr-=1
        # Assuming that the filters are square
        filter_size = filters.shape[0]
        # Size of the output image with padding
        img_w = nc * (filter_size + padding) - padding
        img_h = nr * (filter_size + padding) - padding
        # Initialize the output image
        filter_img = np.zeros((img_h, img_w))
    
        if normalize:
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

The output looks like this:

<img src="https://github.com/arashsaber/Displaying-filters-of-a-convolutional-layer/blob/master/sample_output.png" width="400">

Note that, in case you want to visualize the filters of a convolutional layer and for example want to check the sparsity, make sure to use the absolute values. That is, change the line that reading the weights to

    filters = abs(model.get_weights(variable))
    
The code with an example is available [here](https://github.com/arashsaber/Displaying-filters-of-a-convolutional-layer/blob/master/displayer.py).

        
  
