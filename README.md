# Displaying Filters of a convolutional layer
We want to display the filters of convolutional layer as tiles beside each other in one photo. Such photos are quite helpful in understanding CNN's.

We assume the NN model is developed by tflearn, which simplifies accessing the weights of each layer easy. The complete code can be found in

To use, simply type:
    
    filter_displayer(model, layer='name_of_the_layer', padding=1)

The output looks like this:

<img src="https://github.com/arashsaber/Displaying-filters-of-a-convolutional-layer/blob/master/Figs/layer1_filters.png" width="400">

Recall that each convolutional kernel is a third degree tensor. SO to display all
weights, we stack them in a column and show them all together. For example,
the weights of a convolutional layer mapping 32 channels to 64 will be shown like this:

<img src="https://github.com/arashsaber/Displaying-filters-of-a-convolutional-layer/blob/master/Figs/layer2_filters.png" width="400">

Note that, in case you want to visualize the filters of a convolutional layer and for example want to check the sparsity, make sure to use the absolute values. That is, change the line that reading the weights to

    filters = abs(model.get_weights(variable))
    
The code with an example is available [here](https://github.com/arashsaber/Displaying-filters-of-a-convolutional-layer/blob/master/displayer.py).

        
  
