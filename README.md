# Displaying Filters of a layer
We want to display the filters of convolutional layer as tiles beside each other in one photo. Such photos are quite helpful in understanding CNN's.

We assume the NN model is developed by tflearn, which simplifies accessing the weights of each layer easy. The completer code can be found in

We use the following function for the task:

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
      n = filters.shape[2] * filters.shape[3]
      # Ensure the output image is square and compute number of tiles per row
      per_row = int(np.ceil(np.sqrt(n)))
      # Assuming that the filters are square
      filter_size = filters.shape[0]
      # Size of the output image with padding
      img_size = per_row * (filter_size + padding) - padding
      # Initialize the output image
      filter_img = np.zeros((img_size, img_size))

      # Normalize image to 0-1
      fmin = filters.min()
      fmax = filters.max()
      filters = (filters - fmin) / (fmax - fmin)

      # Starting the tiles
      filter_x = 0
      filter_y = 0
      for r in range(filters.shape[3]):
          for c in range(filters.shape[2]):
              if filter_x == per_row:
                  filter_y += 1
                  filter_x = 0
              for i in range(filter_size):
                  for j in range(filter_size):
                      filter_img[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = \
                          filters[i, j, c, r]
              filter_x += 1

      # Plot figure
      plt.figure(figsize=(10, 10))
      plt.axis('off')
      plt.imshow(filter_img, cmap='gray', interpolation='nearest')
      plt.show()
