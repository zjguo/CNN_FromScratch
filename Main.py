import numpy as np
import struct as st
import matplotlib.pyplot as plt
import time

# Open MNIST data files
train_images = open('train-images.idx3-ubyte', 'rb')
train_labels = open('train-labels.idx1-ubyte', 'rb')
test_images = open('t10k-images.idx3-ubyte', 'rb')
test_labels = open('t10k-labels.idx1-ubyte', 'rb')


# Helper function to extract images from idx3-ubyte file
def extract_images(file):
    with file as tr:
        tr.seek(0)  # Start read from 0 offset
        # Read magic number
        magic_number = st.unpack('>4B', tr.read(4))  # >4B, expected 4 bytes of high endian data

        # Read number of images
        # [0] is included so as the numbers do not remain as a 1 element list
        num_images = st.unpack('>I', tr.read(4))[0]  # Read number of images as integer

        # Read number of rows
        num_rows = st.unpack('>I', tr.read(4))[0]

        # Read number of columns
        num_cols = st.unpack('>I', tr.read(4))[0]

        # Read the image data
        # The last dimension changes the fastest [num_images, rows, cols]
        total_bytes = num_images * num_rows * num_cols
        images_array = 255 -\
            np.asarray(st.unpack('>' + 'B' * total_bytes, tr.read(total_bytes))).reshape((num_images, num_rows, num_cols))

        images_array = images_array[:,:,:, np.newaxis]   # Ensure images input is a 4D tensor
        return num_images, num_rows, num_cols, images_array


# Extract training images
nTrainingImages, nTRows, nTCols, Training_Images_Array = extract_images(train_images)

# Extract test images
nTestImages, nTestRows, nTestCols, Test_Images_Array = extract_images(test_images)


# Helper function to extract labels
def extract_labels(file):
    with file as f:
        f.seek(0)

        magic_number = st.unpack('>4B', f.read(4))
        num_labels = st.unpack('>I', f.read(4))[0]
        label_array = st.unpack('>'+'B'*num_labels, f.read(num_labels))

    return num_labels, label_array


# Extract training labels
nTrainingLabels, Training_Label_Array = extract_labels(train_labels)

# Extract test labels
nTestLabels, Test_Label_Array = extract_labels(test_labels)

# Create sparce label array
sparce_label_arr = np.zeros((nTrainingLabels, 10))
sparce_label_arr[np.arange(nTrainingLabels), Training_Label_Array] = 1

# Helper function to initialize weights
def new_weights(shape):
    return np.random.normal(scale=0.1, size=shape)  # Random numbers with std of 0.1 and mean of 0


# Helper function to do batch - normalization
def normalize(input):

    tiny = 1e-8
    mean = np.mean(input)
    std = np.std(input)
    out = (input - mean) / (std + tiny)

    return out


# Helper function to convolve an image
def convolve(input,     # Input will be a 4D tensor: [img_number, rows, cols, num_filters]
             filter,    # Expected shape is [filter_size, filter_size, num_input_channels, num_filters]
             padding='SAME'):

    # Set up filter
    new_filter = new_weights(filter)
    new_filter = np.reshape(new_filter, (filter[0]*filter[1]*filter[2],-1), order='F')   # Used to matmul the per_field

    # Set us bias term
    bias = 1

    mask_buffer = int((filter[0]+1)/2 - 1)  # Start convolution at top left of input

    # Pad image with 0s
    npad = ((0, 0), (mask_buffer, mask_buffer+2), (mask_buffer, mask_buffer+2), (0, 0)) # +2 to fix indexing issues
    if padding is 'SAME':
        pad_in = np.pad(input, pad_width=npad, mode='constant', constant_values=0)

    # Make the input in the form [num_images, fsize x fsize x numfilters, num_mask_blocks]
    processed_input_array = np.zeros((pad_in.shape[0],pad_in.shape[1],pad_in.shape[2],filter[3]))  # Initialize processed array

    # Iterate over all rows and cols in the input and matmul the perceptive field with the filter
    for i in range(mask_buffer, input.shape[1]+mask_buffer):
        for j in range(mask_buffer, input.shape[2]+mask_buffer):
            per_field = pad_in[:, i-mask_buffer:i+mask_buffer+1, j-mask_buffer:j+mask_buffer+1,:]   # Current perceptive field
            per_field_line = np.reshape(per_field, (pad_in.shape[0], -1), order='F')    # [num_images, fsizexfsizexnfilters]
            processed_input_array[:, i, j, :] = np.matmul(per_field_line, new_filter) + bias

    # Trim away the zero padding from processed_input_array
    processed_input_array = processed_input_array[:,mask_buffer:pad_in.shape[1]-mask_buffer-2,mask_buffer:pad_in.shape[2]-mask_buffer-2,:]

    return  processed_input_array, new_filter


# Helper function for max-pooling
def maxPool_2x2(input):

    # Set-up mask buffer for loop
    mask_buffer = 2

    # Set-up processed array
    in_shp = input.shape
    processed_array = np.zeros((in_shp[0],int(in_shp[1]/2), int(in_shp[2]/2),in_shp[3]))   # Pooled dims are divided by 2

    npad = ((0, 0), (0, 1), (0, 1), (0, 0))  # Fix indexing issues
    input = np.pad(input, pad_width=npad, mode='constant')

    # Iterate through every other row and col of the input and take the amax of each mask
    for i in range(0, input.shape[1]-mask_buffer+1,2):
        for j in range(0, input.shape[2]-mask_buffer+1,2):
            mask = input[:, i:i+mask_buffer+1, j:j+mask_buffer+1, :]    # The significant portion at this iteration
            max = np.amax(mask, axis=(1,2)) # Find the numerical max of the portion mask
            processed_array[:,int(i/2), int(j/2),:] = max   # Fill in the pre-initialized array

    return processed_array

# Helper function for ReLu
def relu(input):

    bool_array = np.asarray(np.greater(input,0)).astype(int)    # Cast boolean array to 0s and 1s for easy element wise mul
    out = input*bool_array

    return out


# Helper function for creating new convolutional layer
def new_conv_layer(input,   # Input will be a 4D tensor: [img_number, rows, cols, num_filters]
                   filter_size,
                   num_input_channels,
                   num_filters,
                   max_pool=True):  # 2x2 max-pool


    # Normalize input
    input = normalize(input)

    # Convolve the input with filter
    out, weights = convolve(input, [filter_size, filter_size, num_input_channels, num_filters], padding='SAME')

    # Max pooling
    if max_pool is True:
        out = maxPool_2x2(out)

    # Implement ReLu
    out = relu(out)

    return out, weights


# Helper function to flatten the layer for input into fully connected network
def flatten(input):

    # Reshape the input to [num_images, num_features]
    shape = input.shape
    out = input.reshape((shape[0],-1), order='F')
    num_features = out.shape[1]


    return out, num_features


# Helper function to make new FC layer
def new_fc_layer(input, weights, bias, num_outputs):

    input ,num_features = flatten(input)

    # Normalize input
    input = normalize(input)

    out = np.matmul(input, weights) + bias

    # Use ReLu
    out = relu(out)

    return out, weights, bias

# Helper function to find softmax
def softmax(input):

    input_exp = np.exp(input)
    exp_sum = np.sum(input_exp)

    out = input_exp/exp_sum

    return out


# Helper function to find cross entropy
def cross_entropy_cost(input):

    out = - np.log(input[[np.arange(input.shape[0])], [Training_Label_Array[0:input.shape[0]]]])
    out = np.mean(out)
    return out



# Helper function to run
def run( iterations):
    batchsize = 64
    W3 = new_weights([147, 10])
    b3 = np.ones((1,10))

    for i in range(iterations):
        rand_index = np.random.randint(nTrainingImages, size = batchsize)
        batch_img_arr = Training_Images_Array[rand_index,:,:,:] # Obtain 64 random images from Training_Images_Array
        learning_rate = 0.01

        # First convolutional layer
        L1, W1 = new_conv_layer(batch_img_arr, filter_size=3, num_input_channels=1, num_filters=3)

        # Third convolutional layer
        L2, W2 = new_conv_layer(L1, filter_size=3, num_input_channels=3, num_filters=3)

        # Fully Connected layer
        FC3, W3, b3 = new_fc_layer(L2, W3, b3, num_outputs=10)

        loss = cross_entropy_cost(softmax(FC3))

        # Gradient for FC layer
        dwFC = np.subtract(softmax(FC3),sparce_label_arr[rand_index,:])
        dwb3 = dwFC
        bgt0 = dwb3 > 0
        dwb3 = dwb3*dwb3
        flatL2, numL2features = flatten(L2)
        dwFC = np.matmul(np.transpose(dwFC), flatL2)
        bgt0 = dwFC > 0
        dwFC = bgt0*dwFC

        # Update FC layer weights
        W3 = np.add(W3,learning_rate*np.transpose(dwFC))
        b3 = np.add(b3,np.sum(learning_rate*dwb3, axis = 0))

    return loss

print(run(100))