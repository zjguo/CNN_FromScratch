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

# Helper function to intialize bias
def new_bias(len):
    return np.ones(len)

# Helper function normalize input images
def normalize_input(input):

    tiny = 1e-8
    mean = np.mean(input)
    std = np.std(input)
    out = (input - mean) / (std + tiny)

    return out, mean, std


# Normalize Training Images
normalized_input, mean_Training_Imgs, std_Training_Imgs = normalize_input(Training_Images_Array)

# Helper function to convolve an image
def convolve(input,     # Input will be a 4D tensor: [img_number, rows, cols, num_filters]
             filter,    # Expected shape is [filter_size, filter_size, num_input_channels, num_filters]
             bias,
             padding='SAME'):

    mask_buffer = int((filter.shape[0]+1)/2 - 1)  # Start convolution at top left of input

    # Pad image with 0s
    npad = ((0, 0), (mask_buffer, mask_buffer+2), (mask_buffer, mask_buffer+2), (0, 0)) # +2 to fix indexing issues
    if padding is 'SAME':
        pad_in = np.pad(input, pad_width=npad, mode='constant', constant_values=0)

    # Initialize output array to have form [num_img, rows(padded), cols(padded), n_output_filter]
    out_array = np.zeros((pad_in.shape[0],pad_in.shape[1], pad_in.shape[2],filter.shape[3]))  # Initialize processed array

    # Iterate over all rows and cols in the input and matmul the perceptive field with the filter
    for i in range(mask_buffer, input.shape[1]+mask_buffer):
        for j in range(mask_buffer, input.shape[2]+mask_buffer):
            per_field = pad_in[:, i-mask_buffer:i+mask_buffer+1, j-mask_buffer:j+mask_buffer+1,:]   # Current perceptive field
            out_array[:, i, j, :] = np.einsum('ijkl,jklm -> im', per_field, filter) + bias  # Element-wise multiply PF and F + bias

    # Trim away the zero padding from processed_input_array
    out_array = out_array[:,mask_buffer:pad_in.shape[1]-mask_buffer-2,mask_buffer:pad_in.shape[2]-mask_buffer-2,:]

    return  out_array   # out_array: [img_num, rows, cols, num_filters]


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
                   weights,
                   bias,
                   max_pool=True):  # 2x2 max-pool

    # Convolve the input with filter
    out = convolve(input, weights, bias, padding='SAME')

    # Max pooling
    if max_pool is True:
        out = maxPool_2x2(out)

    # ReLu
    out = relu(out)

    return out


# Helper function to flatten the layer for input into fully connected network
def flatten(input):

    # Reshape the input to [num_images, num_features]
    shape = input.shape
    out = input.reshape((shape[0],-1), order='F')
    num_features = out.shape[1]


    return out, num_features


# Helper function to make new FC layer
def new_fc_layer(input, weights, bias, include_nonlin = True):

    input ,num_features = flatten(input)

    out = np.matmul(input, weights) + bias

    # Use ReLu
    if include_nonlin:
        out = relu(out)

    return out

# Helper function to find softmax
def softmax(input):

    input_exp = np.exp(input)
    exp_sum = np.sum(input_exp,axis=1)
    exp_sum = exp_sum.reshape((exp_sum.shape[0],1))

    out = input_exp/exp_sum

    return out


# Helper function to find cross entropy
def cross_entropy_cost(input, true_label_array):

    input = softmax(input)
    # Use true label array to pick out prediction corresponding to correct class
    out = - np.log(input[[np.arange(input.shape[0])], [true_label_array[0:input.shape[0]]]])
    out = np.mean(out)
    return out

# Layers information
L1_filtersize = 5
L1_out_channels = 3

L2_filtersize = 3
L2_out_channels = 3

FC1_neurons  = 128
n_classes = 10


# Helper function to optimize in batches
def run( iterations, grad_check=False):
    batchsize = 64

    # Initiate all weights and biases
    W1 = new_weights([L1_filtersize, L1_filtersize, 1, L1_out_channels])
    b1 = new_bias(L1_out_channels)
    W2 = new_weights([L2_filtersize, L2_filtersize, L1_out_channels, L2_out_channels])
    b2 = new_bias(L2_out_channels)
    W3 = new_weights([7*7*L2_out_channels, FC1_neurons])
    b3 = new_bias(FC1_neurons)
    W4 = new_weights([FC1_neurons, n_classes])
    b4 = new_bias(n_classes)

    for i in range(iterations):
        rand_index = np.random.randint(nTrainingImages, size = batchsize)
        batch_img_arr = normalized_input[rand_index,:,:,:] # Obtain 64 random images from Training_Images_Array
        learning_rate = 0.001

        # First convolutional layer
        L1 = new_conv_layer(batch_img_arr, W1, b1, max_pool=True)

        # Second convolutional layer
        L2 = new_conv_layer(L1, W2, b2, max_pool=True)

        # Fully Connected layer
        FC3 = new_fc_layer(L2, W3, b3)

        # Output Layer
        FC4 = new_fc_layer(FC3, W4, b4, include_nonlin=False)

        # Prepare a true label array from the random indexes
        true_labels = np.asarray(Training_Label_Array)
        true_labels = true_labels[rand_index]

        # Find cross entropy loss
        loss = cross_entropy_cost(FC4, true_labels)

        # Gradient for Final layer
        True_Label_Sparce_Array = sparce_label_arr[rand_index,:]
        Predict_sub_true = softmax(FC4) - True_Label_Sparce_Array
        dwb4 = Predict_sub_true
        dwb4 = np.sum(dwb4,axis=0)/batchsize
        dW4 = np.matmul(FC3.T,Predict_sub_true)
        dW4 = dW4/batchsize

        # Gradient for layer 4
        sensitivity_j = np.matmul(Predict_sub_true, W4.T)
        # Obtain FC3 before Relu
        z3 = new_fc_layer(L2, W3, b3, include_nonlin=False)
        z3_gte0 = z3 >= 0
        sensitivity_j = sensitivity_j * z3_gte0
        flatL2, numL2Feat = flatten(L2)
        dW3 = np.matmul(sensitivity_j.T, flatL2).T / batchsize
        dwb3 = np.sum(sensitivity_j, axis = 0)/ batchsize

        # Grad check is done in seperate layers for easy debugging
        if grad_check:
            # Gradient check for final layer
            tiny_num = 1e-7
            tempW4 = W4.copy()
            tempb4 = b4.copy()
            test_grad = np.ones(W4.shape)  # Dummy array same size as W4
            test_gradb = np.ones(b4.shape)  # Dummy array same size as W4

            for i in range(len(W4)):
                for j in range(len(W4[0])):
                    # Perturb W4 and obtain loss
                    tempW4[i][j] = W4[i][j] + tiny_num
                    FC4_test_upper = new_fc_layer(FC3, tempW4, b4, include_nonlin=False)
                    tempW4[i][j] = W4[i][j] - tiny_num
                    FC4_test_lower = new_fc_layer(FC3, tempW4, b4, include_nonlin=False)
                    tempW4[i][j] = W4[i][j]

                    loss_upper = cross_entropy_cost(FC4_test_upper, true_labels)
                    loss_lower = cross_entropy_cost(FC4_test_lower, true_labels)
                    test_grad[i][j] = (loss_upper-loss_lower)/(2*tiny_num)


                    # Perturb b4 and obtain loss
                    tempb4[j] = b4[j] + tiny_num
                    FC4_test_upperb = new_fc_layer(FC3, W4, tempb4, include_nonlin=False)
                    tempb4[j] = b4[j] - tiny_num
                    FC4_test_lowerb = new_fc_layer(FC3, W4, tempb4, include_nonlin=False)
                    tempb4[j] = b4[j]

                    loss_upper = cross_entropy_cost(FC4_test_upperb, true_labels)
                    loss_lower = cross_entropy_cost(FC4_test_lowerb, true_labels)
                    test_gradb[j] = (loss_upper-loss_lower)/(2*tiny_num)

            err = np.linalg.norm(test_grad - dW4) / (np.linalg.norm(test_grad) + np.linalg.norm(dW4))
            print('W4 error = {}'.format(err))
            err = np.linalg.norm(test_gradb - dwb4) / (np.linalg.norm(test_gradb) + np.linalg.norm(dwb4))
            print('dwb4 error = {}'.format(err))

            # Gradient check for first FC layer
            tempW3 = W3.copy()
            tempb3 = b3.copy()
            test_grad = np.ones(W3.shape)  # Dummy array same size as W3
            test_gradb = np.ones(b3.shape)  # Dummy array same size as W3

            for i in range(len(W3)):
                for j in range(len(W3[0])):
                    # Perturb W3 and obtain loss
                    tempW3[i][j] = W3[i][j] + tiny_num
                    FC3_test_upper = new_fc_layer(L2, tempW3, b3)
                    tempW3[i][j] = W3[i][j] - tiny_num
                    FC3_test_lower = new_fc_layer(L2, tempW3, b3)
                    tempW3[i][j] = W3[i][j]

                    # do forward pass
                    FC4_test_upper = new_fc_layer(FC3_test_upper, W4, b4, include_nonlin=False)
                    FC4_test_lower = new_fc_layer(FC3_test_lower, W4, b4, include_nonlin=False)

                    loss_upper = cross_entropy_cost(FC4_test_upper, true_labels)
                    loss_lower = cross_entropy_cost(FC4_test_lower, true_labels)
                    test_grad[i][j] = (loss_upper-loss_lower)/(2*tiny_num)


                    # Perturb b3 and obtain loss
                    tempb3[j] = b3[j] + tiny_num
                    FC3_test_upperb = new_fc_layer(L2, W3, tempb3)
                    tempb3[j] = b3[j] - tiny_num
                    FC3_test_lowerb = new_fc_layer(L2, W3, tempb3)
                    tempb3[j] = b3[j]

                    # do forward pass
                    FC4_test_upperb = new_fc_layer(FC3_test_upperb, W4, b4, include_nonlin=False)
                    FC4_test_lowerb = new_fc_layer(FC3_test_lowerb, W4, b4, include_nonlin=False)

                    loss_upper = cross_entropy_cost(FC4_test_upperb, true_labels)
                    loss_lower = cross_entropy_cost(FC4_test_lowerb, true_labels)
                    test_gradb[j] = (loss_upper-loss_lower)/(2*tiny_num)


            err = np.linalg.norm(test_grad - dW3) / (np.linalg.norm(test_grad) + np.linalg.norm(dW3))
            print('W3 error = {}'.format(err))
            err = np.linalg.norm(test_gradb - dwb3) / (np.linalg.norm(test_gradb) + np.linalg.norm(dwb3))
            print('dwb3 error = {}'.format(err))


        # Update FC layer weights
        #W4 = W4+learning_rate*dW4
        #b4 = np.add(b4,np.sum(learning_rate*dwb4, axis = 0))

        print(loss)


    return

run(1, grad_check=True)
