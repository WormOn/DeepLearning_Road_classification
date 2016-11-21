from__future__ import print_function

import tensorflow as tf
import os
import numpy as np
from scipy import misc
import glob
import PIL
from PIL import Image
import requests
from StringIO import StringIO

# CNN related parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 30
display_step = 10
N = 500
n_input = N * N * 3
n_classes = 2
dropout = 0.00  # Dropout, probability to keep unit

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def png_to_np_array(batch_size):
    # bad images_first
    bad_base_url = "https://storage.googleapis.com/ini_practicum_drones/bad_roads/"
    good_base_url = "https://storage.googleapis.com/ini_practicum_drones/good_roads/"
    image_list = []
    label_list = []

    # Append bad images first and then good images
    image_list, label_list = get_images_from_url(image_list, label_list, batch_size, bad_base_url, 0)
    # image_list,label_list = get_images_from_url(image_list,label_list,batch_size,good_base_url,1)

    # We are returning 2*num_images along with the labels to the network
    return [image_list, label_list]


def get_images_from_url(image_list, label_list, num_images, url, good):
    for i in range(num_images):
        image_url = url + str(100 + i) + ".png"

        # print('image_url : ',image_url)
        response = requests.get(image_url)
        image = np.array(Image.open(StringIO(response.content)))
        image = np.array(Image.open(StringIO(response.content)))

        # Get the dimensions of image for conversion
        shape_vec = image.shape
        reshaped_vec_size = 1
        for i in range(len(shape_vec)):
            reshaped_vec_size *= shape_vec[i]

        # Convert it into a vector from [w,h,channels] -> w*h*channels
        image = image.reshape((1, reshaped_vec_size))

        # Now convert the type from uint8 into float32
        image = image.astype(np.float32)

        # Normalize it to be between 0.0 and 1.0
        image = image * 1.0 / 255.0

        image_list.append(image[0, :])
        # print('image shape',image[0,:].shape)

        # label of type float32
        label = np.ones((1, 2), dtype=np.float32)
        label[0, good] = 0.0
        label_list.append(label[0, :])

    return [image_list, label_list]

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases):

    # Reshape input picture
    print('Executing conv_net')
    x = tf.reshape(x, shape=[-1, N, N, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print("Conv_1 : ", conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print("MaxPool_1 : ", conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print("Conv_2 : ", conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print("MaxPool_2 : ", conv2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print("Conv_3 : ", conv3)
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    print("MaxPool_3 : ", conv3)

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    print("Conv_4 : ", conv4)
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)
    print("MaxPool_4 : ", conv4)

    # Fully connected layer
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print("fc1 : ",fc1)
    fc1 = tf.nn.relu(fc1)
    print('Done with fc1 execution')

    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print("Out : ",out)
    return out

def print_step(step):
    print("step = ",step)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 5X5 conv,64 input
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),

    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32*32*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    while step < 100:
        result = png_to_np_array(batch_size)
        batch_x = np.array(result[0])
        batch_y = np.array(result[1])
        #print(batch_x.shape,batch_y.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        print_step(step)
        if step % 1 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for the given test image
    result = png_to_np_array(batch_size)
    test_image = np.array(result[0])
    test_label = np.array(result[1])

    print("Final Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_image, y: test_label}))
