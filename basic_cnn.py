from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
from scipy import misc
import glob
import PIL
from PIL import Image,ImageOps

# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_input.py

# input our images into numpy arrays
def png_to_np_array():
    
    """
        
        filename_list = glob.glob("/good_images/*.png")
        
        for i in range(list(filename_list)):
        image_list = misc.imread(filename_list[i])
        
        filename_list = glob.glob("/bad_images/*.png")
        
        for i in range(list(filename_list)):
        image_list = image = misc.imread(filename_list[i])
        """
    
    # Read the image and convert it into numpy array of type uint8
    print(' Converting Image from 500*500 to 512*512')
    #image = misc.imread('images/8.png')
    PIL.ImageOps.expand(Image.open('images/8.png'), border=12, fill='black').save('8_border.png')
    
    image = misc.imread('8_border.png')
    # have a look at the image 3D numpy with values between 0 and 255
    # print(image[1, 1:10])
    
    # Get the dimensions of image for conversion
    shape_vec = image.shape
    print(' Image Shape : ',shape_vec)
    reshaped_vec_size = 1
    for i in range(len(shape_vec)):
        reshaped_vec_size *= shape_vec[i]
    print('reshaped vec size :',reshaped_vec_size)

# Convert it into a vector from [w,h,channels] -> w*h*channels
image = image.reshape((1,reshaped_vec_size))
    
    # Now convert the type from uint8 into float32
    image = image.astype(np.float32)
    
    # Normalize it to be between 0.0 and 1.0
    image = image*1.0/255.0
    
    # cut the image for initial 28*28 pixels for now
    #cut_image = np.ones((1, 784*3), dtype=np.float32)
    #for i in range(784*3):
    #    cut_image[0][i] = image[0][i];
    #print(cut_image)
    
    # Check the shape and the type of the image
    #print(cut_image.shape)
    #print(cut_image.dtype)
    
    # label of type float32
    label = np.ones((1, 2), dtype=np.float32)
    label[0,0] = 1.0
    #label = np.array([1, 0], dtype=np.float32)
    # Send it in a result to the caller
    result = []
    result.append(image)
    result.append(label)
    
    return result

#data = tf.image.decode_png("images/8.png",channels=3)
#grey_image = tf.image.rgb_to_grayscale(data)
#resized_image = grey_image
#resized_image = tf.image.resize_images(grey_image, [28, 28])
#print("resized_image =",resized_image)
#resized_image = tf.cast(resized_image, tf.float32)
#print(" resized_image = ",resized_image)
#data = np.multiply(resized_image, 1.0 / 255.0)
#print("normalized images =",data.dtype)
#print(data)

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 1
display_step = 10

# Width = Heigth of the input image
N = 500

# Network Parameters
n_input = N*N*3
n_classes = 2
dropout = 0.00 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

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
    print("Conv_1 : ",conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print("MaxPool_1 : ", conv1)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print("Conv_2 : ", conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print("MaxPool_2 : ", conv2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
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
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([128*128*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
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
    while step < 10:
        #while 1==0:
        result = png_to_np_array()
        batch_x = result[0]
        batch_y = result[1]
        # print(batch_x.shape,batch_y.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        print_step(step)
        if step % 2 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

#Calculate accuracy for the given test image
list = png_to_np_array()
    test_image = list[0]
    test_label = list[1]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_image,y: test_label}))ct={x: test_image,
                   y: test_label}))