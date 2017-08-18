import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
import numpy
from numpy import dtype, float32, float64
import random

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="D:/Users/xiangtang/Desktop/model"
MODEL_NAME="mnist_model"


def train(mnist):

    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        23, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            if i % 10 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def randomPic(mnist,size):
    xa,ya=mnist
    x=numpy.zeros((size,784),dtype=float32)
    y=numpy.zeros((size,62),dtype=float64)
    for i in range(size):
        index=random.randint(0,599)
        x[i],y[i]=xa[index],ya[index]
    return x,y 
        
def main(argv=None):
    x=numpy.zeros((2300,784),dtype=float32)
    y=numpy.zeros((2300,62),dtype=float64)
    filepath='D:/Users/xiangtang/Desktop/pic/'
    pathDir =  os.listdir(filepath)
    index=0
    for path in pathDir:
        x[index]=mnist_inference.readPic(filepath+path)
        y[index]=mnist_inference.getNum(path[-5])
        index=index+1
    mnist=x,y
    train(mnist)
 
if __name__ == '__main__':
    tf.app.run()


