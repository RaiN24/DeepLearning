import tensorflow as tf
import numpy
from skimage import io,transform,data
from numpy import dtype, float32, float64

INPUT_NODE = 784
OUTPUT_NODE = 62
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

# chÊÇ×Ö·û
def getNum(ch):
    c=ch[0]
    result=numpy.zeros(62,dtype=float64)
    if ord(c)>=ord('0') and ord(c)<=ord('9'):
        result[ord(c)-ord('0')]=1.0
    elif ord(c)>=ord('a') and ord(c)<=ord('z'):
        result[ord(c)-ord('a')+10]=1.0
    elif ord(c)>=ord('A') and ord(c)<=ord('Z'):
        result[ord(c)-ord('A')+36]=1.0
    return result

def readPic(path):
    old_img=io.imread(path)
    img=transform.resize(old_img,(28,28))
    result=numpy.zeros(784,dtype=float32)
    for i in range(28):
        for j in range(28):
            index=i*28+j
            result[index]=img[i][j]
    return result

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2