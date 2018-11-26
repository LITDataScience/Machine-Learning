################### How to create and use Tensors using Tensorflow ###################


import tensorflow as tf
from tensorflow.python.framework import ops

## to clear old session nodes and reset the graph to default state...
ops.reset_default_graph()

## For creating Tensors, first we've to initialize a session.
sess = tf.Session()

## Let's create a tensor of zeros to use at multiple places...
t = tf.zeros([1,20])

## Creating a variable. This will maintain state accross all the calls.
var = tf.Variable(t, name = 'var1')

## Dimentions in ML are the number of attributes. We'll create row/col dimentions, i.e. two variables.

row = 2
col = 3

## Tensors of zeros and ones
t0 = tf.zeros([row, col])
t1 = tf.ones([row, col])

## Let's create some variables based on the row/col using 0s and 1s
var_zero = tf.Variable(t0)
var_one = tf.Variable(t1)

## Initializing
sess.run(var_zero.initializer)
sess.run(var_one.initializer)

## Create zerolike and oneslike tensor.
similar_to_zero = tf.Variable(tf.zeros_like(var_zero))
similar_to_one = tf.Variable(tf.ones_like(var_zero))

## Initializing
sess.run(similar_to_zero.initializer)
sess.run(similar_to_one.initializer)

## Adding a constant to fill the shape
fillVar = tf.Variable(tf.fill[row, col], -1)

## Creating a constant Variable
constantVar = tf.Variable(tf.constant(-1, shape = [row, col]))

## Generating seq using linspace
linVar = tf.Variable(tf.linspace(start=0, stop=1.0, num=3))
seqVar = tf.Variable(tf.range(start=6, limit=15, delta=3))

## Creating some Random Normal Numbers.
tf.random_normal([row, col], mean=0.0, stddev=1.0)

# Initialize graph writer:
writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# Initialize operation
init_op = tf.global_variables_initializer()

# Run initialization of variable
sess.run(init_op)