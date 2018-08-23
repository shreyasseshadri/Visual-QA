import tensorflow as tf
from tensorflow.contrib import rnn

time_steps=50
num_units=128
n_input=6
learning_rate=0.001
n_classes=300
# batch_size=97
x=tf.placeholder("float",[None,time_steps,n_input])
y=tf.placeholder([None,n_classes])

# out_weights=tf.Variable(tf.random_normal([num_units,n_classes],seed=123))
# out_bias=tf.Variable(tf.random_normal([n_classes],seed=123))
inp=tf.unstack(x,time_steps,1)
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,inp,dtype="float32")
lstm_output=outputs[-1]