import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
#read data
spectrum_path="/home/gurkan/Desktop/casedata/data/normal/"
data=np.load(spectrum_path+'fft_all_normal.npy')
data=data
print(np.size(data,axis=0))
print(np.size(data,axis=1))
#spectrum_path="/home/gurkan/Desktop/sparsityawarelearning/KSVD/spectrum/"

#dividing set into training and test sets
data_tr=np.transpose(data)
training_idx=np.random.rand(np.size(data_tr,0))<0.8
training_set=data_tr[training_idx,:]
test_set=data_tr[~training_idx,:]
print(np.size(training_set,axis=1))
print(np.size(training_set,axis=0))

#parameters
learning_rate=0.01#rate of updates for gradient descent
momentum=0.01#rate of upgrades from previous gradient
num_of_steps=5000#number of training steps
batch_size=10#number of


#display tools
display_step=1000
examples_to_show=1

#model parameters
num_of_layers=3
num_hidden_1=256
num_hidden_2=84
num_hidden_3=42
num_input=1025#input is fft soectrum of signals length 1024

#input
X=tf.placeholder(tf.float32,[None,num_input])

#weights
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1]),dtype=tf.float32),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]),dtype=tf.float32),
    #'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3]),dtype=tf.float32),
    #'decoder_h3': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2]),dtype=tf.float32),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1]),dtype=tf.float32),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input]),dtype=tf.float32),
}
#biases
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),dtype=tf.float32),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2]),dtype=tf.float32),
    #    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3]),dtype=tf.float32),
    #'decoder_b3': tf.Variable(tf.random_normal([num_hidden_2]),dtype=tf.float32),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1]),dtype=tf.float32),
    'decoder_b1': tf.Variable(tf.random_normal([num_input]),dtype=tf.float32),
}

#model
#stacked AE 1st layer
def encoder_1(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    return layer_1
def decoder_1(x):
    layer_1=(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    return layer_1
#stacked AE 2nd layer
def encoder_2(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h2']),biases['encoder_b2']))
    return layer_1
def decoder_2(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h2']),biases['decoder_b2']))
    return layer_1
#pretraining 3st layer
#def encoder_1(x):
#    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
#def decoder_1(x):
#    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))



#constructing stacked AE 1st layer
encoder_op1=encoder_1(X)
decoder_op1=decoder_1(encoder_op1)
#prediction
y_pred=decoder_op1
#target values
y_true=X
#defining loss and optimizer
loss=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#initialization
init=tf.global_variables_initializer()
#pretraining
sess=tf.Session()
#initialize variables
sess.run(init)
for i in range(1,num_of_steps+1):
    batch_x=training_set[np.random.randint(training_set.shape[0],size=batch_size),:]
    #run optimization
    _,l=sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))

batch_x=test_set
error=tf.reduce_sum(np.absolute(test_set-decoder_op1),1)
g=sess.run(error,feed_dict={X:batch_x})
batch_x=training_set[np.random.randint(training_set.shape[0],size=np.size(test_set,0)),:]
error=tf.reduce_sum(np.absolute(test_set-decoder_op1),1)
h=sess.run(error,feed_dict={X:batch_x})
'''
print(np.mean(g), np.std(g), np.max(g))
plt.figure(1)
test_error_figure,=plt.plot(g,label='Test Error')
plt.legend(handles=[test_error_figure])
print(np.mean(h), np.std(h), np.max(h))
plt.figure(2)
trainig_error_figure,=plt.plot(h,label="Training error")
plt.legend(handles=[trainig_error_figure])
plt.show()
'''




#constructing autoencoder 2nd layer
layer_2_input=sess.run(encoder_op1,feed_dict={X:data_tr})
print(layer_2_input.shape[0])
print(layer_2_input.shape[1])

training_idx2=np.random.randint(layer_2_input.shape[0],size=int(num_hidden_1*0.75))
training_set2=layer_2_input[training_idx2,:]
test_set2=layer_2_input[~training_idx2,:]
X2=tf.placeholder(tf.float32,[None,num_hidden_1])
encoder_op2=encoder_2(X2)
decoder_op2=decoder_2(encoder_op2)
#prediction
y_pred=decoder_op2
#target values
y_true=X2
#defining loss and optimizer
loss=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate*10).minimize(loss)

#pretraining
for i in range(1,3*num_of_steps+1):
    batch_x=training_set2[np.random.randint(training_set2.shape[0],size=batch_size),:]
    #run optimization
    _,l=sess.run([optimizer, loss], feed_dict={X2: batch_x})
    # Display logs per step
    if i % (5*display_step) == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))



#encoder
def encoder(x):
    #sigmoid activation function
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

#decoder
def decoder(x):
    #sigmoid activation function
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h2']),biases['decoder_b2']))
    layer_2=(tf.add(tf.matmul(layer_1,weights['decoder_h1']),biases['decoder_b1']))
    return layer_2




#constructing full autoencoder

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

#prediction
y_pred=decoder_op
#target values
y_true=X

#defining loss and optimizer
loss=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
for i in range(1,num_of_steps+1):
    batch_x=training_set[np.random.randint(training_set.shape[0],size=batch_size),:]
    #run optimization
    _,l=sess.run([optimizer, loss], feed_dict={X: batch_x})
    # Display logs per step
    if i % (display_step) == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))



#testing

batch_x=test_set[np.random.randint(test_set.shape[0],size=1),:]
print(batch_x.shape[0])
print(batch_x.shape[1])
error=abs(batch_x-decoder_op)
h,g=sess.run([decoder_op,error],feed_dict={X:batch_x})

'''print(np.size(g,axis=1))
print(np.size(batch_x,axis=1))
print(np.size(g,axis=0))
print(np.size(batch_x,axis=0))
print("Original Spectrum")
plt.figure(1)
plt.plot(np.transpose(batch_x))
print("Reconstructed Spectrum")'''
plt.figure(1)
plt.plot(np.transpose(g))
plt.figure(2)
plt.plot(np.transpose(batch_x))
plt.figure(3)
plt.plot(np.transpose(h))
plt.show()
