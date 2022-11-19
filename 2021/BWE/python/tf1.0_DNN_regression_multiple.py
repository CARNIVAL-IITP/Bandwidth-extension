import numpy as np
import h5py, os
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.python.ops.gen_math_ops import cos_eager_fallback

tf.compat.v1.disable_v2_behavior()

### load trainset & testset dir: path to directory + filename
dir_trainset = '' 
dir_testset = ''

### Set output dir
output_dir = ''

# Parameter setup
epoch = 500
batch_size = 1000
learning_rate = 0.001
display_step = 1

cost_list = []
best_epoch = 0

# Deep neural network layer & node setup
n_input = 75 #127 # input num features
n_hidden_1 = 128 # 1st layer num features
n_hidden_2 = 64 # 2nd layer num features
n_hidden_3 = 32 # 3rd layer num features
n_target = 16 #32 # output num features
input_dropout = 1
hidden_dropout = 0.8

# Initialize weight & bias
# Xavier initialization
weights = {
    'w1': tf.Variable(tf.random.normal([n_input, n_hidden_1]))/np.sqrt(n_input),
    'w2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2]))/np.sqrt(n_hidden_1),
    'w3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3]))/np.sqrt(n_hidden_2),
    'w4': tf.Variable(tf.random.normal([n_hidden_3, n_target]))/np.sqrt(n_hidden_3)
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'b3': tf.Variable(tf.random.normal([n_hidden_3])),
    'b4': tf.Variable(tf.random.normal([n_target]))
}

# Define variable
x = tf.compat.v1.placeholder("float", [None, n_input])
y = tf.compat.v1.placeholder("float", [None, n_target])
input_dropout = tf.compat.v1.placeholder("float")
hidden_dropout = tf.compat.v1.placeholder("float")

# Define model: Deep Neural Netowrk 
def deep_neural_network(x, weights, biases, input_dropout, hidden_dropout):
    x = tf.nn.dropout(x, rate=1 - (input_dropout))
    _layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1'])) #Hidden layer with ReLU activation function
    layer_1 = tf.nn.dropout(_layer_1, rate=1 - (hidden_dropout))
    _layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])) #Hidden layer with ReLU activation function
    layer_2 = tf.nn.dropout(_layer_2, rate=1 - (hidden_dropout))
    _layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])) #Hidden layer with ReLU activation function
    layer_3 = tf.nn.dropout(_layer_3, rate=1 - (hidden_dropout))
    return tf.matmul(layer_3, weights['w4']) + biases['b4']

pred = deep_neural_network(x, weights, biases, input_dropout, hidden_dropout)
error = y - pred

## penality
#penalty = np.sqrt(2)*tf.to_float(tf.less(y, pred))
#no_penalty = tf.to_float(tf.greater_equal(y, pred))

# Loss functions to choose
mse = losses.MeanSquaredError()
huber = losses.Huber()
mae = losses.MeanAbsoluteError()

# cost and optimization method
cost = losses.huber(y_true = y, y_pred = pred, delta=1) # Huber loss 
# cost = mae(y_true= y, y_pred= pred) # Mean Abolute Error loss
# cost = tf.reduce_mean(input_tensor=tf.pow(error, 2)) # Mean Squared Error loss
# cost = tf.reduce_mean(tf.pow(penalty*error+no_penalty*error, 2)) # Squared error with over-estimation penalty

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimization method

##########

# Initializing the variables
init = tf.compat.v1.initialize_all_variables()

# Launch the graph
with tf.compat.v1.Session() as sess:
	sess.run(init)
	
	# Compute average cost & accuracy
	for epoch in range(epoch):
		train_avg_cost = 0.
		test_avg_cost = 0.
		# Read trainset ('mat' format files)
		for train_file in range(40): 
			trainset = h5py.File(dir_trainset + str(train_file+1) + '.mat','r')
			train_input = trainset.get('input1_' + str(train_file+1))
			train_input = np.array(train_input)
			train_input = np.transpose(train_input)
			train_target = trainset.get('target_' + str(train_file+1))
			train_target = np.array(train_target)
			train_target = np.transpose(train_target)
			train_batch = int(train_input.shape[0]/batch_size)

			for j in range(train_batch): 
				train_batch_xs = train_input[batch_size*j:batch_size*(j+1)]
				train_batch_ys = train_target[batch_size*j:batch_size*(j+1)]
				# Compute average cost
				train_avg_cost += sess.run(cost, feed_dict={x: train_batch_xs, y: train_batch_ys, input_dropout: 1, hidden_dropout: 1})/1000        

		# Read testset ('mat' format files)
		for test_file in range(18):
			testset = h5py.File(dir_testset + str(test_file+1)+'.mat','r')
			test_input = testset.get('input1_'+str(test_file+1))
			test_input = np.array(test_input)
			test_input = np.transpose(test_input)
			test_target = testset.get('target_'+str(test_file+1))
			test_target = np.array(test_target)
			test_target = np.transpose(test_target)			
			test_batch = int(test_input.shape[0]/batch_size)
			
			for j in range(test_batch):
				test_batch_xs = test_input[batch_size*j:batch_size*(j+1)]
				test_batch_ys = test_target[batch_size*j:batch_size*(j+1)]
				# Compute average cost
				test_avg_cost += sess.run(cost, feed_dict={x: test_batch_xs,y: test_batch_ys, input_dropout: 1, hidden_dropout: 1})/1885

		# Display logs per epoch step
        
		if epoch % display_step == 0:
			print ("Epoch:", '%04d' % (epoch), "train_cost=", "{:.9f}".format(train_avg_cost), "test_cost=", "{:.9f}".format(test_avg_cost))
			if epoch >=0:
    				cost_list.append(test_avg_cost)
			
			if min(cost_list) >= cost_list[-1]:
    				best_epoch = epoch

			print("best epoch is:",best_epoch, " with test_avg_cost", min(cost_list) )
		
			# Create txt
			f = open(output_dir + str(epoch) + '_w1.txt', 'w') 	
			f = open(output_dir + str(epoch) + '_w2.txt', 'w')
			f = open(output_dir + str(epoch) + '_w3.txt', 'w')
			f = open(output_dir + str(epoch) + '_w4.txt', 'w')
			f = open(output_dir + str(epoch) + '_b1.txt', 'w')
			f = open(output_dir + str(epoch) + '_b2.txt', 'w')
			f = open(output_dir + str(epoch) + '_b3.txt', 'w')
			f = open(output_dir + str(epoch) + '_b4.txt', 'w')
			f.close()
            
			# write weight & bias parameter in text file
            # np.savetxt(output_dir).eval()
			np.savetxt(output_dir + str(epoch) + '_w1.txt', weights['w1'].eval())
			np.savetxt(output_dir + str(epoch) + '_w2.txt',weights['w2'].eval()) 
			np.savetxt(output_dir + str(epoch) + '_w3.txt',weights['w3'].eval())
			np.savetxt(output_dir + str(epoch) + '_w4.txt',weights['w4'].eval())
			np.savetxt(output_dir + str(epoch) + '_b1.txt',biases['b1'].eval())
			np.savetxt(output_dir + str(epoch) + '_b2.txt',biases['b2'].eval())
			np.savetxt(output_dir + str(epoch) + '_b3.txt',biases['b3'].eval())
			np.savetxt(output_dir + str(epoch) + '_b4.txt',biases['b4'].eval())
	
			# Training cycle
		for i in range(train_batch):
			train_batch_xs = train_input[batch_size*i:batch_size*(i+1)]
			train_batch_ys = train_target[batch_size*i:batch_size*(i+1)]
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: train_batch_xs, y: train_batch_ys, input_dropout: 1, hidden_dropout: 0.8})
			
	print ("Optimization Finished!")