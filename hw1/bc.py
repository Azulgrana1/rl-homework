import numpy as np
import tensorflow as tf
import pickle as pkl
import gym
import matplotlib.pyplot as plt
#haha hwerqr
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

''' Train A Behavior Cloning Policy '''
envname = 'Ant-v1'
num_iteration = 10000
num_rollouts = 100
max_steps = 100
h_dim = [16,16,16,16,16]
# load expert data
expert_data = pkl.load(open(envname+'_expert_data.pkl', 'rb'))
expert_obs = expert_data['observations']
expert_act = expert_data['actions']
expert_act = expert_act.reshape([expert_act.shape[0], -1])

# define the network
input_dim = expert_obs.shape[1]
output_dim = expert_act.shape[1]
print input_dim, output_dim

x = tf.placeholder(tf.float32, shape=[None, input_dim])
y_ = tf.placeholder(tf.float32, shape=[None, output_dim])

# Layer1
W1 = weight_variable([input_dim, h_dim[0]])
b1 = bias_variable([h_dim[0]])
h1 = tf.matmul(x, W1) + b1
h11 = tf.nn.relu(h1)

# Layer2
W2 = weight_variable([h_dim[0], h_dim[1]])
b2 = bias_variable([h_dim[1]])
h2 = tf.matmul(h11, W2) + b2
h22 = tf.nn.relu(h2)

# Layer 3
W3 = weight_variable([h_dim[1], h_dim[2]])
b3 = bias_variable([h_dim[2]])
h3 = tf.matmul(h22, W3) + b3
h33 = tf.nn.relu(h3)

# Layer 4
W4 = weight_variable([h_dim[2], h_dim[3]])
b4 = bias_variable([h_dim[3]])
h4 = tf.matmul(h33, W4) + b4
h44 = tf.nn.relu(h4)

# Layer 5
W5 = weight_variable([h_dim[3], h_dim[4]])
b5 = bias_variable([h_dim[4]])
h5 = tf.matmul(h44, W5) + b5
h55 = tf.nn.relu(h5)

# Layer 6
W6 = weight_variable([h_dim[4], output_dim])
b6 = bias_variable([output_dim])
y_out = tf.matmul(h55, W6) + b6

# Loss
loss = tf.reduce_mean(tf.square(tf.subtract(y_out, y_)))
regu = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)#+ #\
	#tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(b4)
loss = loss + 0.4 * regu
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

''' Train the Network '''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.InteractiveSession(config=config)
sess.run(tf.initialize_all_variables())
loss_list = []
j = 0
for i in range(num_iteration):
	coord = j%expert_obs.shape[0]
	coord2 = (j+500)%expert_obs.shape[0]
	if coord > coord2:
		temp = coord
		coord = coord2
		coord2 = temp
	# print coord, coord2
	if i% 200 == 0:
		print 'training'
		# print coord, coord2
	xx = expert_obs[coord:coord2, :]
	yy = expert_act[coord:coord2, :]
	train_step.run(feed_dict={x:xx, y_: yy})
	loss_list.append(loss.eval(feed_dict={x:expert_obs, y_:expert_act}))
	j += 500
plt.plot(loss_list)
plt.show()

''' Evaluate '''

env = gym.make(envname)
returns = []
for i in range(num_rollouts):
	obs = env.reset()
	done = False
	totalr = 0.
	steps = 0
	while not done:
		action = y_out.eval(feed_dict={x:obs[None,:]})
		action = action.reshape((output_dim,))
		obs, r, done, _ = env.step(action)
		totalr += r
		steps += 1
		env.render()
		if steps >= max_steps:
			break
	returns.append(totalr)
print('mean of return is', np.mean(returns))
print('std of return is', np.std(returns))

