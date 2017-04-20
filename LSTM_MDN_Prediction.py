from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
import sys

from MDNClass import MDN

# Choose 1 or 2 for different ydata graphs
PLOT = 1

# Make everything float32 
d_type = tf.float32

# Batch size for training
train_batch_size = 20

# Number of steps (RNN rollout) for training
train_num_steps = 8

# Dimension of LSTM input/output
hidden_size = 3

# should we do dropout? (1.0 = nope)
train_keep_prob = 0.95

# number of training epochs
num_epochs = 2500

# how often to print/plot
update_every = 250

# initial weight scaling
init_scale = 0.1

# Number of things in our cascade
steps_in_cascade = 3

# Input dimension
input_dimension = 3

######################################################################
# Helper function for below

def get_xy_data(n):
	u = np.arange(n)*0.4 + np.random.random()*10
	if PLOT == 1:
		x = u
		y = 8.0*(np.abs((0.125*u - np.floor(0.125*u)) - 0.5)-0.25)
	else:
		x = u + 3.0*np.sin(u)
		y = -2.0*np.cos(u)
                
        x -= x.min()
        y -= y.min()
	return x, y

######################################################################
# Get training data -- the format returned is xi, yi, 0 except for new
# "strokes" which are xi, yi, 1 every time the "pen is lifted".
    
def get_data(total_size):

    cur_count = 0
    all_data = []
    y0 = 0.0

    # add a number of "strokes" each with different y coords
    while cur_count < total_size:

        # get number of points in this stroke
        n = np.random.randint(50, 150)

        # get xy data for stroke
        x, y = get_xy_data(n)

        # reshape and add y offset
        x = x.reshape(-1, 1)-x.mean()
        y = y.reshape(-1, 1)+y0

        # make pen up/down feature
        z = np.zeros_like(y)
        # vector (n, 1)
        z[0] = 1

        # add random noise
        x += np.random.normal(size=x.shape, scale=0.05)
        y += np.random.normal(size=y.shape, scale=0.05)

        # append data
        all_data.append(np.hstack((x,y,z)))

        # update count & y offset
        cur_count += n
        y0 += 6.0

    # vstack all the data
    return np.vstack(tuple(all_data))

######################################################################
# Input class modeled off of PTB demo for feeding into
# train/valid/test models.

class Input(object):

    def __init__(self, posdata, config):

        batch_size = config.batch_size
        num_steps = config.num_steps

        # I think we need this name scope to make sure that each
        # condition (train/valid/test) has its own unique producer?
        with tf.name_scope('producer', [posdata, batch_size, num_steps]):

            # Convert original raw data to tensor
            raw_data = tf.convert_to_tensor(posdata, name='raw_data', dtype=d_type)

            # These will be tensorflow variables
            data_len = tf.size(raw_data)//3
            batch_len = data_len // batch_size
            epoch_size = (batch_len - 1) // num_steps

            # Prevent computation if epoch_size not positive
            assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")

            with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

            # Truncate our raw_data and reshape it into batches
            # This is just saying grab as much of it as we can to make a clean reshaping
            data = tf.reshape(raw_data[:batch_size*batch_len, :],
                              [batch_size, batch_len, 3])

            # i is a loop variable that indexes which batch we are on
            # within an epoch
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

            # each slice consists of num_steps*batch_size examples
            x = tf.slice(data, [0, i*num_steps, 0], [batch_size, num_steps, 3])
            y = tf.slice(data, [0, i*num_steps+1, 0], [batch_size, num_steps, 2])

        # Assign member variables
        self.x = x
        self.y = y
        self.epoch_size = ((len(posdata) // batch_size)-1) // num_steps


######################################################################
# Class of Cascading LSTMs 
class LSTMCascade(object):

    def __init__(self, config, model_input, is_train, external_targets=None):

        # Stash some variables from config
        hidden_size = config.hidden_size
        batch_size = config.batch_size
        num_steps = config.num_steps
        keep_prob = config.keep_prob

        # Scale factor so we can vary dataset size and see "average" loss
        self.loss_scale = batch_size * num_steps * model_input.epoch_size

        # Stash input
        self.model_input = model_input

        # we don't need to reshape the data! 
        self.lstm_input = model_input.x

        # this is going to be the final dimension 
        # interestingly enough this is always even
        final_high_dimension = input_dimension * steps_in_cascade * (steps_in_cascade+1) // 2

        # note: input dimension is equivalent to the hidden size of the LSTM cell
        hidden_size = input_dimension

        # this will hold all of our cells
        lstm_stack = []

        # this will hold all of our states as it goes
        state_stack = []

        # this will hold the initial states
        init_state_stack = []

        # This will reduce our final outputs to the appropriate lower dimension
        # Make weights to go from LSTM output size to 2D output
        w_output_to_y = tf.get_variable('weights_output_to_y', [final_high_dimension, 2],
                                        dtype=d_type)

        # we need to # LSTMS = # steps in cascade
        for i in range(steps_in_cascade):

            # Make an LSTM cell
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                hidden_size * (i+1), forget_bias=0.0,
                state_is_tuple=True)

            # Do dropout if needed
            if is_train and keep_prob < 1.0:
                print('doing dropout with prob {}'.format(config.keep_prob))
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=keep_prob)

            initial_state = lstm_cell.zero_state(batch_size, d_type)

            lstm_stack.append(lstm_cell)
            init_state_stack.append(initial_state)
            state_stack.append(initial_state)

        # cash our initial states
        self.initial_state = init_state_stack

        # Need an empty total output list of ys
        outputs = []

        # we need this variable scope to prevent us from creating multiple
        # independent weight/bias vectors for LSTM cell
        with tf.variable_scope('RNN'):

            # For each time step
            for time_step in range(num_steps):

                # This is y_i for a single time step
                time_step_output = []

                # Prevent creating indep weights for LSTM
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                for i in range(steps_in_cascade):

                    with tf.variable_scope("RNN"+str(i)):
                        # Run the lstm cell using the current timestep of
                        # input and the previous state to get the output and the new state
                        # print('input is', self.lstm_input[:, time_step, :])
                        # print('state', state)
                        curr_lstm_cell = lstm_stack[i]
                        curr_state = state_stack[i]

                        # Need a special base case for the first lstm input 
                        if i == 0:
                            cell_input = self.lstm_input[:, time_step, :]
                        else:
                            # All of these variables will be defined because of our base case
                            # print('error on line 250')
                            # print('self.lstm_input[:, time_step, :]', self.lstm_input[:, time_step, :])
                            # print('cell_output', cell_output)
                            cell_input = tf.concat(concat_dim = 1, values = [self.lstm_input[:, time_step, :], cell_output])

                        # print('input: ', cell_input)
                        # print('state: ', curr_state)
                        (cell_output, curr_state) = curr_lstm_cell(cell_input,
                                             curr_state)

                        # Update our state list
                        state_stack[i] = curr_state

                        # Update the output for the single cell
                        time_step_output.append(cell_output)

                # print('time_step_output',time_step_output)
                # For every timestep, we need a valid y output that should be of N*L*(L+1)/2 
                concated_time_steps = tf.concat(concat_dim = 1 , values = time_step_output)
                outputs.append(concated_time_steps)

        # we need to bookmark the final state to preserve continuity
        # across batches when we run an epoch (see below)
        # note, this is a list
        self.final_state = state_stack

        # concatenate all the outputs together into a big rank-2
        # matrix where each row is of dimension hidden_size
        # not sure what this concatenate is doing
        lstm_output_rank2 = tf.reshape(tf.concat(1, outputs), [-1, final_high_dimension])


        if external_targets is None:
            # reshape original targets down to rank-2 tensor
            targets_rank2 = tf.reshape(model_input.y, [batch_size*num_steps, 2])
        else:
            targets_rank2 = external_targets

        ourMDN = MDN(lstm_output_rank2, targets_rank2, final_high_dimension, is_train)

        # The loss is now calculated from our MDN
        loss = ourMDN.compute_loss()

        # What we now care about is the mixture probabilities from our MDN
        self.mixture_prob = ourMDN.return_mixture_prob()

        # loss is calculated in our MDN
        self.loss = loss

        # generate a train_op if we need to
        if is_train:
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
        else:
            self.train_op = None


    def run_epoch(self, session, return_predictions=False):
        # we always fetch loss because we will return it, we also
        # always fetch the final state because we need to pass it
        # along to the next batch in the loop below.
        # final state is now a list!! Update!! of three state tensors
        fetches = {
            'loss': self.loss,
            'final_state': self.final_state
        }

        # we need to run the training op if we are doing training
        if self.train_op is not None:
            fetches['train_op'] = self.train_op

        # we need to fetch the network outputs if we are doing predictions
        if return_predictions:
            fetches['p'] = self.mixture_prob

        # run the initial state to feed the LSTM - this should just be
        # zeros
        state = session.run(self.initial_state)

        # we will sum up the total loss
        total_loss = 0.0

        # all_outputs = [np.array([[0.0, 0.0]])] # NOTE: why does Matt do this?!?!?!
        all_outputs = []

        ##################################################
        # for each batch:

        # scoping issue 
        # vals = None
        
        for step in range(self.model_input.epoch_size):

            for level in range(len(state)):
                # the input producer will take care of feeding in x/y,
                # but we need to feed in the LSTM state
                c, h = self.initial_state[level]
                feed_dict = { c: state[level].c, h: state[level].h }

                # run the computation graph?
                vals = session.run(fetches, feed_dict)

                # get the final LSTM state for the next iteration
                state = vals['final_state']

            # stash output if necessary
            if return_predictions:
                all_outputs.append(vals['p'])

            # update total loss
            total_loss += vals['loss']

        # do average
        total_loss /= self.loss_scale

        # return one or two things
        if not return_predictions:
            return total_loss
        else:
            return total_loss, np.vstack(all_outputs)
            

######################################################################
# plot input vs predictions

def make_plot(epoch, loss, test_data, pred):

    titlestr = '{} test set loss = {:.2f}'.format(epoch, loss)
    print(titlestr)

    plt.clf()
    plt.plot(test_data[:,0], test_data[:,1], 'b.')
    plt.plot(pred[:,0], pred[:,1], 'r.')
    plt.axis('equal')
    plt.title(titlestr)
    plt.savefig('test_data_pred_lstm_2.pdf')

######################################################################
# main function
    
def main():

    # configs are just named tuples
    Config = namedtuple('Config', 'batch_size, num_steps, hidden_size, keep_prob')

    # generate training and test configurations
    train_config = Config(batch_size=train_batch_size,
                          num_steps=train_num_steps,
                          hidden_size=hidden_size,
                          keep_prob=train_keep_prob)
    
    test_config = Config(batch_size=1,
                         num_steps=1,
                         hidden_size=hidden_size,
                         keep_prob=1)

    # range to initialize all weights to
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    # generate training data
    train_data = get_data(2000)

    # generate test data
    test_data = get_data(1000)

    # generate visualization data 
    query_data = test_data[0:200, :]
    xmin, xmax = test_data[:,0].min(), test_data[:,0].max()
    ymin, ymax = test_data[:,1].min(), test_data[:,1].max()

    xrng = np.linspace(xmin, xmax, 200, True)
    yrng = np.linspace(ymin, ymax, 200, True)

    xg, yg = np.meshgrid(xrng, yrng)

    xreshape, yreshape = xg.reshape(-1,1), yg.reshape(-1,1)

    mesh_target = np.hstack([xreshape, yreshape])
    mesh_target = mesh_target.reshape(-1, 1, 2).astype('float32')


    # generate input producers and models -- again, not 100% sure why
    # we do the name_scope here...
    with tf.name_scope('train'):
        train_input = Input(train_data, train_config)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            train_model = LSTMCascade(train_config, train_input, is_train=True)

    with tf.name_scope('valid'):
        valid_input = Input(train_data, train_config)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            valid_model = LSTMCascade(train_config, train_input, is_train=False)
            
    with tf.name_scope('test'):
        test_input = Input(test_data, test_config)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            test_model = LSTMCascade(test_config, test_input, is_train=False)

    with tf.name_scope('query'):
        query_input = Input(query_data, test_config)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            query_model = LSTMCascade(test_config, query_input, is_train=False, external_targets=mesh_target)

    # print out all trainable variables:
    tvars = tf.trainable_variables()
    print('trainable variables:')
    print('\n'.join(['  - ' + tvar.name for tvar in tvars]))

    # let's save our computation graph
    saver = tf.train.Saver()


    # create a session
    session = tf.Session()

    # need to explicitly start the queue runners so the index variable
    # doesn't hang. (not sure how PTB did this - I think the
    # Supervisor takes care of it)
    tf.train.start_queue_runners(session)
    for tvar in tvars:
        tf.add_to_collection('tvars',tvar)

    # initialize all the variables
    session.run(tf.global_variables_initializer())

    # for each epoch
    for epoch in range(num_epochs):

        # run the epoch & get training loss
        l = train_model.run_epoch(session)
        print('training loss at epoch {}    is {:.2f}'.format(epoch, l))
        if epoch % 250 == 0:
            print('Saving model..... ')
            saver.save(session, 'LSTM-MDN-model')
            print('training loss at epoch {} is {:.2f}'.format(epoch, l))

        # see if we should do a printed/graphical update
        if epoch % update_every == 0:

            print()

            l = valid_model.run_epoch(session)
            print('validation loss at epoch {} is {:.2f}'.format(epoch, l))

            l, pred = test_model.run_epoch(session, return_predictions=True)
            # make_plot('epoch {}'.format(epoch), l, test_data, pred)
            
            print()

    # do final update
    l, pred = query_model.run_epoch(session, return_predictions=True)
    make_plot('final', l, test_data, pred)
    

if __name__ == '__main__':

    main()
