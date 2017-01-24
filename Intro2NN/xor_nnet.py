######################################################################
#
# E27 - Computer Vision - Spring 2017 - Homework 12
#
# Homework submitted by: John Larkin
#

import numpy as np

# Activation function for our neural network
def f(x):
    return np.tanh(x)

# Derivative of activation function
def fprime(x):
    return 1-np.tanh(x)**2

# Set of all inputs to XOR function
inputs = np.array([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
])

# Set of all desired outputs of XOR function corresponding to each
# input above.
targets = np.array([
    -1.0,
    1.0,
    1.0,
    -1.0
])

# Learning rate
alpha = 0.5

# Stop when all outputs within tol of target
tol = 0.1

# Should we initialize to known good weights?
cheat = False

if cheat:

    # We derived weights like these in class
    w03, w13, w23 = -2.0,  2.0, -2.0
    w04, w14, w24 = -2.0, -2.0,  2.0
    w05, w35, w45 =  2.0,  2.0,  2.0

else:

    # Chosen to seem random but converge in less than 100 iterations.
    w03, w13, w23 =  0.14,  0.13, -0.23
    w04, w14, w24 = -0.07, -0.23,  0.11
    w05, w35, w45 =  0.08,  0.04, -0.05

# Train for 100 iterations:
for training_epoch in range(100):

    # Initialize all weight Deltas to zero for all weights.
    D03 = 0.0
    D13 = 0.0
    D23 = 0.0

    D04 = 0.0
    D14 = 0.0
    D24 = 0.0
    
    D05 = 0.0
    D35 = 0.0
    D45 = 0.0

    # Initialize
    err_sum = 0.0

    # We will stop training once everything is right; if any answer is
    # wrong below, we will set can_stop = False.
    can_stop = True

    print 'Starting training epoch {}'.format(training_epoch)
    print

    # Loop over inputs and desired outputs
    for input_vec, target in zip(inputs, targets):

        # Separate out inputs
        y1, y2 = tuple(input_vec)

        # Step 1: Feed-forward from input layer to hidden layer, then
        # hidden to output.
        x3 = w03 + w13*y1 + w23*y2
        y3 = f(x3)

        x4 = w04 + w14*y1 + w24*y2
        y4 = f(x4)
        
        x5 = w05 + w35*y3 + w45*y4
        y5 = f(x5)

        # Step 2a: Compute the difference between the target and the
        # actual output, and use that to compute the error and update
        # the error sum.
        diff = target - y5
        err = 0.5*diff*diff
        err_sum += err

        print ('  Output for input ({: 2.0f}, {: 2.0f}) is {: 1.9f}; '
               'target={: 2.0f}, diff={: 1.9f}'.format(
                   y1, y2, y5, target, diff))
        
        # We will halt training when all outputs are within a
        # pre-specified tolerance of the target
        if np.abs(diff) >= tol:
            can_stop = False
        
        # Step 2b: Compute the node deltas for all non-input nodes.
        # You will need to compute d5, then compute d3 and d4. Note
        # you can use the derivative of the activation function,
        # defined above.

        d5 = (target - y5) * fprime(x5) # should be good
        d4 = fprime(x4) * d5 * w45
        d3 = fprime(x3) * d5 * w35
        
        # Step 3a: Compute the gradient terms gij for all weights
        # wij. You can do this in any order, once the node deltas have
        # been computed. You will need 9 gradient terms for the 9 weights.

        # G33 = y3 * D03, G43 = y4 * D03, G53 = y5 * D03
        # G34 = y3 * D04, G44 = y4 * D04, G54 = y5 * D04
        # G35 = y3 * D05, G45 = y4 * D05, G55 = y5 * D05

        G03 = d3 * 1 # not sure which y to multiply by
        G13 = d3 * y1
        G23 = d3 * y2

        G04 = d4 * 1
        G14 = d4 * y1
        G24 = d4 * y2
        
        G05 = d5 * 1
        G35 = d5 * y3
        G45 = d5 * y4
        
        # Step 3b: Accumulate all 9 gradient terms into weight
        # Deltas. If you want, you can merge steps 3a and 3b together.

        D03 += G03
        D13 += G13
        D23 += G23

        D04 += G04
        D14 += G14
        D24 += G24
        
        D05 += G05
        D35 += G35
        D45 += G45
        
    ##################################################
    # Finished looking at all input/output pairs

    print
    print '  Error sum at iteration {} is {}'.format(training_epoch, err_sum)
    print

    if can_stop:
        print 'All correct, so stopping!'
        break

    # Now do weight update
    w13 += alpha * D13
    w23 += alpha * D23
    w03 += alpha * D03

    w14 += alpha * D14
    w24 += alpha * D24
    w04 += alpha * D04

    w35 += alpha * D35
    w45 += alpha * D45
    w05 += alpha * D05

##################################################
# all done

if not can_stop:
    print 'Sad, did not learn classifier :('

