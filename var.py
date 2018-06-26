from data import ppls
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

######################################################################################################################
#
# Links:
#   http://jrmeyer.github.io/machinelearning/2016/02/01/TensorFlow-Tutorial.html
#
#
# Results:
#   5% errors, var_size=5, batch_size=1000, epochs=3, learning_rate=0.1 - Testing Accuracy =  0.9995
#   5% errors, var_size=261, batch_size=1000, epochs=3, learning_rate=0.1 - Testing Accuracy =  0.95
#   5% errors, var_size=261, batch_size=500, epochs=2,3, learning_rate=0.1 - Testing Accuracy =  0.95
#   15% errors, var_size=261, batch_size=500,1000, epochs=2, learning_rate=0.1 - Testing Accuracy =  0.85
#   15% errors, var_size=261, batch_size=1000, epochs=2, learning_rate=0.1 - VaR [1.0-2.0] - Testing Accuracy =  0.9997
#   20% errors, var_size=261, batch_size=1000, epochs=2, learning_rate=0.1 - Testing Accuracy =  0.80
#
# <<< COST IS ALWAYS THE % ERRORS WHEN VAR IS 261; WHEN 5 THEN COST APPROACHES 0.
#
######################################################################################################################

#############
# Parameters
#############

learning_rate = 0.1

training_epochs = 3

batch_size = 1000

# TODO - Move this to data module
def next_batch(x, y, size, num):
    s = num * size
    e = s + size
    return x[s:e], y[s:e]

###################
# Step 1 - Set-up
###################

# Load all the data
X_train, y_train = ppls.load_data()

num_features = X_train.shape[1]

num_labels = y_train.shape[1]

# Graph inputs
X = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_labels])

# Model weights
W = tf.Variable(tf.truncated_normal([num_features, num_labels], stddev=0.1), name="W")
b = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1), name="b")

init = tf.global_variables_initializer()

##############################################################
# Step 2 - Define Model (i.e. activation function)
##############################################################

yhat = tf.nn.sigmoid(tf.matmul(X, W) + b, name="activation")

#####################################################
# Step 3 - Definition Evaluation (i.e. cost function)
#####################################################

#cost = tf.nn.l2_loss(yhat-y, name="squared_error_cost")
cost = tf.reduce_mean(tf.square(yhat - y))


####################
# Step 4 - Optimiser
####################

optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

################
# Visualisations
################

# TODO - FIX: The plot does not update

plt.ion()
# Create the main, super plot
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()
plt.pause(.3)

step_values = []
accuracy_values = []
cost_values = []

########################
# Step 5 - Training loop
########################

sess = tf.Session()

sess.run(init)

prediction = tf.argmax(yhat, axis=1)

correct_prediction = tf.equal(prediction, tf.argmax(y, axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

yhat_summary = tf.summary.histogram("yhat", yhat)

cost_summary = tf.summary.scalar("cost", cost)

accuracy_summary = tf.summary.scalar("accuracy", accuracy)

weight_summary = tf.summary.histogram("weights", W.eval(session=sess))

bias_summary = tf.summary.histogram("biases", b.eval(session=sess))

all_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("./summary", sess.graph)

print("Training Epochs: {}".format(training_epochs))

for epoch in range(training_epochs):

    num_batches = int(len(X_train) / batch_size)

    print("Number of batches: {} ({})".format(num_batches, batch_size))

    for i in range(num_batches):

        step = epoch*num_batches+i

        x_train_batch, y_train_batch = next_batch(X_train, y_train, batch_size, i)

        train_data = {X: x_train_batch, y: y_train_batch}

        sess.run(optimiser, feed_dict=train_data)

        if step % 10 == 0:

            step_values.append(step)

            curr_prediction, curr_accuracy, curr_cost, curr_summary = sess.run([prediction, accuracy, cost, all_summary], feed_dict=train_data)

            accuracy_values.append(curr_accuracy)

            cost_values.append(curr_cost)

            writer.add_summary(curr_summary, step)

            print(step, "accuracy=", curr_accuracy, ", cost=", curr_cost)

            print(step, len(curr_prediction[curr_prediction == 1]))

            # Plot progress
            ax1.plot(step_values, accuracy_values)
            ax2.plot(step_values, cost_values)
            fig.canvas.draw()


####################
# Step 6: Evaluation
####################

x_test, y_test = ppls.load_data(file='var_test.npz')

test_data = {X: x_test, y: y_test}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))


