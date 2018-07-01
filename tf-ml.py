import tensorflow as tf
import matplotlib.pyplot as plt
from data import ppls

#############
# Parameters
#############

training_epochs = 3

batch_size = 100

num_labels = 2

######################
# Setup Visualisations
# TODO - FIX: The plot does not update
######################

plt.ion()
# Create the main, super plot
# fig = plt.figure()
# Create two subplots on their own axes and give titles
# ax1 = plt.subplot("211")
# ax1.set_title("TRAINING ACCURACY", fontsize=18)
# ax2 = plt.subplot("212")
# ax2.set_title("TRAINING COST", fontsize=18)
# plt.tight_layout()
# plt.pause(.3)

# step_values = []
# accuracy_values = []
# cost_values = []


def ppl_model(ppls_size, error_frac, learning_rate, run_lbl, normalise = True):
    tf.reset_default_graph()
    sess = tf.Session()

    # Step 1 - Set-up

    # Graph inputs
    X = tf.placeholder(tf.float32, [None, ppls_size], name="X")
    y = tf.placeholder(tf.float32, [None, num_labels], name="labels")

    # Model weights
    W = tf.Variable(tf.truncated_normal([ppls_size, num_labels], stddev=0.1), name="W")
    b = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1), name="b")
    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)

    # Step 2 - Define Model (i.e. activation function)

    act = tf.nn.sigmoid(tf.matmul(X, W) + b, name="activation")
    tf.summary.histogram("activations", act)

    # Step 3 - Definition Evaluation (i.e. cost function)

    with tf.name_scope("cost"):
        cost = tf.nn.l2_loss(act-y, name="squared_error_cost")
        tf.summary.scalar("cost", cost)

    # Step 4 - Optimiser

    with tf.name_scope("train"):
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Step 5 - Training loop

    with tf.name_scope("accuracy"):
        prediction = tf.argmax(act, axis=1)
        correct_prediction = tf.equal(prediction, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./summary/" + run_lbl)
    writer.add_graph(sess.graph)

    print("Training Epochs: {}".format(training_epochs))

    for epoch in range(training_epochs):

        epoch_ppls = ppls.load_data(ppls_size, error_frac)

        num_batches = int(epoch_ppls.train.length / batch_size)

        print("Number of batches: {} ({})".format(num_batches, batch_size))

        for i in range(num_batches):

            step = epoch*num_batches+i

            x_train_batch, y_train_batch = epoch_ppls.train.next_batch(batch_size, normalise=normalise)

            train_data = {X: x_train_batch, y: y_train_batch}

            sess.run(train, feed_dict=train_data)

            if step % 5 == 0:

                curr_accuracy, curr_cost, curr_summ = sess.run([accuracy, cost, summ], feed_dict=train_data)
                writer.add_summary(curr_summ, step)

                # step_values.append(step)
                # accuracy_values.append(curr_accuracy)
                # cost_values.append(curr_cost)

                # Plot progress
                # ax1.plot(step_values, accuracy_values)
                # ax2.plot(step_values, cost_values)
                # fig.canvas.draw()

                print(step, "accuracy=", curr_accuracy, ", cost=", curr_cost)

    # Step 6: Evaluation
    x_test, y_test = ppls.load_data(ppls_size, error_frac).test.all_data(normalise)
    test_data = {X: x_test, y: y_test}
    print("Testing Accuracy = ", sess.run(accuracy, feed_dict=test_data))


def main():

    normalise = True

    for s in [261]:
        for e in [0.3]:
            for lr in [0.1]:
                run_lbl = "ppl_%s,e_%s,lr_%.0E,norm_%s" % (s, e, lr, normalise)
                print("Run label: " + run_lbl)
                ppl_model(s, e, lr, run_lbl, normalise)

    # for s in [5, 10, 261]:
    #     for e in [0.1, 0.3, 0.5, 0.7]:
    #         for lr in [0.05, 0.1, 0.5]:
    #             run_lbl = "ppl_%s,e_%s,lr_%.0E,norm_%s" % (s, e, lr, normalise)
    #             print("Run label: " + run_lbl)
    #             ppl_model(s, e, lr, run_lbl, normalise)

if __name__ == '__main__':
    main()



