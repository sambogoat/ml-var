import tensorflow as tf
from data import ppls

#############
# Parameters
#############

training_epochs = 3

batch_size = 100

num_labels = 2

L1 = 100
L2 = 50


def ppl_model(ppls_size, error_frac, learning_rate, run_lbl):
    tf.reset_default_graph()
    sess = tf.Session()

    # Step 1 - Set-up

    # Graph inputs
    X = tf.placeholder(tf.float32, [None, ppls_size], name="X")
    y = tf.placeholder(tf.float32, [None, num_labels], name="labels")

    # Step 2 - Define Model

    with tf.name_scope("L1"):
        w = tf.Variable(tf.truncated_normal([ppls_size, L1], stddev=0.1), name="W")
        b = tf.Variable(tf.truncated_normal([L1], stddev=0.1), name="b")
        act_l1 = tf.nn.relu(tf.matmul(X, w) + b)
        tf.summary.histogram("w", w)
        tf.summary.histogram("b", b)
        tf.summary.histogram("activations", act_l1)

    with tf.name_scope("L2"):
        w = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1), name="W")
        b = tf.Variable(tf.truncated_normal([L2], stddev=0.1), name="b")
        act_l2 = tf.nn.relu(tf.matmul(act_l1, w) + b)
        tf.summary.histogram("w", w)
        tf.summary.histogram("b", b)
        tf.summary.histogram("activations", act_l2)

    with tf.name_scope("out"):
        w = tf.Variable(tf.truncated_normal([L2, num_labels], stddev=0.1), name="W")
        b = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1), name="b")
        logits = tf.matmul(act_l2, w) + b
        # Step 3 - Loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits), name="loss")
        tf.summary.histogram("w", w)
        tf.summary.histogram("b", b)
        tf.summary.histogram("activations", loss)
        tf.summary.scalar("loss", loss)

    # Step 4 - Optimiser

    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Step 5 - Training loop

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1), name="prediction")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./nn_summary/" + run_lbl)
    writer.add_graph(sess.graph)

    print("Training Epochs: {}".format(training_epochs))

    for epoch in range(training_epochs):

        epoch_ppls = ppls.load_data(ppls_size, error_frac)

        num_batches = int(epoch_ppls.train.length / batch_size)

        print("Number of batches: {} ({})".format(num_batches, batch_size))

        for i in range(num_batches):

            step = epoch*num_batches+i

            x_train_batch, y_train_batch = epoch_ppls.train.next_batch(batch_size)

            train_data = {X: x_train_batch, y: y_train_batch}

            sess.run(train, feed_dict=train_data)

            if step % 5 == 0:
                curr_accuracy, curr_cost, curr_summ = sess.run([accuracy, loss, summ], feed_dict=train_data)
                writer.add_summary(curr_summ, step)
                print(step, "accuracy=", curr_accuracy, ", cost=", curr_cost)


def main():

    # for s in [261]:
    #     for e in [0.3]:
    #         for lr in [0.1]:
    #             run_lbl = "ppl_%s,e_%s,lr_%.0E" % (s, e, lr)
    #             print("Run label: " + run_lbl)
    #             ppl_model(s, e, lr, run_lbl)

    for s in [5, 10, 261]:
        for e in [0.1, 0.3, 0.5, 0.7]:
            for lr in [0.05, 0.1, 0.5]:
                run_lbl = "ppl_%s,e_%s,lr_%.0E" % (s, e, lr)
                print("Run label: " + run_lbl)
                ppl_model(s, e, lr, run_lbl)

if __name__ == '__main__':
    main()



