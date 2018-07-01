from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

from data import ppls

#############
# Parameters
#############

training_epochs = 3

batch_size = 100

num_labels = 2


def ppl_model(ppls_size, error_frac, learning_rate, run_lbl):

    tensorboard = TensorBoard(log_dir='./keras_summary/' + run_lbl, histogram_freq=0,
                              write_graph=True)
    x, labels = ppls.load_data(ppls_size, error_frac).train.all_data()

    validation = ppls.load_data(ppls_size, error_frac).test.all_data()

    l1 = 200
    l2 = 100
    l3 = 60
    l4 = 30

    model = Sequential()
    model.add(Dense(l1, input_dim=ppls_size, activation='relu', name="dense-1"))
    model.add(Dense(l2, activation='relu', name="dense-2"))
    model.add(Dense(l3, activation='relu', name="dense-3"))
    model.add(Dense(l4, activation='relu', name="dense-4"))
    model.add(Dense(num_labels, activation='sigmoid', name="dense-out"))

    adam = optimizers.adam(lr=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(x, labels,
              validation_data=validation,
              epochs=training_epochs,
              batch_size=batch_size,
              callbacks=[tensorboard])


def main():

    for s in [5, 10, 261]:
        for e in [0.1, 0.3, 0.5, 0.7]:
            for lr in [0.05, 0.1, 0.5]:
                run_lbl = "ppl_%s,e_%s,lr_%.0E" % (s, e, lr)
                print("Run label: " + run_lbl)
                ppl_model(s, e, lr, run_lbl)

    # Step 6: Evaluation

    # x_test, y_test = ppls.load_data(file='var_test.npz')

    # test_data = {X: x_test, y: y_test}
    # print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))

if __name__ == '__main__':
    main()
