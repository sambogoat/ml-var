import numpy as np


def generate_and_save_data(train_size=100, var_size=261, file='var.npz'):

    # The % of VaR errors
    errors = 0.05
    ok_error_split = int(train_size * errors)

    # The VaR 'features' that represent an vector in error, i.e. first, middle and last
    features_in_error = [0, int(var_size / 2), var_size-1]

    ppls = np.random.uniform(low=1.0, high=100.0, size=(train_size, var_size))

    # Create the error ppls, i.e. setting the features to 0
    error_ppls = ppls[0:ok_error_split]
    for i in features_in_error:
        error_ppls[:, i] = 0

    # Error labels
    error_labels = np.transpose(np.array([np.zeros(len(error_ppls), dtype=np.int32),
                                          np.ones(len(error_ppls), dtype=np.int32)]))

    # Valid ppls and labels, i.e. 1.
    good_ppls = ppls[ok_error_split:len(ppls)]
    good_labels = np.transpose(np.array([np.ones(len(good_ppls), dtype=np.int32),
                                         np.zeros(len(good_ppls), dtype=np.int32)]))

    # Concatenate the two arrays and shuffle maintaining relataive ordering between ppls and labels.
    all_x = np.concatenate((good_ppls, error_ppls))
    all_y = np.concatenate((good_labels, error_labels))
    perm = np.random.permutation(train_size)
    all_x_shuffled = all_x[perm]
    all_y_shuffled = all_y[perm]
    assert np.array_equal(np.where(np.amin(all_x_shuffled, axis=1) == 0)[0], np.where(all_y_shuffled[:,1] == 1)[0])

    np.savez(file, x=all_x_shuffled, y=all_y_shuffled)
    #np.savetxt("./ppls.csv", np.append(all_x_shuffled, all_y_shuffled, axis=1), delimiter=",", newline="\n")


def load_data(file='var.npz'):
    with np.load(file) as f:
        x, y = f['x'], f['y']
        return x, y

generate_and_save_data(train_size=100000, var_size=261)

generate_and_save_data(train_size=10000, var_size=261, file="var_test.npz")

