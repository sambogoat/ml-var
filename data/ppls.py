import numpy as np


class DataSet:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.batch_no = 0
        self.length = x.shape[0]

    @staticmethod
    def normalise_features(x):
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        return (x - mu) / sigma

    def all_data(self, normalise=True):
        if normalise:
            return DataSet.normalise_features(self.x), self.y
        else:
            return self.x, self.y

    def next_batch(self, size, normalise=True):
        start = self.batch_no * size
        end = start + size
        self.batch_no += 1
        if normalise:
            return DataSet.normalise_features(self.x[start:end],), self.y[start:end]
        else:
            return self.x[start:end], self.y[start:end]


class PPLs:
    """PPL training and test data"""

    def __init__(self, size, error_frac):

        with np.load("ppls_{}_{}.npz".format(size, error_frac)) as f:
            self.train = DataSet(f['x'], f['y'])

        with np.load("ppls_{}_{}_test.npz".format(size, error_frac)) as f:
            self.test = DataSet(f['x'], f['y'])


# #############
# PPL Loader
# #############
def load_data(size, error_frac):
    return PPLs(size, error_frac)


# #############
# PPL Generator
# #############
def generate_and_save_data(train_size, var_size, error_frac, is_test=False, decimals=4, high=100):

    # The % of VaR errors
    ok_error_split = int(train_size * error_frac)

    # The VaR 'features' that represent an vector in error, i.e. first, middle and last
    features_in_error = [0, int(var_size / 2), var_size-1]

    ppls = np.around(np.random.uniform(low=1.0, high=high, size=(train_size, var_size)), decimals=decimals)

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

    # Concatenate the two arrays and shuffle maintaining relative ordering between ppls and labels.
    all_x = np.concatenate((good_ppls, error_ppls))
    all_y = np.concatenate((good_labels, error_labels))
    perm = np.random.permutation(train_size)
    all_x_shuffled = all_x[perm]
    all_y_shuffled = all_y[perm]
    assert np.array_equal(np.where(np.amin(all_x_shuffled, axis=1) == 0)[0], np.where(all_y_shuffled[:, 1] == 1)[0])

    if is_test:
        file = "ppls_{}_{}_test.npz".format(var_size, error_frac)
    else:
        file = "ppls_{}_{}.npz".format(var_size, error_frac)

    np.savez(file, x=all_x_shuffled, y=all_y_shuffled)

# ##########
# Plot Data
# ##########
def plot():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    ppls_size = 261
    x, y = load_data(ppls_size, 0.3).test.all_data()
    l = len(x)
    i = list(range(ppls_size))
    a = np.reshape(np.array(i * l), [l, ppls_size])
    plt.scatter(a, x)
    plt.show()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(x))))
    for xi in x:
        plt.scatter(i, xi, color=next(colors))

# Generate training and test data
# for s in [5, 10, 261]:
#     for e in [0.1, 0.3, 0.5, 0.7]:
#         generate_and_save_data(train_size=100000, var_size=s, error_frac=e)
#        generate_and_save_data(train_size=10000, var_size=s, error_frac=e, is_test=True)

if __name__ == "__main__":
    plot()
