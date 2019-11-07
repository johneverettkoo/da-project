import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import gist
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import gc


def extract_gist_features(images, nblocks=4):
    """extract gist features from images"""

    gist_features = [gist.extract(images[i, :, :, :], nblocks) for i in np.arange(images.shape[0])]
    gist_features = np.stack(gist_features)
    return gist_features


def get_greedy_perm(points, k):
    """
    https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    points: ndarray (N, p)
        An Nxp data matrix
    k: int
    Return
    ------
    permutation (N-length array of indices)
    """

    N = points.shape[0]

    idx = np.random.randint(N, size=1)
    permutation = np.zeros(k, dtype=np.int64)
    permutation[0] = idx

    dist_matrix = pairwise_distances(points, metric='euclidean')
    ds = dist_matrix[idx, :]
    # ds = distance.cdist(points[idx, :], points, 'euclidean')
    for i in range(1, k):
        idx = np.argmax(ds)
        permutation[i] = idx
        ds = np.minimum(ds, dist_matrix[idx, :])
    return permutation


def similar_or_diverse(points, k=50, method='similar'):
    """find either a diverse or similar set of points from a given sample"""

    if method == 'diverse':
        # solution_set, solution_set_indices incremental_farthest_search(points, k)
        # solution_set = np.stack(solution_set, 0)
        solution_set_indices = get_greedy_perm(points, k)
        solution_set = points[solution_set_indices, :]
    elif method == 'similar':
        N = points.shape[0]
        idx = np.random.randint(N, size=1)
        ds = spatial.distance.cdist(points[idx, :], points, 'euclidean')
        sorted_idx = np.argsort(ds).flatten()
        solution_set_indices = sorted_idx[np.arange(k)]
        solution_set = points[solution_set_indices, :]
    else:
        raise ValueError('not a valid method')

    return solution_set, solution_set_indices


def split_dataset_diversity(images, gist_features, train_size=50, val_size=1000):
    """splits a dataset into similar, diverse, random, and validation sets"""

    # we assume the first index is the number of observations
    n = gist_features.shape[0]
    indices = np.arange(n)

    # set aside validation set
    np.random.shuffle(indices)
    val_ind = indices[0:val_size]
    val_images = images[val_ind, :, :, :]
    train_ind = indices[val_size:n]
    train_set = gist_features[train_ind, :]
    train_images = images[train_ind, :, :, :]

    # find a set of diverse images
    _, far_ind = similar_or_diverse(train_set, train_size, 'diverse')
    diverse_images = train_images[far_ind, :, :, :]

    # find a set of similar images
    _, near_ind = similar_or_diverse(train_set, train_size, 'similar')
    similar_images = train_images[near_ind, :, :, :]

    # find a set of random images
    random_images = train_images[0:train_size, :, :, :]

    return diverse_images, similar_images, random_images, val_images


def initialize_vgg16(input_shape, n_classes, lr=.001, momentum=.9):
    """initialize vgg16 with imagenet weights"""

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                  metrics=['acc'])

    return model


def construct_train_val_by_class(x_train, y_train, x_gist,
                                 train_size=100, val_size=1000):
    """construct divers,e similar, and random subsets of training data"""

    # what are the unique classes and how many are there
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    # prepare the training and validation data
    y_diverse = np.repeat(unique_classes, train_size).reshape(n_classes * train_size, 1)
    y_similar = np.repeat(unique_classes, train_size).reshape(n_classes * train_size, 1)
    y_random = np.repeat(unique_classes, train_size).reshape(n_classes * train_size, 1)
    y_val = np.repeat(unique_classes, val_size).reshape(n_classes * val_size, 1)
    x_diverse = []
    x_similar = []
    x_random = []
    x_val = []

    # split the data for each class to maintain class balance
    for cl in unique_classes:
        # get the images and gist features for this class
        x_cl = x_train[y_train[:, 0] == cl, :, :, :]
        x_cl_g = x_gist[y_train[:, 0] == cl, :]

        # construct the training sets and validation set for this class
        x_cl_div, x_cl_sim, x_cl_rand, x_cl_val = \
            split_dataset_diversity(x_cl, x_cl_g, train_size, val_size)

        # save
        x_diverse.append(x_cl_div)
        x_similar.append(x_cl_sim)
        x_random.append(x_cl_rand)
        x_val.append(x_cl_val)

    # reshape to valid numpy arrays
    x_diverse = np.concatenate(x_diverse, 0)
    x_similar = np.concatenate(x_similar, 0)
    x_random = np.concatenate(x_random, 0)
    x_val = np.concatenate(x_val, 0)

    # return
    return x_diverse, y_diverse, x_similar, y_similar, x_random, y_random, x_val, y_val


def repeat_experiment(x_train, y_train, x_val, y_val, x_test, y_test,
                      runs=10,
                      lr=.001, momentum=.9, batch_size=32,
                      patience=2, max_epochs=100, verbose=1):
    """train vgg16 for a dataset multiple times"""

    # what is the image shape
    x_length = x_train.shape[1]
    x_width = x_train.shape[2]
    x_channels = x_train.shape[3]

    # how many classes
    n_classes = len(np.unique(y_train))

    # reshape responses for training
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test)

    # determine how to stop the training
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    # run experiment on diverse set
    accuracies = np.empty(runs)
    losses = np.empty(runs)
    for run in np.arange(runs):
        # shuffle the training data
        train_ind = np.arange(x_train.shape[0])
        np.random.shuffle(train_ind)
        x_train = x_train[train_ind, :, :, :]
        y_train = y_train[train_ind, :]

        # train the model
        model = initialize_vgg16((x_length, x_width, x_channels), n_classes, lr, momentum)
        model.fit(x_train, y_train,
                  validation_data=(x_val, y_val),
                  epochs=max_epochs,
                  batch_size=batch_size,
                  verbose=verbose,
                  shuffle=True,
                  callbacks=[earlystopping])

        # compute performance metrics
        loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
        losses[run] = loss
        accuracies[run] = accuracy

        # clear memory
        K.clear_session()
        del model
        gc.collect()

    # return results
    return accuracies, losses


def diversity_experiment_single(x_train, y_train,
                                x_gist,
                                x_test, y_test,
                                train_size=100, val_size=1000,
                                runs=10,
                                lr=.001, momentum=.9, batch_size=32,
                                patience=2, max_epochs=100,
                                verbose=0):
    """perform diversity experiment for a single train size"""

    # run experiments for the three datasets and repeat
    diverse_accuracies = np.empty(runs)
    diverse_losses = np.empty(runs)
    similar_accuracies = np.empty(runs)
    similar_losses = np.empty(runs)
    random_accuracies = np.empty(runs)
    random_losses = np.empty(runs)
    for run in np.arange(runs):
        # construct the training sets and the validation set
        x_diverse, y_diverse, x_similar, y_similar, x_random, y_random, x_val, y_val = \
            construct_train_val_by_class(x_train, y_train, x_gist, train_size, val_size)

        diverse_accuracy, diverse_loss = repeat_experiment(x_diverse, y_diverse,
                                                           x_val, y_val,
                                                           x_test, y_test,
                                                           1,
                                                           lr, momentum, batch_size,
                                                           patience, max_epochs,
                                                           verbose)
        similar_accuracy, similar_loss = repeat_experiment(x_diverse, y_diverse,
                                                           x_val, y_val,
                                                           x_test, y_test,
                                                           1,
                                                           lr, momentum, batch_size,
                                                           patience, max_epochs,
                                                           verbose)
        random_accuracy, random_loss = repeat_experiment(x_diverse, y_diverse,
                                                         x_val, y_val,
                                                         x_test, y_test,
                                                         1,
                                                         lr, momentum, batch_size,
                                                         patience, max_epochs,
                                                         verbose)
        diverse_accuracies[run] = diverse_accuracy
        diverse_losses[run] = diverse_loss
        similar_accuracies[run] = similar_accuracy
        similar_losses[run] = similar_loss
        random_accuracies[run] = random_accuracy
        random_losses[run] = random_loss

    # compile results into dataframe
    losses = np.concatenate([diverse_losses, similar_losses, random_losses])
    accuracies = np.concatenate([diverse_accuracies, similar_accuracies, random_accuracies])
    train_sets = np.concatenate([np.repeat('diverse', runs),
                                 np.repeat('similar', runs),
                                 np.repeat('random', runs)])
    results_df = pd.DataFrame({'train_size': np.repeat(train_size, runs * 3),
                               'train_set_type': train_sets,
                               'loss': losses,
                               'accuracy': accuracies})

    return results_df


def diversity_experiment(x_train, y_train,
                         x_gist,
                         x_test, y_test,
                         train_sizes,
                         val_size=1000,
                         runs=10,
                         lr=.0001, momentum=.9, batch_size=32,
                         patience=2, max_epochs=100,
                         verbose=0):
    """perform diversity experiment for a range of train sizes"""

    results_df = []

    for train_size in train_sizes:
        print(f'performing experiment for train size={train_size}')

        sub_df = diversity_experiment_single(x_train, y_train,
                                             x_gist,
                                             x_test, y_test,
                                             train_size=train_size, val_size=val_size, runs=runs,
                                             lr=lr, momentum=momentum, batch_size=batch_size,
                                             patience=patience, max_epochs=max_epochs,
                                             verbose=verbose)
        results_df.append(sub_df)

    results_df = pd.concat(results_df)

    return results_df
