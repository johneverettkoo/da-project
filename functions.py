import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cluster
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


def split_dataset_diversity_subsample(images, gist_features, train_size=50, val_size=1000):
    """splits a subsampled dataset into similar, diverse, random, and validation sets"""

    n = gist_features.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    # set aside validation set
    val_ind = indices[0:val_size]
    val_images = images[val_ind, :, :, :]
    train_ind = indices[val_size:n]
    train_features = gist_features[train_ind, :]
    train_images = images[train_ind, :, :, :]

    # sample a random training subset
    # and sample a subset to divide into similar vs diverse training subsets
    indices = np.arange(train_features.shape[0])
    np.random.shuffle(indices)
    diverse_similar_ind = indices[0:(2 * train_size)]
    random_ind = indices[(2 * train_size):(3 * train_size)]
    diverse_similar_features = train_features[diverse_similar_ind, :]
    diverse_similar_images = train_images[diverse_similar_ind, :, :, :]
    random_images = train_images[random_ind, :, :, :]

    # divide subset into similar/diverse
    _, diverse_ind = similar_or_diverse(diverse_similar_features, train_size, 'diverse')
    diverse_images = diverse_similar_images[diverse_ind, :, :, :]
    similar_images = np.delete(diverse_similar_images, diverse_ind, 0)

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


def construct_train_val_by_class(x_train, y_train, gist_train,
                                 train_size=100, val_size=1000):
    """construct diverse, similar, and random subsets of training data"""

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
        x_cl = x_train[y_train.flatten() == cl, :, :, :]
        x_cl_g = gist_train[y_train.flatten() == cl, :]

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


def construct_train_val_by_class_subsampled(x_train, y_train, gist_train,
                                            train_size=100, val_size=1000):
    """construct diverse, similar, and random subsets of training data after subsampling"""

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
        x_cl_g = gist_train[y_train[:, 0] == cl, :]

        # construct the training sets and validation set for this class
        x_cl_div, x_cl_sim, x_cl_rand, x_cl_val = \
            split_dataset_diversity_subsample(x_cl, x_cl_g, train_size, val_size)

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


def construct_train_val_by_class_edited(x_train, y_train, gist_train, remove_ind,
                                        train_size=100, val_size=1000):
    """construct diverse, similar, and random subsets of training data with editing"""

    # construct an edited training set
    x_train_edited = np.delete(x_train, remove_ind, 0)
    y_train_edited = np.delete(y_train, remove_ind, 0)
    gist_train_edited = np.delete(gist_train, remove_ind, 0)

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
        # get the images and gist features for this class from both edited and unedited sets
        x_cl = x_train[y_train[:, 0] == cl, :, :, :]
        x_cl_g = gist_train[y_train[:, 0] == cl, :]
        x_cl_edit = x_train_edited[y_train_edited[:, 0] == cl, :, :, :]
        x_cl_g_edit = gist_train_edited[y_train_edited[:, 0] == cl, :]

        # construct the training sets and validation set for this class
        _, x_cl_sim, x_cl_rand, x_cl_val = \
            split_dataset_diversity(x_cl, x_cl_g, train_size, val_size)
        _, x_cl_div_ind = similar_or_diverse(x_cl_g_edit, train_size, 'diverse')
        x_cl_div = x_cl_edit[x_cl_div_ind, :, :, :]

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


def experiment(x_train, y_train, x_val, y_val, x_test, y_test,
               lr=.0001, momentum=.9, batch_size=32,
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

    # clear memory
    K.clear_session()
    del model
    gc.collect()

    # return results
    return accuracy, loss


def diversity_experiment_single(x_train, y_train,
                                gist_train,
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
            construct_train_val_by_class(x_train, y_train, gist_train, train_size, val_size)

        diverse_accuracy, diverse_loss = experiment(x_diverse, y_diverse,
                                                    x_val, y_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'diverse accuracy: {diverse_accuracy}')
        similar_accuracy, similar_loss = experiment(x_diverse, y_diverse,
                                                    x_val, y_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'similar accuracy: {similar_accuracy}')
        random_accuracy, random_loss = experiment(x_diverse, y_diverse,
                                                  x_val, y_val,
                                                  x_test, y_test,
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
                         gist_train,
                         x_test, y_test,
                         train_sizes,
                         val_size=1000,
                         runs=10,
                         lr=.0001, momentum=.9, batch_size=32,
                         patience=2, max_epochs=100,
                         verbose=0):
    """perform diversity experiment for a range of train sizes"""

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    results_df = []

    for train_size in train_sizes:
        print(f'performing experiment for train size={train_size}')

        sub_df = diversity_experiment_single(x_train, y_train,
                                             gist_train,
                                             x_test, y_test,
                                             train_size=train_size, val_size=val_size, runs=runs,
                                             lr=lr, momentum=momentum, batch_size=batch_size,
                                             patience=patience, max_epochs=max_epochs,
                                             verbose=verbose)
        results_df.append(sub_df)

    results_df = pd.concat(results_df)

    return results_df


def split_train_val(x, y, val_prop=.5):
    """split data into training and validation sets with equal class proportions between both"""

    classes = np.unique(y)

    x_train = []
    x_val = []
    y_train = []
    y_val = []

    for cl in classes:
        x_sub = x[y.flatten() == cl, :, :, :]
        n = x_sub.shape[0]
        train_size = int(n * (1 - val_prop))
        np.random.shuffle(x_sub)
        x_train_sub = x_sub[0:train_size, :, :, :]
        x_val_sub = x_sub[train_size:n, :, :, :]
        x_train.append(x_train_sub)
        x_val.append(x_val_sub)
        y_train.append(np.repeat(cl, train_size))
        y_val.append(np.repeat(cl, n - train_size))

    x_train = np.concatenate(x_train, 0)
    x_val = np.concatenate(x_val, 0)
    y_train = np.concatenate(y_train, 0)
    y_val = np.concatenate(y_val, 0)

    y_train = y_train.reshape(len(y_train), 1)
    y_val = y_val.reshape(len(y_val), 1)

    return x_train, y_train, x_val, y_val


def diversity_experiment_single_with_constrained_val(x_train, y_train,
                                                     gist_train,
                                                     x_test, y_test,
                                                     train_size=100, val_prop=.5,
                                                     runs=10,
                                                     lr=.001, momentum=.9, batch_size=32,
                                                     patience=2, max_epochs=100,
                                                     verbose=0):
    """perform diversity experiment for single train size and nonrandom sampled validation set"""

    # run experiments for the three datasets and repeat
    diverse_accuracies = np.empty(runs)
    diverse_losses = np.empty(runs)
    similar_accuracies = np.empty(runs)
    similar_losses = np.empty(runs)
    random_accuracies = np.empty(runs)
    random_losses = np.empty(runs)
    for run in np.arange(runs):
        # construct the training sets
        x_diverse, y_diverse, x_similar, y_similar, x_random, y_random, _, _ = \
            construct_train_val_by_class(x_train, y_train, gist_train, int(train_size / val_prop), 0)

        # set aside some training data as validation data
        x_diverse_train, y_diverse_train, x_diverse_val, y_diverse_val = \
            split_train_val(x_diverse, y_diverse, val_prop)
        x_similar_train, y_similar_train, x_similar_val, y_similar_val = \
            split_train_val(x_similar, y_similar, val_prop)
        x_random_train, y_random_train, x_random_val, y_random_val = \
            split_train_val(x_random, y_random, val_prop)

        # run the three experiments
        diverse_accuracy, diverse_loss = experiment(x_diverse_train, y_diverse_train,
                                                    x_diverse_val, y_diverse_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'diverse accuracy: {diverse_accuracy}')
        similar_accuracy, similar_loss = experiment(x_similar_train, y_similar_train,
                                                    x_similar_val, y_similar_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'similar accuracy: {similar_accuracy}')
        random_accuracy, random_loss = experiment(x_random_train, y_random_train,
                                                  x_random_val, y_random_val,
                                                  x_test, y_test,
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


def diversity_experiment_constrained_val(x_train, y_train,
                                         gist_train,
                                         x_test, y_test,
                                         train_sizes,
                                         val_prop,
                                         runs=10,
                                         lr=.001, momentum=.9, batch_size=32,
                                         patience=2, max_epochs=100,
                                         verbose=0):
    """perform diversity experiment for a range of train sizes and nonrandom sampled validation set"""

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    results_df = []

    for train_size in train_sizes:
        print(f'performing experiment for train size = {train_size}')
        sub_df = diversity_experiment_single_with_constrained_val(x_train, y_train,
                                                                  gist_train,
                                                                  x_test, y_test,
                                                                  train_size=train_size, val_prop=val_prop, runs=runs,
                                                                  lr=lr, momentum=momentum, batch_size=batch_size,
                                                                  patience=patience, max_epochs=max_epochs,
                                                                  verbose=verbose)
        results_df.append(sub_df)

    results_df = pd.concat(results_df)

    return results_df


def diversity_experiment_single_subsampled(x_train, y_train,
                                           gist_train,
                                           x_test, y_test,
                                           train_size=100, val_size=1000,
                                           runs=10,
                                           lr=.001, momentum=.9, batch_size=32,
                                           patience=2, max_epochs=100,
                                           verbose=0):
    """perform diversity experiment for a single train size (subsampled variety)"""

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
            construct_train_val_by_class_subsampled(x_train, y_train, gist_train, train_size, val_size)

        diverse_accuracy, diverse_loss = experiment(x_diverse, y_diverse,
                                                    x_val, y_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'diverse accuracy: {diverse_accuracy}')
        similar_accuracy, similar_loss = experiment(x_diverse, y_diverse,
                                                    x_val, y_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'similar accuracy: {similar_accuracy}')
        random_accuracy, random_loss = experiment(x_diverse, y_diverse,
                                                  x_val, y_val,
                                                  x_test, y_test,
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


def diversity_experiment_subsampled(x_train, y_train,
                                    gist_train,
                                    x_test, y_test,
                                    train_sizes,
                                    val_size=1000,
                                    runs=10,
                                    lr=.0001, momentum=.9, batch_size=32,
                                    patience=2, max_epochs=100,
                                    verbose=0):
    """perform diversity experiment for a range of train sizes (subsampled variant)"""

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    results_df = []

    for train_size in train_sizes:
        print(f'performing experiment for train size={train_size}')

        sub_df = diversity_experiment_single_subsampled(x_train, y_train,
                                                        gist_train,
                                                        x_test, y_test,
                                                        train_size=train_size, val_size=val_size, runs=runs,
                                                        lr=lr, momentum=momentum, batch_size=batch_size,
                                                        patience=patience, max_epochs=max_epochs,
                                                        verbose=verbose)
        results_df.append(sub_df)

    results_df = pd.concat(results_df)

    return results_df


def diversity_experiment_single_edited(x_train, y_train,
                                       gist_train,
                                       x_test, y_test,
                                       remove_ind,
                                       train_size=100, val_prop=.5,
                                       runs=10,
                                       lr=.001, momentum=.9, batch_size=32,
                                       patience=2, max_epochs=100,
                                       verbose=0):
    """perform diversity experiment for a single train size using edited data"""

    # run experiments for the three datasets and repeat
    diverse_accuracies = np.empty(runs)
    diverse_losses = np.empty(runs)
    similar_accuracies = np.empty(runs)
    similar_losses = np.empty(runs)
    random_accuracies = np.empty(runs)
    random_losses = np.empty(runs)
    for run in np.arange(runs):
        # construct the training sets
        x_diverse, y_diverse, x_similar, y_similar, x_random, y_random, _, _ = \
            construct_train_val_by_class_edited(x_train, y_train, gist_train, remove_ind, train_size * 2, 0)

        # set aside some training data as validation data
        x_diverse_train, y_diverse_train, x_diverse_val, y_diverse_val = \
            split_train_val(x_diverse, y_diverse, val_prop)
        x_similar_train, y_similar_train, x_similar_val, y_similar_val = \
            split_train_val(x_similar, y_similar, val_prop)
        x_random_train, y_random_train, x_random_val, y_random_val = \
            split_train_val(x_random, y_random, val_prop)

        # run the three experiments
        diverse_accuracy, diverse_loss = experiment(x_diverse_train, y_diverse_train,
                                                    x_diverse_val, y_diverse_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'diverse accuracy: {diverse_accuracy}')
        similar_accuracy, similar_loss = experiment(x_similar_train, y_similar_train,
                                                    x_similar_val, y_similar_val,
                                                    x_test, y_test,
                                                    lr, momentum, batch_size,
                                                    patience, max_epochs,
                                                    verbose)
        # print(f'similar accuracy: {similar_accuracy}')
        random_accuracy, random_loss = experiment(x_random_train, y_random_train,
                                                  x_random_val, y_random_val,
                                                  x_test, y_test,
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


def diversity_experiment_edited(x_train, y_train,
                                gist_train,
                                x_test, y_test,
                                remove_ind,
                                train_sizes,
                                val_prop,
                                runs=10,
                                lr=.001, momentum=.9, batch_size=32,
                                patience=2, max_epochs=100,
                                verbose=0):
    """perform diversity experiment for a range of train sizes and nonrandom sampled validation set"""

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    results_df = []

    for train_size in train_sizes:
        print(f'performing experiment for train size = {train_size}')
        sub_df = diversity_experiment_single_edited(x_train, y_train,
                                                    gist_train,
                                                    x_test, y_test,
                                                    remove_ind,
                                                    train_size=train_size, val_prop=val_prop, runs=runs,
                                                    lr=lr, momentum=momentum, batch_size=batch_size,
                                                    patience=patience, max_epochs=max_epochs,
                                                    verbose=verbose)
        results_df.append(sub_df)

    results_df = pd.concat(results_df)

    return results_df


def diversity_experiment_kmeans(x_train, y_train,
                                gist_train,
                                x_test, y_test,
                                train_sizes,
                                edit_indices=None,
                                n_clust=5,
                                val_prop=.5,
                                runs=10,
                                lr=.001, momentum=.9, batch_size=32,
                                patience=2, max_epochs=100,
                                n_jobs=4,
                                verbose=0):
    """experiment where samples are drawn according to kmeans clustering"""

    # remove outliers
    if edit_indices is not None:
        x_train = np.delete(x_train, edit_indices, 0)
        y_train = np.delete(y_train, edit_indices, 0)
        gist_train = np.delete(gist_train, edit_indices, 0)

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    # number of classes
    k = len(np.unique(y_train))

    random_accs = []
    random_losses = []
    random_by_cluster_accs = []
    random_by_cluster_losses = []
    diverse_by_cluster_accs = []
    diverse_by_cluster_losses = []

    # assign clusters by class
    x_train, y_train, gist_train, clusters = assign_clusters_by_class(x_train, y_train, gist_train, n_clust, n_jobs)

    for train_size in train_sizes:
        print(f'performing experiment for train size = {train_size}')
        y = np.repeat(np.arange(k), int(train_size / val_prop)).reshape(k * int(train_size / val_prop), 1)
        for _ in np.arange(runs):
            # draw samples by method
            x_random = []
            x_random_by_cluster = []
            x_diverse_by_cluster = []
            for cls in np.arange(k):
                x_cls = x_train[y_train.flatten() == cls, :, :, :]
                gist_cls = gist_train[y_train.flatten() == cls, :]
                clust_cls = clusters[y_train.flatten() == cls]

                x_cls_random = x_cls[np.random.choice(x_cls.shape[0],
                                                      int(train_size / val_prop),
                                                      replace=False),
                                     :, :, :]

                x_cls_clust_diverse, _, _, _, x_cls_clust_random, _, _, _ = \
                    construct_train_val_by_class(x_cls, clust_cls, gist_cls, int(train_size / val_prop / n_clust), 0)

                x_random.append(x_cls_random)
                x_random_by_cluster.append(x_cls_clust_random)
                x_diverse_by_cluster.append(x_cls_clust_diverse)
            x_random = np.concatenate(x_random, 0)
            x_random_by_cluster = np.concatenate(x_random_by_cluster, 0)
            x_diverse_by_cluster = np.concatenate(x_diverse_by_cluster, 0)

            # split each sample into train/val
            x_random_train, y_random_train, x_random_val, y_random_val = split_train_val(x_random, y, val_prop)
            x_random_by_cluster_train, y_random_by_cluster_train, x_random_by_cluster_val, y_random_by_cluster_val = \
                split_train_val(x_random_by_cluster, y, val_prop)
            x_diverse_by_cluster_train, y_diverse_by_cluster_train, \
                x_diverse_by_cluster_val, y_diverse_by_cluster_val = \
                split_train_val(x_diverse_by_cluster, y, val_prop)

            # fit models and get test metrics
            random_acc, random_loss = experiment(x_random_train, y_random_train,
                                                 x_random_val, y_random_val,
                                                 x_test, y_test,
                                                 lr, momentum, batch_size,
                                                 patience, max_epochs,
                                                 verbose)
            random_by_cluster_acc, random_by_cluster_loss = experiment(x_random_by_cluster_train,
                                                                       y_random_by_cluster_train,
                                                                       x_random_by_cluster_val,
                                                                       y_random_by_cluster_val,
                                                                       x_test, y_test,
                                                                       lr, momentum, batch_size,
                                                                       patience, max_epochs,
                                                                       verbose)
            diverse_by_cluster_acc, diverse_by_cluster_loss = experiment(x_diverse_by_cluster_train,
                                                                         y_diverse_by_cluster_train,
                                                                         x_diverse_by_cluster_val,
                                                                         y_diverse_by_cluster_val,
                                                                         x_test, y_test,
                                                                         lr, momentum, batch_size,
                                                                         patience, max_epochs,
                                                                         verbose)
            random_accs.append(random_acc)
            random_losses.append(random_loss)
            random_by_cluster_accs.append(random_by_cluster_acc)
            random_by_cluster_losses.append(random_by_cluster_loss)
            diverse_by_cluster_accs.append(diverse_by_cluster_acc)
            diverse_by_cluster_losses.append(diverse_by_cluster_loss)

    results_df = pd.DataFrame({'train_size': np.repeat(np.tile(train_sizes, 3), runs),
                               'train_set_type': np.repeat(['random',
                                                            'random by cluster',
                                                            'diverse by cluster'],
                                                           runs * len(train_sizes)),
                               'accuracy': random_accs + random_by_cluster_accs + diverse_by_cluster_accs,
                               'loss': random_losses + random_by_cluster_losses + diverse_by_cluster_losses})
    return results_df


def assign_clusters_by_class(x_train, y_train, gist_train, n_clusters=5, n_jobs=4):
    """split the data by class, then cluster each class"""

    classes = np.unique(y_train)

    x_out = []
    y_out = []
    gist_out = []
    clust_out = []

    for cls in classes:
        x_cls = x_train[y_train.flatten() == cls, :, :, :]
        gist_train_cls = gist_train[y_train.flatten() == cls, :]

        kmeans = cluster.KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit(gist_train_cls)

        x_out.append(x_cls)
        y_out.append(np.repeat(cls, x_cls.shape[0]))
        gist_out.append(gist_train_cls)
        clust_out.append(kmeans.labels_)

    x_out = np.concatenate(x_out, 0)
    y_out = np.concatenate(y_out, 0)
    y_out = y_out.reshape(len(y_out), 1)
    gist_out = np.concatenate(gist_out, 0)
    clust_out = np.concatenate(clust_out, 0)

    return x_out, y_out, gist_out, clust_out


def diversity_experiment_one_obs_per_cluster(x_train, y_train,
                                             gist_train,
                                             x_test, y_test,
                                             train_sizes,
                                             edit_indices=None,
                                             val_prop=.5,
                                             runs=10,
                                             lr=.001, momentum=.9, batch_size=32,
                                             patience=2, max_epochs=100,
                                             n_jobs=4,
                                             verbose=0):
    """experiment where observations are each drawn from their own clusters"""

    # remove outliers
    if edit_indices is not None:
        x_train = np.delete(x_train, edit_indices, 0)
        y_train = np.delete(y_train, edit_indices, 0)
        gist_train = np.delete(gist_train, edit_indices, 0)

    # normalize pixel values
    x_train = x_train / 255.
    x_test = x_test / 255.

    # number of classes
    k = len(np.unique(y_train))

    random_accs = []
    random_losses = []
    by_cluster_accs = []
    by_cluster_losses = []
    for train_size in train_sizes:
        print(f'performing experiment for train size = {train_size}')
        # one cluster per draw
        x_train_clust, y_train_clust, gist_clust, clusters = \
            assign_clusters_by_class(x_train, y_train, gist_train, train_size, n_jobs)

        y = np.repeat(np.arange(k), int(train_size / val_prop)).reshape(k * int(train_size / val_prop), 1)
        for _ in np.arange(runs):
            # draw samples by method
            x_random = []
            x_cluster = []
            for cls in np.arange(k):
                x_cls = x_train[y_train.flatten() == cls, :, :, :]
                clust_cls = clusters[y_train.flatten() == cls]

                # draw random sample
                x_cls_random = x_cls[np.random.choice(x_cls.shape[0],
                                                      int(train_size / val_prop),
                                                      replace=False),
                                     :, :, :]

                # draw one sample per cluster
                x_cls_cluster = []
                for clust in np.arange(train_size):
                    x_cluster_cls = x_cls[clust_cls == clust, :, :, :]
                    x_cls_cluster.append(x_cluster_cls[np.random.choice(x_cluster_cls.shape[0],
                                                                        int(1 / val_prop),
                                                                        replace=False),
                                                       :, :, :])
                x_cls_cluster = np.concatenate(x_cls_cluster, 0)

                x_random.append(x_cls_random)
                x_cluster.append(x_cls_cluster)
            x_random = np.concatenate(x_random, 0)
            x_cluster = np.concatenate(x_cluster, 0)

            # split each sample into train/val
            x_random_train, y_random_train, x_random_val, y_random_val = split_train_val(x_random, y, val_prop)
            x_cluster_train, y_cluster_train, x_cluster_val, y_cluster_val = split_train_val(x_cluster, y, val_prop)

            # fit models and get test metrics
            random_acc, random_loss = experiment(x_random_train, y_random_train,
                                                 x_random_val, y_random_val,
                                                 x_test, y_test,
                                                 lr, momentum, batch_size,
                                                 patience, max_epochs,
                                                 verbose)
            cluster_acc, cluster_loss = experiment(x_cluster_train, y_cluster_train,
                                                   x_cluster_val, y_cluster_val,
                                                   x_test, y_test,
                                                   lr, momentum, batch_size,
                                                   patience, max_epochs,
                                                   verbose)
            random_accs.append(random_acc)
            random_losses.append(random_loss)
            by_cluster_accs.append(cluster_acc)
            by_cluster_losses.append(cluster_loss)

    results_df = pd.DataFrame({'train_size': np.repeat(np.tile(train_sizes, 2), runs),
                               'train_set_type': np.repeat(['random',
                                                            'one image per cluster'],
                                                           runs * len(train_sizes)),
                               'accuracy': random_accs + by_cluster_accs,
                               'loss': random_losses + by_cluster_losses})

    return results_df
