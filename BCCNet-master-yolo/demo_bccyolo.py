#  Copyright (c) 2019. University of Oxford

from re import S
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

from NNArchitecture.lenet5_mnist import cnn_for_mnist
from NNArchitecture import yolov5
from SyntheticCrowdsourcing.synthetic_crowd_volunteers import generate_volunteer_labels
from VariationalInference.VB_iteration_yolo import VB_iteration
from utils.utils_dataset_processing import shrink_arrays
from VariationalInference import confusion_matrix

rseed = 1000
np.random.seed(rseed)
tf.random.set_seed(rseed)


def set_params(mode = 'mnist'):
    # Set parameters
    if mode == 'mnist':
        params = {'n_classes': 10,
                'crowdsourced_labelled_train_data_ratio': 0.5,
                'n_crowd_members': 4,
                'crowd_member_reliability_level': 0.6,
                'confusion_matrix_diagonal_prior': 1e-1,
                'n_epoch': 100,
                'batch_size': 32,
                'convergence_threshold': 1e-6}
    elif mode == 'dental':
        params = {'n_classes': 2,
                # 'crowdsourced_labelled_train_data_ratio': 0.5,
                # 'n_crowd_members': 4,
                # 'crowd_member_reliability_level': 0.6,
                'confusion_matrix_diagonal_prior': 1e-1,
                'n_epoch': 100,
                'batch_size': 32,
                'convergence_threshold': 1e-6}
    return params


def prepare_data(x_train, x_test):
    # expand images for a cnn
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    return x_train, x_test


def simulate_crowdsourcing(x_train, y_train, params):
    # select subsample of train data to be "labelled" by crowd members
    labelled_train, whole_train = shrink_arrays(
        [x_train, y_train], params['crowdsourced_labelled_train_data_ratio'], is_shuffle=True)
    # x_labelled_train = labelled_train[0]
    y_labelled_train = labelled_train[1]
    x_train = whole_train[0]
    y_train = whole_train[1]

    # generate synthetic crowdsourced labels
    crowdsourced_labels = generate_volunteer_labels(n_volunteers=params['n_crowd_members'], n_classes=params['n_classes'], gt_labels=y_labelled_train,
                                                    n_total_tasks=x_train.shape[0],
                                                    reliability_level=params['crowd_member_reliability_level'])
    return crowdsourced_labels


def load_and_prepare_all_data(params):
    # load data
    if params['mode'] == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
            path=os.getcwd() + '/mnist.npz')
        x_train, x_test = prepare_data(x_train, x_test)
        crowdsourced_labels = simulate_crowdsourcing(
            x_train, y_train, x_test, y_test, params)
    elif params['mode'] == 'dental':
        train_data, test_data = yolov5.load_data('radiographs')
        crowdsourced_labels = None
        x_train, y_train = train_data
        x_test, y_test = test_data
        x_train, y_train, x_test, y_test, crowdsourced_labels        
    return x_train, y_train, x_test, y_test, crowdsourced_labels


def compute_param_confusion_matrices(params):
    # set up variational parameters
    prior_param_confusion_matrices = confusion_matrix.initialise_prior(n_classes=params['n_classes'], n_volunteers=params['n_crowd_members'],
                                                                       alpha_diag_prior=params['confusion_matrix_diagonal_prior'])
    variational_param_confusion_matrices = np.copy(
        prior_param_confusion_matrices)
    return {'prior': prior_param_confusion_matrices, 'variational': variational_param_confusion_matrices}


def init_nn_output(x_train, params):
    # initial variational inference iteration (initialisation of approximating posterior of true labels)
    nn_output_0 = np.random.randn(x_train.shape[0], params['n_classes'])
    return nn_output_0


def init_metrics(n_epochs):
    metrics = {'accuracy': np.zeros((n_epochs,), dtype=np.float64)}
    return {'train': metrics, 'test': metrics, 'posterior_estimate': metrics}


def train_one_nn_epoch(model, x_train, q_t, batch_size):
    model.fit(x_train, q_t, epochs=1, shuffle=True,
                      batch_size=batch_size, verbose=0)


def nn_predict(model, x):
    # update of approximating posterior for the true labels and confusion matrices
    # get current predictions from a neural network
    nn_output_for_vb_update = model.predict(x)
    # for numerical stability
    nn_output_for_vb_update = nn_output_for_vb_update - \
        np.tile(np.expand_dims(np.max(nn_output_for_vb_update, axis=1),
                axis=1), (1, nn_output_for_vb_update.shape[1]))
    return nn_output_for_vb_update


def update_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch, verbose=True):
    metrics['train']['accuracy'][epoch] = np.mean(np.argmax(yhat_train, axis=1) == y_train)
    metrics['posterior_estimate']['accuracy'][epoch] = np.mean(np.argmax(q_t, axis=1) == y_train)
    metrics['test']['accuracy'][epoch] = np.mean(np.argmax(yhat_test, axis=1) == y_test)
    if verbose:
        print(f"\t nn training accuracy: {metrics['train']['accuracy'][epoch]}")
        print(f"\t posterior estimate training accuracy: {metrics['posterior_estimate']['accuracy'][epoch]}")
        print(f"\t nn test accuracy: {metrics['test']['accuracy'][epoch]}")

def plot_results(n_epoch, metrics):
    # plotting
    plt.plot(range(n_epoch), metrics['train']['accuracy'][:n_epoch], label='nn train')
    plt.plot(range(n_epoch), metrics['posterior_estimate']['accuracy'][:n_epoch], label='posterior train')
    plt.plot(range(n_epoch), metrics['test']['accuracy'][:n_epoch], label='nn test')
    plt.legend()
    plt.show()

def main():
    params = set_params(mode='mnist')
    x_train, y_train, x_test, y_test, crowdsourced_labels = load_and_prepare_all_data(params)

    model = yolov5.get_model()

    pcm = compute_param_confusion_matrices(params)

    # for each object we have 10 logits coresponding to 10 classes
    nn_output_0 = init_nn_output(x_train, params)
    q_t, pcm['variational'], old_lower_bound = VB_iteration(
        crowdsourced_labels, nn_output_0, pcm['variational'], pcm['prior'])

    metrics = init_metrics(params['n_epochs'])

    # main cycle of training
    for epoch in range(params['n_epoch']):
        print(f'epoch {epoch}:')

        train_one_nn_epoch(model, x_train, q_t, params['batch_size'])
        yhat_train = nn_predict(model, x_train)
        
        q_t, pcm['variational'], lower_bound = VB_iteration(
            crowdsourced_labels, yhat_train, pcm['variational'], pcm['prior'])
    
        # evaluation
        yhat_test = nn_predict(model, x_test)
        update_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch)

        # check convergence
        if np.abs((lower_bound - old_lower_bound) / old_lower_bound) < params['convergence_threshold']:
            break

        old_lower_bound = lower_bound

    # save weights
    model.save_weights(os.getcwd() + '/trained_weights')
    plot_results(params['n_epoch'])

if __name__ == '__main__':
    main()
