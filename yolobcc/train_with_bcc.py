import train
from utils.general import check_dataset
import detect
import numpy as np
import os
from lib.BCCNet.VariationalInference import VB_iteration_yolo
from lib.BCCNet.VariationalInference import confusion_matrix
from matplotlib import pyplot as plt

def read_crowdsourced_labels(data_name):
    data_dict = check_dataset(data_name)
    cs_root_path = os.path.join('{}_crowdsourced'.format(data_dict['path']), 'labels')
    users = os.listdir(cs_root_path)
    modes = ['train', 'test', 'val']
    y_crowdsourced = {m: [] for m in modes}
    for u in users:
        for m in modes:
            path = os.path.join(cs_root_path, m)
            for file_name in os.listdir(path):
                with open(file_name, 'r') as f:
                    img_labels = f.readlines()
                y_crowdsourced[m].append(img_labels)
    y_crowdsourced = {k: np.array(v) for k, v in y_crowdsourced.items()}
    return y_crowdsourced

def get_x_paths(data_name):
    data_dict = check_dataset(data_name)
    return data_dict['train'], data_dict['val'], data_dict['test']

def read_results(path):
    with open(path, 'r') as f:
        results = np.array(f.readlines())
    return results

def update_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch, verbose = True):
    metrics['train']['accuracy'][epoch] = np.mean(np.argmax(yhat_train, axis=1) == y_train)
    metrics['posterior_estimate']['accuracy'][epoch] = np.mean(np.argmax(q_t, axis=1) == y_train)
    metrics['test']['accuracy'][epoch] = np.mean(np.argmax(yhat_test, axis=1) == y_test)
    if verbose:
        print(f"\t nn training accuracy: {metrics['train']['accuracy'][epoch]}")
        print(f"\t posterior estimate training accuracy: {metrics['posterior_estimate']['accuracy'][epoch]}")
        print(f"\t nn test accuracy: {metrics['test']['accuracy'][epoch]}")

def read_labels(data_name):
    data_dict = check_dataset(data_name)
    modes = ['train', 'test', 'val']
    labels = {m: [] for m in modes}
    for m in modes:
        with open(data_dict[m]) as f:
            labels[m].append(f.readlines())
    labels = {k: np.array(v) for k, v in labels.items()}
    return labels['train'], labels['val'], labels['test']


def compute_param_confusion_matrices(bcc_params):
    # set up variational parameters
    prior_param_confusion_matrices = confusion_matrix.initialise_prior(n_classes=bcc_params['n_classes'],
                                                                       n_volunteers=bcc_params['n_crowd_members'],
                                                                       alpha_diag_prior=bcc_params['confusion_matrix_diagonal_prior'])
    variational_param_confusion_matrices = np.copy(prior_param_confusion_matrices)
    return {'prior': prior_param_confusion_matrices, 'variational': variational_param_confusion_matrices}

def plot_results(n_epoch, metrics):
    plt.plot(range(n_epoch), metrics['train']['accuracy'][:n_epoch], label='nn train')
    plt.plot(range(n_epoch), metrics['posterior_estimate']['accuracy'][:n_epoch], label='posterior train')
    plt.plot(range(n_epoch), metrics['test']['accuracy'][:n_epoch], label='nn test')
    plt.legend()
    plt.show()

def init_bcc_params():
    bcc_params = {'n_classes': 2,
                  'n_crowd_members': 4,
                  'confusion_matrix_diagonal_prior': 1e-1}
    return bcc_params

def init_metrics(n_epochs):
    metrics = {'accuracy': np.zeros((n_epochs,), dtype=np.float64)}
    return {'train': metrics, 'test': metrics, 'posterior_estimate': metrics}

def convert_to_logits(output):
    pass

def train_with_bcc(hyp, opt, device):
    y_crowdsourced = read_crowdsourced_labels(opt.data)
    x_train_path, _, x_test_path = get_x_paths(opt.data)
    y_train, _, y_test = read_labels(opt.data)
    bcc_epochs = opt.epochs
    opt.epochs = 1
    bcc_params = init_bcc_params()
    pcm = compute_param_confusion_matrices(bcc_params)
    metrics = init_metrics(bcc_epochs)
    for epoch in range(bcc_epochs):
        print(f'epoch: {epoch}')
        train.train(hyp, opt, device)
        opt.weights = opt.save_dir / 'weights' / 'last.pt'
        yhat_train_txt_path = detect.run(weights = opt.weights, source = x_train_path, save_txt=True, save_conf=True)
        yhat_train = read_results(yhat_train_txt_path)
        yhat_train_logits = convert_to_logits(yhat_train)
        q_t, pcm['variational'], lower_bound = \
            VB_iteration_yolo(y_crowdsourced['train'], yhat_train_logits, pcm['variational'], pcm['prior'])
        
        yhat_test_txt_path = detect.run(weights = opt.weights, source = x_test_path, save_text = True, save_conf = True)
        yhat_test = read_results(yhat_test_txt_path)
        update_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch)

        if np.abs((lower_bound - old_lower_bound) / old_lower_bound) < bcc_params['convergence_threshold']:
            break

        old_lower_bound = lower_bound
    plot_results(bcc_epochs)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = train.parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    train.main(opt)

if __name__ == "__main__":
    run(bcc=True, data='toy.yaml')
