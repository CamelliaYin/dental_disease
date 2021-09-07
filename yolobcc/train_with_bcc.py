import train
import detect
import numpy as np
from lib.BCCNet.VariationalInference import VB_iteration_yolo

def read_crowdsourced_labels(data_name):
    pass

def get_x_paths(data_name):
    pass

def read_results(path):
    pass

def update_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch):
    pass

def read_labels(path):
    pass

def compute_param_confusion_matrices(bcc_params):
    pass

def plot_results(n_epoch, metrics):
    pass

def init_bcc_params():
    pass

def init_metrics(n_epochs):
    pass

def train_with_bcc(hyp, opt, device):
    y_crowdsourced = read_crowdsourced_labels(opt.data)
    x_train_path, x_test_path = get_x_paths(opt.data)
    y_train, y_test = read_labels(opt.data)
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
        q_t, pcm['variational'], lower_bound = \
            VB_iteration_yolo(y_crowdsourced, yhat_train, pcm['variational'], pcm['prior'])
        
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
