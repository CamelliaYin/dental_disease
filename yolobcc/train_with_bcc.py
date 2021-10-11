from label_converter import BACKGROUND_CLASS_ID
from label_converter import init_yolo_labels, yolo2bcc
# import train
from torchvision.ops import nms
from utils.general import xywh2xyxy
from utils.general import check_dataset, check_file, increment_path
import detect
import numpy as np
import os
from pathlib import Path
from lib.BCCNet.VariationalInference import VB_iteration_yolo
from lib.BCCNet.VariationalInference import confusion_matrix
from matplotlib import pyplot as plt
from utils.torch_utils import select_device
import torch

DEFAULT_G = np.array([1.0/32, 1.0/16, 1.0/8])


VOL_ID_MAP = {'Camellia': 0, 'Conghui': 1, 'HaoWen': 2, 'Xiongjie': 3}
SINGLE_VOL_ID_MAP = {'Camellia': 0}

def perform_nms_filtering(batch_qtargets_yolo, batch_qtargets, nms_thres = 0.45):
    n_images = batch_qtargets.shape[0]
    per_im_tgts = []
    for i in range(n_images):
        y1 = batch_qtargets_yolo[batch_qtargets_yolo[:, 0] == i]
        orig_im_count = y1.shape[0]
        y = batch_qtargets[i]
        inds = torch.argmax(y, axis=1) != 2
        y = y[inds, :]
        y = torch.cat([y1, y[:, [-1]]], axis=1)
        c = 1
        per_cl_tgts = []
        for c in range(2):
            z = y[y[:, 1] == c, :]
            orig_im_cl_count = z.shape[0]
            inds = nms(xywh2xyxy(z[:, 2:6]), z[:,-1], nms_thres)
            z = z[inds, :]
            per_cl_tgts.append(z)
        per_im_tgts.extend(per_cl_tgts)
    filtered_batch_qtargets_yolo = torch.cat(per_im_tgts, axis=0)
    return filtered_batch_qtargets_yolo

def get_file_volunteers_dict(data_dict, mode='train', vol_id_map=VOL_ID_MAP):
    vol_path = os.path.join(data_dict['path'], 'volunteers')
    vol_mode_path = os.path.join(vol_path, mode)
    file_names = [x for x in os.listdir(vol_mode_path) if not x.startswith('.')]
    file_vols_dict = {}
    for fn in file_names:
        vol_file_path = os.path.join(vol_mode_path, fn)
        with open(vol_file_path) as f:
            vol_seq = [vol_id_map[x.strip()] for x in f.readlines()]
        file_vols_dict[fn] = torch.tensor(vol_seq).int()
    return file_vols_dict

def init_nn_output(n_train, G, n_anchor_choices, params, bkgd=False):
    # bkgd for "background".
    # initial variational inference iteration (initialisation of approximating posterior of true labels)
    # try:
    #     count = x_train.shape[0]
    # except AttributeError:
    #     count = len(x_train)
    # TODO: might need to use expanded (grid- and anchor-based) n_train.
    num_grid_cells = int(((1/G) * (1/G)).sum())
    n_effective_grid_cells = num_grid_cells * n_anchor_choices
    n_effective_classes = params['n_classes'] + (1 if bkgd else 0)
    nn_output_0 = np.random.randn(n_train, n_effective_grid_cells, n_effective_classes)
    return nn_output_0

def read_crowdsourced_labels(data):
    if isinstance(data, str):
        data_dict = check_dataset(data)
    elif isinstance(data, dict):
        data_dict = data
    cs_root_path = os.path.join('{}_crowdsourced'.format(data_dict['path']), 'labels')
    users = [x for x in os.listdir(cs_root_path) if (not x.startswith('.') and x != 'all')]
    modes = ['train', 'test', 'val']
    y_crowdsourced = {m: [] for m in modes}
    for m in modes:
        image_file_prefixes = [os.path.splitext(n)[0] for n in \
        os.listdir(os.path.join('{}_crowdsourced'.format(data_dict['path']), 'images', m)) \
            if not n.startswith('.')]
        y_per_mode = []
        for u in users:
            y_per_mode_per_user = []
            path = os.path.join(cs_root_path, u, m)
            for file_prefix in image_file_prefixes:
                label_file_name = f'{file_prefix}.txt'
                label_file_path = os.path.join(path, label_file_name)
                try:
                    img_labels = np.loadtxt(label_file_path)
                except OSError:
                    img_labels = np.array([-1, 0, 0, 0, 0])
                if img_labels.ndim == 1:
                    img_labels = np.expand_dims(img_labels, axis = 0)
                # with open(os.path.join(path, file_name), 'r') as f:
                #     img_labels = f.readlines()
                y_per_mode_per_user.append(img_labels)
            y_per_mode.append(y_per_mode_per_user)
        y_crowdsourced[m] = y_per_mode
    return y_crowdsourced

def get_x_paths(data_name):
    data_dict = check_dataset(data_name)
    return data_dict['train'], data_dict['val'], data_dict['test']

def read_results(path):
    
    with open(path, 'r') as f:
        results = np.array(f.readlines())
    return results

def update_bcc_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch, verbose = True):
    metrics['train']['accuracy'][epoch] = np.mean(np.argmax(yhat_train, axis=1) == y_train)
    metrics['posterior_estimate']['accuracy'][epoch] = np.mean(np.argmax(q_t, axis=1) == y_train)
    metrics['test']['accuracy'][epoch] = np.mean(np.argmax(yhat_test, axis=1) == y_test)
    if verbose:
        print(f"\t nn training accuracy: {metrics['train']['accuracy'][epoch]}")
        print(f"\t posterior estimate training accuracy: {metrics['posterior_estimate']['accuracy'][epoch]}")
        print(f"\t nn test accuracy: {metrics['test']['accuracy'][epoch]}")

def read_labels(data_name):
    data_dict = check_dataset(data_name)
    labels_root_path = os.path.join(data_dict['path'], 'labels')
    modes = ['train', 'test', 'val']
    labels = {m: [] for m in modes}
    for m in modes:
        path = os.path.join(labels_root_path, m)
        for file_name in os.listdir(path):
            if file_name.startswith('.'):
                    continue
            with open(os.path.join(path, file_name)) as f:
                img_labels = np.loadtxt(os.path.join(path, file_name))
                labels[m].append(img_labels)
    # labels = {k: np.array(v) for k, v in labels.items()}
    return labels['train'], labels['val'], labels['test']

def compute_param_confusion_matrices(bcc_params, torchMode=False):
    # set up variational parameters
    prior_param_confusion_matrices = confusion_matrix.initialise_prior(n_classes=bcc_params['n_classes'],
                                                                       n_volunteers=bcc_params['n_crowd_members'],
                                                                       alpha_diag_prior=bcc_params['confusion_matrix_diagonal_prior'],
                                                                       torchMode = torchMode)
    variational_param_confusion_matrices = prior_param_confusion_matrices.detach().clone() if torchMode else np.copy(prior_param_confusion_matrices)
    return {'prior': prior_param_confusion_matrices, 'variational': variational_param_confusion_matrices}

def plot_results(n_epoch, metrics):
    plt.plot(range(n_epoch), metrics['train']['accuracy'][:n_epoch], label='nn train')
    plt.plot(range(n_epoch), metrics['posterior_estimate']['accuracy'][:n_epoch], label='posterior train')
    plt.plot(range(n_epoch), metrics['test']['accuracy'][:n_epoch], label='nn test')
    plt.legend()
    plt.show()

def init_bcc_params(K=4):
    bcc_params = {'n_classes': 3,
                  'n_crowd_members': K,
                  'confusion_matrix_diagonal_prior': 1e-1,
                  'convergence_threshold': 1e-6}
    return bcc_params

def init_metrics(n_epochs):
    metrics = {'accuracy': np.zeros((n_epochs,), dtype=np.float64)}
    return {'train': metrics, 'test': metrics, 'posterior_estimate': metrics}

def convert_to_logits(yolo_output):
    n_batches = len(yolo_output)
    for b in range(n_batches):
        y_b = yolo_output[b]
        n_images_per_batch = len(y_b)
        for i in range(n_images_per_batch):
            y_i = y_b[i]

def convert_yolo2bcc(y_yolo, Na, Nc, G, intermediate_yolo_mode = False, torchMode = False):
    y_bcc = []
    n_images = len(y_yolo)
    for i in range(n_images):
        y_image_yolo = y_yolo[i]
        yolo_labels = {'labels': init_yolo_labels(y_image_yolo, Na, G, torchMode = torchMode), 'G': G, 'Nc': Nc}
        bcc_labels = yolo2bcc(yolo_labels, intermediate_yolo_mode = intermediate_yolo_mode, torchMode = torchMode)
        y_image_bcc = bcc_labels['labels']
        y_bcc.append(y_image_bcc)
    y_bcc = torch.tensor(y_bcc) if torchMode else np.array(y_bcc)
    return y_bcc

def convert_cs_yolo2bcc_with_vol_list(targets, Na=3, Nc=2, G=DEFAULT_G, intermediate_yolo_mode = False, volunteer_list = None):
    pass

def convert_cs_yolo2bcc_wo_vol_list(y_cs_yolo, Na=3, Nc=2, G=DEFAULT_G, intermediate_yolo_mode = False):
    modes = ['train', 'val', 'test']
    y_cs_bcc = {}
    for m in modes:
        y_yolo = y_cs_yolo[m]
        n_users = len(y_yolo)
        y_bcc = []
        for u in range(n_users):
            y_per_user_yolo = y_yolo[u]
            y_per_user_bcc = convert_yolo2bcc(y_per_user_yolo, Na, Nc, G, intermediate_yolo_mode=intermediate_yolo_mode)
            y_bcc.append(y_per_user_bcc)
        y_cs_bcc[m] = np.array(y_bcc).transpose(1, 2, 0)# (I x UV x K)
    return y_cs_bcc

def convert_cs_yolo2bcc(y_cs_yolo, Na=3, Nc=2, G=DEFAULT_G, intermediate_yolo_mode = False, volunteer_list = None):
    if volunteer_list is None:
        return convert_cs_yolo2bcc_wo_vol_list(y_cs_yolo, Na, Nc, G, intermediate_yolo_mode)
    return convert_cs_yolo2bcc_with_vol_list(y_cs_yolo, Na, Nc, G, intermediate_yolo_mode, volunteer_list)

def convert_target_volunteers_yolo2bcc(target_volunteers, Na=3, Nc=2, G=DEFAULT_G, batch_size=None, vol_id_map=VOL_ID_MAP):
    n_images = batch_size
    n_vols = len(vol_id_map)
    Ng = G.shape[0]

    vigcwh_list = [] # [v]olunteer, [i]mage, [g]rid choice, grid [c]ell id, [w]idth, [h]eight    
    targets_per_i_bcc_list = []
    for i in range(n_images):
        target_vols_per_i = target_volunteers[target_volunteers[:, 0] == i][:, 1:]
        targets_per_ig_bcc_list = []
        for g in range(Ng):  # per grid choice
            g_frac = G[g]
            S_g = np.ceil(1/g_frac).astype(int)**2
            # Don't need a loop for anchor-boxes as we are simply repeating Na times below.
            targets_per_iv_bcc_list = []
            for v in range(n_vols):
                targets_per_iv = target_vols_per_i[target_vols_per_i[:, -1] == v][:, :-1]
                c, x, y, w, h = targets_per_iv.T # w and h are ignored

                x_cell_ids = torch.where(x<1, x/g_frac, torch.ones(x.shape)*(np.ceil(1/g_frac))).int()
                y_cell_ids = torch.where(y<1, y/g_frac, torch.ones(y.shape)*(np.ceil(1/g_frac))).int()
                gc_ids = ((y_cell_ids)*(np.ceil(1/g_frac)) + x_cell_ids).long()
                vigcwh_list.append(torch.cat([torch.tensor([v, i, g])*torch.ones(w.shape[0], 3), gc_ids.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], axis=1))
                
                targets_per_iv_bcc = BACKGROUND_CLASS_ID * torch.ones(S_g)
                targets_per_iv_bcc[gc_ids] = c
                targets_per_iv_bcc_list.append(targets_per_iv_bcc)
            targets_per_iga_bcc = torch.stack(tuple(targets_per_iv_bcc_list)).T
            targets_per_ig_bcc = targets_per_iga_bcc.repeat((Na, 1))
            targets_per_ig_bcc_list.append(targets_per_ig_bcc)
        targets_per_i_bcc = torch.cat(targets_per_ig_bcc_list)
        targets_per_i_bcc_list.append(targets_per_i_bcc)
    target_volunteers_bcc = torch.stack(tuple(targets_per_i_bcc_list))
    vigcwh = torch.cat(vigcwh_list)
    return target_volunteers_bcc, vigcwh

def xywhpc1ck_to_cxywh(y):
    y[:, 4] = y[:, 5:].max(dim=1).indices
    y = y[:, :5]
    y = y[:, [4, 0, 1, 2, 3]]
    return y

def nn_predict(model, x, imgsz, offgrid_translate_flag=True, normalize_flag=True, transform_format_flag=True):
    # update of approximating posterior for the true labels and confusion matrices
    # get current predictions from a neural network
    y = model(x)[0]
    if offgrid_translate_flag:
        # TODO: A quickfix here is to force-translate any point lying outside the grid to the nearest grid cell.
        y[..., :2][y[..., :2] < 0] = 0
        y[..., :2][y[..., :2] > imgsz] = imgsz
    if normalize_flag:
        y[..., :2] = y[..., :2] / imgsz
    if transform_format_flag: # Transform (x, y, w, h, prob, c1, ..., ck) to (c, x, y, w, h)
        z = y[..., :5]
        n_images = y.shape[0]
        for i in range(n_images):
            z[i] = xywhpc1ck_to_cxywh(y[i])
        y = z
    return y


def train_with_bcc(hyp, opt, device):
    # THIS IS AN UNUSABLE FUNCTION. TO BE USED ONLY WHEN DOUBLE-CHECKED.
    y_crowdsourced_yolo = read_crowdsourced_labels(opt.data)
    x_train_path, x_val_path, x_test_path = get_x_paths(opt.data)
    x_paths = {'train': x_train_path, 'test': x_test_path, 'val': x_val_path}
    y_crowdsourced_bcc = convert_yolo2bcc(y_crowdsourced_yolo)
    y_train, _, y_test = read_labels(opt.data)
    bcc_epochs = opt.epochs
    opt.epochs = 1
    bcc_params = init_bcc_params()
    pcm = compute_param_confusion_matrices(bcc_params)
    metrics = init_metrics(bcc_epochs)
    for epoch in range(bcc_epochs):
        print(f'epoch: {epoch}')
        # train.train(hyp, opt, device)
        opt.weights = os.path.join(opt.save_dir, 'weights', 'last.pt')
        yhat_train = detect.run(weights = opt.weights, source = x_train_path, save_txt=True, save_conf=True, nosave=True, nonms=False, conf_thres=0, iou_thres=0)
        # yhat_train = read_results(yhat_train_txt_path)
        yhat_train_logits = convert_to_logits(yhat_train)
        q_t, pcm['variational'], lower_bound = \
            VB_iteration_yolo(y_crowdsourced_bcc['train'], yhat_train_logits, pcm['variational'], pcm['prior'])
        
        yhat_test = detect.run(weights = opt.weights, source = x_test_path, save_text = True, save_conf = True, nosave=True, nonms=False, conf_thres=0, iou_thres=0)
        # yhat_test = read_results(yhat_test_txt_path)
        update_bcc_metrics(metrics, q_t, yhat_train, y_train, yhat_test, y_test, epoch)

        if np.abs((lower_bound - old_lower_bound) / old_lower_bound) < bcc_params['convergence_threshold']:
            break

        old_lower_bound = lower_bound
    plot_results(bcc_epochs)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    # opt = train.parse_opt(True)
    # for k, v in kwargs.items():
    #     setattr(opt, k, v)
    # # opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
    # opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    # device = select_device(opt.device, batch_size=opt.batch_size)
    # train_with_bcc(opt.hyp, opt, device)
    pass

if __name__ == "__main__":
    run(bcc=True, data='dental_disease/yolobcc/data/toy.yaml',
        hyp='dental_disease/yolobcc/data/hyps/hyp.scratch.yaml')
