import torch
MAX_BBS_PER_IMAGE_FALLBACK = 10

def filter_qt(qt_yolo, qt_thres_mode, qt_thres, conf, qt=None, torchMode = False, device=None):
    raise Exception("Filters have been disabled since they weren't tested. We have moved to the 3-class idea (background-based)")
    if qt_thres_mode == '':
        return qt_yolo
    n_images = conf.shape[0]
    if qt_thres_mode == 'conf-val':
        conf_thres = qt_thres
        return filter_qt_conf_val(qt_yolo, conf, conf_thres)
    if qt_thres_mode == 'conf-count':
        count_thres = qt_thres
        return filter_qt_conf_count(qt_yolo, conf, count_thres)
    if qt_thres_mode == 'entropy':
        entropy_thres = qt_thres
        return filter_qt_entropy(qt_yolo, qt, entropy_thres)
    if qt_thres_mode.startswith('hybrid'):
        entropy_thres, conf_thres = qt_thres
        return filter_qt_hybrid(qt_yolo, qt, entropy_thres, conf_thres)
    raise Exception("Filter not implemented yet.")
    

def print_diff(orig, filt):
    n_total_bb = orig.shape[0]
    n_remaining_bb = filt.shape[0]
    n_lost_bb = n_total_bb - n_remaining_bb
    p_remaining_bb = n_remaining_bb * 100 / n_total_bb
    p_lost_bb = 100 - p_remaining_bb
    print(f'{n_lost_bb} ({p_lost_bb}%) of {n_total_bb} bounding boxes lost in qt-filtering')

def filter_qt_conf_count(qt_yolo, conf, count_thres, silent = False):
    if count_thres <= 0:
        raise Exception("Count-threshod for qt-filtering has to be a positive integer")
    n_images, n_boxes = conf.shape
    top_indices = conf.sort(descending=True).indices[:, :count_thres]
    indices = top_indices + torch.tensor([[i*n_boxes for i in range(n_images)]]).transpose(1, 0)
    indices = indices.reshape(indices.shape[0]*indices.shape[1])
    filtered_qt_yolo = qt_yolo[indices, :]
    if not silent:
        print_diff(qt_yolo, filtered_qt_yolo)
    return filtered_qt_yolo

def filter_qt_conf_val(qt_yolo, conf, conf_thres, silent = False):
    if conf_thres <= 0 or conf_thres >= 1:
        raise Exception("Confidence-threshod for qt-filtering has to be an proper positive fraction")
    filtered_qt_yolo = qt_yolo[(conf >= conf_thres).reshape(qt_yolo.shape[0])]
    if not silent:
        print_diff(qt_yolo, filtered_qt_yolo)
    if filtered_qt_yolo.shape[0] > 0:
        return filtered_qt_yolo
    print(f"No bounding boxes left after conf-val>={conf_thres} filter. Resorting to top {MAX_BBS_PER_IMAGE_FALLBACK}.")
    filtered_qt_yolo = filter_qt_conf_count(qt_yolo, conf, MAX_BBS_PER_IMAGE_FALLBACK, silent=True)
    return filtered_qt_yolo

def filter_qt_entropy(qt_yolo, qt, entropy_thres):
    if qt is None:
        raise Exception("qtargets are None. Need probability information to compute entropies")