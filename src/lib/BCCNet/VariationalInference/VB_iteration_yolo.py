import scipy.special as ss
import numpy as np
import torch
import pdb
# import pdb

def torch_max_fun(t1, t2):
    if not torch.is_tensor(t1):
        t1 = torch.tensor(t1)
    if not torch.is_tensor(t2):
        t2 = torch.tensor(t2)
    return torch.max(t1, t2)

def VB_iteration(X, nn_output, alpha_volunteers, alpha0_volunteers, torchMode=False, device=None, invert_classes = False):
    """
    performs one iteration of variational inference update for BCCNet (E-step)
    -- update for approximating posterior of true labels and confusion matrices
    I - number of data points
    M - number of true classes
    N - number of classes used by volunteers (normally M == N)
    K - number of volunteers (W)
    :param X: I X U X V X K volunteers answers, for image i, the grid choice u, the vth anchor box, the kth volunteer,
              -1 encodes a missing answer (where the volunteer can not identify abnormality there.)
    :param nn_output: (I x U X V) x M logits (not a softmax output!) note here the nn_output is only the partial output
                      from the object detection NN.
    :param alpha_volunteers: M X N X K - current parameters of posterior Dirichlet for confusion matrices
    :param alpha0_volunteers: M X N -  parameters of the prior Dirichlet for confusion matrix
    :return: q_t - approximating posterior for true labels, alpha_volunteers - updated posterior for confusion matrices,
        lower_bound_likelihood - ELBO
    """
    # torchMode = all([torch.is_tensor(x) for x in [X, nn_output, alpha_volunteers, alpha0_volunteers]])

    # pdb.set_trace()
    ElogPi_volunteer = expected_log_Dirichlet_parameters(alpha_volunteers, torchMode, device=device)

    # q_t
    q_t, Njl, rho = expected_true_labels(X, nn_output, ElogPi_volunteer, torchMode, device=device)

    # q_pi_workers
    alpha_volunteers = update_alpha_volunteers(alpha0_volunteers, Njl, torchMode, device=device)

    # Low bound
    lower_bound_likelihood = compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output, torchMode)

    if invert_classes:
        q_t_copy = q_t.clone()
        q_t[:,:,0] = q_t_copy[:,:,1]
        q_t[:,:,1] = q_t_copy[:,:,0]
    return q_t, alpha_volunteers, lower_bound_likelihood

# part of computing loss

torch_module_map = {'base_lib': torch,
                    'gammaln': torch.special.gammaln, 
                    'digamma': torch.digamma,
                    'simple_transpose': lambda x: torch.permute(x, tuple(range(x.ndim)[::-1])),
                    'copy': lambda x: x.clone().detach(),
                    'maxwithdim': lambda x, d: torch.max(x, d).values,
                    'maximum': lambda t1, t2: torch_max_fun(t1, t2)
                    }

np_module_map = {'base_lib': np,
                 'gammaln': ss.gammaln,
                 'digamma': ss.psi,
                 'simple_transpose': np.transpose,
                 'copy': np.copy,
                 'maxwithdim': np.max,
                 'maximum': np.maximum
                 }

def get_modules(torchMode=False, module_names=None):
    module_names = module_names or ['base_lib']
    module_map = torch_module_map if torchMode else np_module_map
    modules = [module_map[x] for x in module_names]
    return modules

# part of computing loss
def logB_from_Dirichlet_parameters(alpha, torchMode=False):
    base_lib, gammaln_fn = get_modules(torchMode, ['base_lib', 'gammaln'])
    logB = base_lib.sum(gammaln_fn(alpha)) - gammaln_fn(base_lib.sum(alpha))
    return logB


def expected_log_Dirichlet_parameters(param, torchMode=False, device=None):
    base_lib, digamma_fn, simple_transpose = get_modules(torchMode, ['base_lib', 'digamma', 'simple_transpose'])        
    size = param.shape
    result = base_lib.zeros_like(param)
    if torchMode:
        result = result.to(device)

    if len(size) == 1:
        result = digamma_fn(param) - digamma_fn(base_lib.sum(param))
    elif len(size) == 2:  # when we take A_0 for everyone
        result = digamma_fn(param) - simple_transpose(base_lib.tile(digamma_fn(base_lib.sum(param, 1)), (size[1], 1)))
    elif len(size) == 3:  # most of time for posterior cm
        for i in range(size[2]):
            result[:, :, i] = digamma_fn(param[:, :, i]) - \
                              simple_transpose(base_lib.tile(digamma_fn(base_lib.sum(param[:, :, i], 1)), (size[1], 1)))
    else:
        raise Exception('param can have no more than 3 dimensions')
    return result


def expected_true_labels(X, nn_output, ElogPi_volunteer, torchMode=False, device=None):
    base_lib, copy_fn, simple_transpose, maxwithdim_fn, maximum_fn = get_modules(torchMode, ['base_lib', 'copy', 'simple_transpose', 'maxwithdim', 'maximum'])
    I, U, K = X.shape  # I = no. of image, U = no. of anc hor boxes in total, K = no. of volunteers
    M = ElogPi_volunteer.shape[0]  # M = Number of classes
    N = ElogPi_volunteer.shape[1]  # N = Number of classes used by volunteers

    rho = copy_fn(nn_output)  # I x U x M logits
    # eq. 12:
    for k in range(K):
        inds = tuple([x.long() for x in base_lib.where(X[:, :, k] > -1)])  # rule out missing values
        rho[inds[0], inds[1], :] += simple_transpose(
            ElogPi_volunteer[:, base_lib.squeeze(X[inds[0], inds[1], k]).long(), k])

    # normalisation: (minus the max of each anchor)
    rho -= simple_transpose(base_lib.tile(simple_transpose(maxwithdim_fn(rho, 2)), (M, 1, 1)))

    # # eq. 11:
    q_t = base_lib.exp(rho) / maximum_fn(1e-60, simple_transpose(base_lib.tile(simple_transpose(base_lib.sum(base_lib.exp(rho), 2)), (M, 1, 1))))
    q_t = maximum_fn(1e-60, q_t)

    # partial of eq. 8: (right side 2nd term)
    f_iu = base_lib.zeros((M, N, K), dtype=base_lib.float64)
    if torchMode:
        f_iu = f_iu.to(device)
    for k in range(K):
        for n in range(N):
            ids0 = base_lib.where(X[:, :, k] == n)[0]
            ids1 = base_lib.where(X[:, :, k] == n)[1]
            f_iu[:, n, k] = base_lib.sum(q_t[ids0, ids1, :], 0)
    rho.shape, rho
    return q_t, f_iu, rho
# dim: (I x U x M), (M x N x K), (I x U x M)


# eq. 8:
def update_alpha_volunteers(alpha0_volunteers, f_iu, torchMode=False, device=None):
    (base_lib,) = get_modules(torchMode, ['base_lib'])
    K = alpha0_volunteers.shape[2]
    alpha_volunteers = base_lib.zeros_like(alpha0_volunteers)
    if torchMode:
        alpha_volunteers = alpha_volunteers.to(device)

    # pdb.set_trace()
    for k in range(K):
        alpha_volunteers[:, :, k] = alpha0_volunteers[:, :, k] + f_iu[:, :, k]

    return alpha_volunteers


def compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output, torchMode=False):
    (base_lib,) = get_modules(torchMode, ['base_lib'])
    
    W = alpha0_volunteers.shape[2]

    ll_pi_worker = 0
    for w in range(W):
        ll_pi_worker -= base_lib.sum(logB_from_Dirichlet_parameters(alpha0_volunteers[:, :, w], torchMode=torchMode) -
                                             logB_from_Dirichlet_parameters(alpha_volunteers[:, :, w], torchMode=torchMode))

    ll_t = -base_lib.sum(q_t * rho) + base_lib.sum(base_lib.log(base_lib.sum(base_lib.exp(rho), axis=1)), axis=0)

    ll_nn = base_lib.sum(q_t * nn_output) - base_lib.sum(base_lib.log(base_lib.sum(base_lib.exp(nn_output), axis=1)), axis=0)

    ll = ll_pi_worker + ll_t + ll_nn  # VB lower bound

    return base_lib.sum(ll)