import scipy.special as ss
import numpy as np


param = np.array([[1.1, 1], [1, 1.1]])

def expected_log_Dirichlet_parameters(param): #Eq.(16)
    # param: alpha_volunteers
    size = param.shape
    # shape:(J X L X W)

    result = np.zeros_like(param)

    if len(size) == 1:
        result = ss.psi(param) - ss.psi(np.sum(param))
        # ss.psi(): The logarithmic derivative of the gamma function evaluated at param
    elif len(size) == 2:
        result = ss.psi(param) - np.transpose(np.tile(ss.psi(np.sum(param, 1)), (size[1], 1)))
    elif len(size) == 3:
        for i in range(size[2]):
            result[:, :, i] = ss.psi(param[:, :, i]) - \
                              np.transpose(np.tile(ss.psi(np.sum(param[:, :, i], 1)), (size[1], 1)))
    else:
        raise Exception('param can have no more than 3 dimensions')

    return result

check_me = expected_log_Dirichlet_parameters(param)
print(check_me)




