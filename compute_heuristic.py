import cvxpy as cp
import numpy as np
from astar_implementation import utilities
from scipy.optimize import linprog


def test_estimated_heuristic(incidence_matrix,consumption_matrix,place_index,trans_index,sync_net,cost_vec,sync_im,sync_fm):
    #
    incidence_matrix, place_index, trans_index = utilities.construct_incident_matrix(sync_net)
    t_index = 0
    for i in trans_index:
        str1 = str(i)
        if str1 == "('a', 'a')":
            print(trans_index[i])
            t_index = trans_index[i]
    b_matrix = np.asmatrix(consumption_matrix.a_matrix).astype(np.float64)
    sub_matrix = [1.0 for x in cost_vec]
    sub_matrix[t_index] = 0
    one_matrix = [1.0 for x in cost_vec]
    marking_difference = np.asarray(utilities.encode_marking(sync_fm,place_index)) - np.asarray(utilities.encode_marking(sync_im,place_index))
    ini_marking = utilities.encode_marking(sync_im,place_index)
    ini_marking = [-1.0*x for x in ini_marking]
    # 先写个k=1的.那么就有x1和y1
    obj = []
    lhs_eq = []
    rhs_eq = []

    #obj的需要两个cost——vec
    obj.append(cost_vec)
    obj.append(cost_vec)

    #lhs_eq有四个个，两个inci，一个one_matrix
    eq1 = [incidence_matrix,incidence_matrix]
    eq2 = [0,b_matrix]
    eq3 = [0,one_matrix]
    eq4 = [0,sub_matrix]
    lhs_eq.append(eq1)
    lhs_eq.append(eq2)
    lhs_eq.append(eq3)
    lhs_eq.append(eq4)

    #
    rhs_eq.append(marking_difference)
    rhs_eq.append(ini_marking)
    rhs_eq.append(0)
    rhs_eq.append(1)
    opt = linprog(c=obj, A_ub=lhs_eq, b_ub=rhs_eq,
                  A_eq=lhs_eq, b_eq=rhs_eq, method="revised simplex")
    print(opt.success)