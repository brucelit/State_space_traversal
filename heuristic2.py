import pulp
import numpy as np
import time
from lp_maker import *

SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1
STD_TAU_COST = 0
STD_SYNC_COST = 0


def compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_dict, x_0, t_index):
    print(split_dict)
    k = len(split_dict) - 1
    if k == 0:
        return ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec)
    split_dict = dict(sorted(split_dict.items(), key=lambda item: item[1]))
    split_lst = list(split_dict.keys())[1:]
    place_num = len(incidence_matrix)
    trans_num = len(incidence_matrix[0])
    print("inc mat", incidence_matrix)
    # define problem
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)

    # define x_i from x_1 to x_k
    var = np.array([[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger)
                     for j in range(trans_num)]
                    for i in range(k)])
    costs = np.array([cost_vec for i in range(k)])

    # add objective
    prob += pulp.lpDot(costs.flatten(), var[0:k].flatten())
    var_y = np.zeros((k, trans_num))
    for i in range(0, k):
        temp = split_lst[i]
        trans_index = t_index[temp]
        var_y[i][trans_index] = 1

    # constraint 1    # align = astar_v2.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
    marking_diff = np.array(fin_vec) - np.array(ini_vec) - np.dot(incidence_matrix, x_0) \
                   - np.dot(incidence_matrix, var_y.sum(axis=0))
    print("marking diff", marking_diff)
    var1 = np.array(var).sum(axis=0)
    ct1 = np.dot(incidence_matrix, var1)
    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])

    # constraint 2
    for a in range(0, k):
        cons_two = np.array(ini_vec) + np.dot(incidence_matrix, x_0) + np.dot(consumption_matrix, var_y[a])
        var2 = np.array([0 for i in cost_vec])
        for b in range(0, a):
            var2 = var2 + var[b] + var_y[b]
        ct2 = np.dot(incidence_matrix, var2)
        for i in range(place_num):
            prob += (pulp.lpSum(ct2[i]) >= -1 * cons_two[i])
    prob.solve()
    print(pulp.LpStatus[prob.status])
    dict1 = {'heuristic': int(pulp.value(prob.objective)) + np.dot(np.transpose(cost_vec), x_0),
             'var': [[int(pulp.value(var[i][j])) for j in range(trans_num)] for i in range(k)]}
    return dict1['heuristic'], np.array(dict1['var']).sum(axis=0) + x_0 + var_y.sum(axis=0), pulp.LpStatus[prob.status]

#
def ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec):
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    ineq = np.zeros(len(incidence_matrix))
    ineq2 = [i for i in range(1, len(ini_vec) + 1)]
    lp = lp_maker(cost_vec, incidence_matrix, marking_diff, ineq, setminim=1)
    o1 = np.ones(len(cost_vec))
    lpsolve('set_int', lp, o1)
    lpsolve("solve", lp)
    obj = lpsolve('get_objective', lp)
    var = lpsolve('get_variables', lp)[0]
    lpsolve('delete_lp',lp)
    return obj, var, "Optimal"


# compute heuristic of marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    ineq = np.zeros(len(inc_matrix))
    # ineq2 = [i for i in range(1, len(ini_vec)+1)]
    lp = lp_maker(cost_vec, inc_matrix, marking_diff, ineq, setminim=1)
    o1 = np.ones(len(cost_vec))
    lpsolve('set_int', lp, o1)
    lpsolve("solve", lp)
    obj = lpsolve('get_objective', lp)
    var = lpsolve('get_variables', lp)[0]
    lpsolve('delete_lp',lp)
    return obj,var

def compute_estimate_heuristic(h_score, solution_x, t_index, cost_vec):
    result_aux = [0 for x in cost_vec]
    not_trust = 1
    if solution_x[t_index] >= 1:
        not_trust = 0
    result_aux[t_index] = 1
    new_solution_x = np.array(solution_x) - np.array(result_aux)
    new_h_score = max(0, h_score - cost_vec[t_index])
    return new_h_score, new_solution_x, not_trust

