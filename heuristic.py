import pulp
import numpy as np
from astar_implementation import heuristic_gurobi
SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1
STD_TAU_COST = 0
STD_SYNC_COST = 0
import time


def compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_dict, x_0, t_index):
    k = len(split_dict) - 1
    if k == 0:
        return heuristic_gurobi.ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec)
    split_dict = dict(sorted(split_dict.items(), key=lambda item: item[1]))
    split_lst = list(split_dict.keys())[1:]
    place_num = len(incidence_matrix)
    trans_num = len(incidence_matrix[0])

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

    # constraint 1
    marking_diff = np.array(fin_vec) - np.array(ini_vec) - np.dot(incidence_matrix, x_0) \
                   - np.dot(incidence_matrix, var_y.sum(axis=0))
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
    dict1 = {'heuristic': int(pulp.value(prob.objective)) + np.dot(np.transpose(cost_vec), x_0),
             'var': [[int(pulp.value(var[i][j])) for j in range(trans_num)] for i in range(k)]}
    return dict1['heuristic'], np.array(dict1['var']).sum(axis=0) + x_0 + var_y.sum(axis=0), pulp.LpStatus[prob.status]


def ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec):
    start_time = time.time()
    marking_diff = np.array(fin_vec) - np.array(ini_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    trans_num = len(cost_vec)
    place_num = len(fin_vec)
    var = np.array([pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(trans_num)])
    prob += pulp.lpDot(cost_vec, var)
    ct1 = np.dot(incidence_matrix, var)
    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])
    prob.solve()
    print(time.time() - start_time)
    dict1 = {'heuristic': int(pulp.value(prob.objective)),
             'var': [int(pulp.value(var[i])) for i in range(trans_num)]}
    return dict1['heuristic'], np.array(dict1['var']),pulp.LpStatus[prob.status]


# compute heuristic of marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    var = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(cost_vec))]
    prob += pulp.lpDot(cost_vec, var)
    var1 = np.dot(inc_matrix, np.array(var))
    for i in range(len(ini_vec)):
        prob += (pulp.lpSum(var1[i]) == marking_diff[i])
    prob.solve()
    dict1 = {'heuristic': int(pulp.value(prob.objective)),
             'var': [int(pulp.value(var[i])) for i in range(len(cost_vec))]}
    return dict1['heuristic'], dict1['var']


def compute_estimate_heuristic(h_score, solution_x, t_index, cost_vec):
    result_aux = [0 for x in cost_vec]
    not_trust = 1
    print(solution_x)
    if solution_x[t_index] >= 1:
        # print("t index", t_index)
        not_trust = 0
    result_aux[t_index] = 1
    new_solution_x = np.array(solution_x) - np.array(result_aux)
    new_h_score = max(0, h_score - cost_vec[t_index])
    return new_h_score, new_solution_x, not_trust
