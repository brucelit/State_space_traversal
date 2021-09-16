import time
import timeit

import pulp
import numpy as np
from lp_maker import *
import gurobipy as gp
from gurobipy import GRB


def compute_ini_heuristic(ini_vec, fin_vec, cost, incidence_matrix,
                          consumption_matrix, split_lst, t_index, p_index,
                          trace_lst_log, trace_lst_sync, set_model_move=set()):
    k = len(split_lst) - 1
    split_lst = sorted(split_lst[1:])
    split_lst.append(len(trace_lst_log))
    place_num = len(p_index)
    trans_num = len(t_index)
    marking_diff = fin_vec - ini_vec

    # define problem
    m = gp.Model("matrix")
    m.Params.LogToConsole = 0

    # Create a 2-D array of integer variables, 0 to k+1
    x = m.addMVar((k + 1, trans_num), vtype=GRB.INTEGER, lb=0)
    y = m.addMVar((k + 1, trans_num), vtype=GRB.INTEGER, lb=0)

    # Set objective
    m.setObjective(sum(cost @ x[i, :] + cost @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)

    # add constraint 1
    z = np.array([0 for i in range(place_num)])
    for i in range(k + 1):
        sum_x = incidence_matrix @ x[i, :]
        sum_y = incidence_matrix @ y[i, :]
        z += sum_x + sum_y
    m.addConstr(z == marking_diff, "cs1")

    # add constraint 2
    a = 1
    ct2_1 = incidence_matrix @ x[0, :]
    cons_two = ini_vec
    while a < k + 1:
        for b in range(1, a):
            if b == a - 1:
                ct2_1 += incidence_matrix @ x[b, :] + incidence_matrix @ y[b, :]
        ct2 = ct2_1 + consumption_matrix @ y[a, :]
        m.addConstr(ct2 >= -1 * cons_two)
        a += 1

    # add constraint 5 and 6:
    y_col = 1
    a = y[0, :].sum()
    m.addConstr(a == 0)
    for i in split_lst[:-1]:
        if trace_lst_sync[i] is not None and trace_lst_log[i] is not None:
            k_1 = trace_lst_sync[i]
            k_2 = trace_lst_log[i]
            m.addConstr(y[y_col, k_1] + y[y_col, k_2] == 1)
        elif trace_lst_sync[i] is None and trace_lst_log[i] is not None:
            k_1 = trace_lst_log[i]
            m.addConstr(y[y_col, k_1] == 1)
        a = y[y_col, :].sum()
        m.addConstr(a == 1)
        y_col += 1
    # Optimize model
    m.optimize()

    if m.status == 2:
        dict1 = {'heuristic': m.objVal,
                 'var_x': x.X,
                 'var_y': y.X}
        return dict1['heuristic'], \
               np.array(dict1['var_x']).sum(axis=0) + np.array(dict1['var_y']).sum(axis=0), \
               "Optimal"
    else:
        return 0, 0, "Infeasible"


# compute heuristic of marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    marking_diff = fin_vec - ini_vec
    m = gp.Model("matrix")
    m.Params.LogToConsole = 0
    x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
    z = np.array(inc_matrix) @ x[0, :]
    m.addConstr(z == marking_diff)
    m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
    m.optimize()
    dict1 = {'heuristic': m.objVal,
             'var': np.array(x.X.sum(axis=0))}
    return dict1['heuristic'], dict1['var']


# computer estimated heuristic from heuristics of previous marking
def compute_estimate_heuristic(h_score, solution_x, t_index, cost_vec):
    result_aux = [0 for x in cost_vec]
    not_trust = 1
    if solution_x[t_index] >= 1:
        not_trust = 0
    result_aux[t_index] = 1
    new_solution_x = np.array(solution_x) - np.array(result_aux)
    new_h_score = max(0, h_score - cost_vec[t_index])
    return new_h_score, new_solution_x, not_trust
