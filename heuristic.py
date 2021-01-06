import pulp
import numpy as np
from scipy.optimize import linprog
from astar_implementation import initialization

import numpy as np
from pm4py.util.lp import solver as lp_solver
from pm4py.objects.petri.petrinet import Marking
from pm4py.objects.petri import semantics
from copy import copy
import sys
from cvxopt import matrix


SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1
STD_TAU_COST = 0
STD_SYNC_COST = 0


def compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix, consumption_matrix, split_dict, x_0, t_index, sync_net, aux_dict, marking):
    k = len(split_dict) - 1
    print("init marking", ini_vec)
    print("fin marking", fin_vec)
    # print("\ncurrent split:", k)
    if k == 0:
        return ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec)
    split_dict = dict(sorted(split_dict.items(), key=lambda item: item[1]))
    split_lst = list(split_dict.keys())[1:]
    # print("current split point:", split_lst)
    print("check split list", split_lst)
    place_num = len(incidence_matrix)
    trans_num = len(incidence_matrix[0])
    # define problem
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)

    # define x_i from x_1 to x_k
    var = np.array([[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger)
                     for j in range(trans_num)]
                    for i in range(k)])
    costs = np.array([cost_vec for i in range(k)])
    # print("costs: ", costs)
    # add objective
    prob += pulp.lpDot(costs.flatten(), var[0:k].flatten())
    # print("cost of x_0: ",np.dot(np.transpose(cost_vec), x_0))
    var_y = np.zeros((k, trans_num))
    for i in range(0, k):
        temp = split_lst[i]
        trans_index = t_index[temp]
        var_y[i][trans_index] = 1

    print("x_0: ", x_0)
    # print("y:", var_y)

    # constraint 1
    marking_diff = np.array(fin_vec) - np.array(ini_vec) - np.dot(incidence_matrix, x_0) \
                   - np.dot(incidence_matrix, var_y.sum(axis=0))
    # print("marking_diff",marking_diff)
    var1 = np.array(var).sum(axis=0)
    # print("var1:", var1)
    ct1 = np.dot(incidence_matrix, var1)
    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])

    # constraint 2
    for a in range(0, k):
        cons_two = np.array(ini_vec) + np.dot(incidence_matrix, x_0) + np.dot(consumption_matrix, var_y[a])
        print("cons_two:", cons_two)
        # print("round: ", a)
        var2 = np.array([0 for i in cost_vec])
        for b in range(0, a):
            var2 = var2 + var[b] + var_y[b]
            print("var2 define:\n", var2)
        ct2 = np.dot(incidence_matrix, var2)
        # print("ct2 define:\n", ct2)
        for i in range(place_num):
            prob += (pulp.lpSum(ct2[i]) >= -1 * cons_two[i])
    prob.solve()

    dict1 = {'heuristic': int(pulp.value(prob.objective)) + np.dot(np.transpose(cost_vec), x_0),
             'var': [[int(pulp.value(var[i][j])) for j in range(trans_num)] for i in range(k)]}
    print("status", pulp.LpStatus[prob.status])
    print("var_y", var_y)
    # print("x_0",x_0)
    # print("solution vec:", np.array(dict1['var']).sum(axis=0) + x_0 + var_y.sum(axis=0))
    # print("init solution vec", np.array(dict1['var']).sum(axis=0) + x_0 + var_y.sum(axis=0))
    return dict1['heuristic'], np.array(dict1['var']).sum(axis=0) + x_0 + var_y.sum(axis=0)


def ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec):
# def ini_heuristic_without_split(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
#     marking, fin_vec, variant,aux_dict, use_cvxopt = False):
#     m_vec = initialization.encode_marking(marking, aux_dict['p_index'])
#     b_term = [i - j for i, j in zip(fin_vec, m_vec)]
#     b_term = np.array([x * 1.0 for x in b_term]).transpose()
#     b_term = matrix(b_term)
#     parameters_solving = {"solver": "glpk"}
#
#     sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving,
#                           variant=variant)
#     prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
#     points = lp_solver.get_points_from_sol(sol, variant=variant)
#
#     prim_obj = prim_obj if prim_obj is not None else sys.maxsize
#     points = points if points is not None else [0.0] * len(sync_net.transitions)
#     print("init h:", prim_obj)
#     print("init solution vec: ", points)
#     return prim_obj, points

    marking_diff = np.array(fin_vec) - np.array(ini_vec)
    # print("marking_diff", marking_diff)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    trans_num = len(cost_vec)
    place_num = len(fin_vec)
    var = np.array([pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(trans_num)])
    prob += pulp.lpDot(cost_vec, var)
    ct1 = np.dot(incidence_matrix, var)
    # print("ct1", ct1)
    # print("marking diff:", marking_diff)
    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])
    prob.solve()
    print("initial status", pulp.LpStatus[prob.status])
    dict1 = {'heuristic': int(pulp.value(prob.objective)),
             'var': [int(pulp.value(var[i])) for i in range(trans_num)]}
    print("init solution vec", np.array(dict1['var']))
    # print("init h", dict1['heuristic'])
    return dict1['heuristic'], np.array(dict1['var'])
    #lhs_eq有四个个，两个inci，一个one_matrix
    # c = np.array(cost_vec)
    # A = incidence_matrix
    # b = marking_diff
    # x0_bounds = (0, None)
    # res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds], method="revised simplex")
    # print(res)


# compute heuristic of marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    print("更新了")
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    var = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger)
           for i in range(len(cost_vec))]
    prob += pulp.lpDot(cost_vec, var)
    var1 = np.dot(inc_matrix, np.array(var))
    for i in range(len(ini_vec)):
        prob += (pulp.lpSum(var1[i]) == marking_diff[i])
    prob.solve()
    dict1 = {'heuristic': int(pulp.value(prob.objective)),
             'var': [int(pulp.value(var[i])) for i in range(len(cost_vec))]}
    # print("compute exact heuristic", dict1['var'], dict1['heuristic'])
    return dict1['heuristic'], dict1['var']


def compute_estimate_heuristic(h_score, solution_x, t_index, cost_vec):
    result_aux = [0 for x in cost_vec]
    not_trust = 1
    if solution_x[t_index] >= 1:
        # print("t index", t_index)
        not_trust = 0
    result_aux[t_index] = 1
    new_solution_x = np.array(solution_x) - np.array(result_aux)
    # print("t_index",t_index)
    # print("old", solution_x)
    # print("new", new_solution_x)
    new_h_score = max(0, h_score - cost_vec[t_index])
    return new_h_score, new_solution_x, not_trust
