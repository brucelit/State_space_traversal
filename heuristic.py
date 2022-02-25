import re
import sys
import timeit
from copy import deepcopy

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.util.lp import solver as lp_solver
from cvxopt import matrix
import pulp

# def get_ini_heuristic(ini_vec, fin_vec, cost, split_lst,
#                       incidence_matrix,
#                       consumption_matrix,
#                       t_index, p_index,
#                       trace_lst_sync,
#                       trace_lst_log):
#     k = len(split_lst)
#     place_num = len(incidence_matrix)
#     trans_num = len(incidence_matrix[0])
#     # define problem
#     prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
#
#     # define x_i from x_0 to x_k, with k+1 x_i
#     var_x = np.array([[pulp.LpVariable(f'x{i}_{j}', lowBound=0)
#                        for j in range(trans_num)] for i in range(k + 1)])
#
#     # define y_i from 0 to k
#     var_y = np.array([[pulp.LpVariable(f'y{i}_{j}', lowBound=0, upBound=1)
#                        for j in range(trans_num)] for i in range(k + 1)])
#
#     # constraint 1
#     marking_diff = np.array(fin_vec) - np.array(ini_vec)
#     ct1 = np.dot(incidence_matrix, var_x.sum(axis=0)) + np.dot(incidence_matrix, var_y.sum(axis=0))
#     for j in range(place_num):
#         prob.addConstraint(
#             pulp.LpConstraint(
#                 e=ct1[j],
#                 sense=pulp.LpConstraintEQ,
#                 rhs=marking_diff[j]))
#
#     # constraint 2
#     a = 1
#     cons_two = np.array(ini_vec)
#     while a < k + 1:
#         var2 = np.array([0 for i in cost])
#         for b in range(1, a):
#             var2 = var2 + var_x[b] + var_y[b]
#         var2 = var2 + var_x[0]
#         ct2 = np.dot(incidence_matrix, var2) + np.dot(consumption_matrix, var_y[a])
#         for j in range(place_num):
#             prob.addConstraint(
#                 pulp.LpConstraint(
#                     e=ct2[j],
#                     sense=pulp.LpConstraintGE,
#                     rhs=-1 * cons_two[j]))
#         a += 1
#
#     # constraint 5,6
#     y_col = 1
#     prob += np.sum(var_y[0]) == 0
#     for i in split_lst:
#         # if len(trace_lst_sync[i]) > 0 and trace_lst_log[i] is not None:
#         #     y_index = 0
#         #     for each_one in trace_lst_sync[i]:
#         #         y_index += var_y[y_col][each_one]
#         #     y_index += var_y[y_col][trace_lst_log[i]]
#         #     # [(var_y[y_col][trace_lst_sync[i][j]], 1) for j in (range(len(trace_lst_sync)),
#         #     #                                                    (var_y[y_col][trace_lst_log[i]], 1))]
#         #     prob += pulp.LpAffineExpression(y_index) == 1
#         # else:
#         #     prob += pulp.LpAffineExpression([(var_y[y_col][trace_lst_log[i]], 1)]) == 1
#         prob += pulp.lpSum(var_y[y_col]) == 1
#         y_col += 1
#
#     # add objective
#     costs = np.array([cost for i in range(2 * k + 2)])
#     x = np.concatenate((var_x, var_y), axis=0)
#     objective = pulp.lpSum(x[i, j] * costs[i, j]
#                            for j in range(trans_num)
#                            for i in range(2 * k + 2))
#     prob.setObjective(objective)
#     prob.solve()
#     # print(np.array([[int(pulp.value(var_y[i][j])) for j in range(trans_num)] for i in range(k + 1)]))
#     if pulp.LpStatus[prob.status] == "Optimal":
#         dict1 = {'heuristic': int(pulp.value(prob.objective)),
#                  'var_x': [[int(pulp.value(var_x[i][j])) for j in range(trans_num)] for i in range(k + 1)],
#                  'var_y': [[int(pulp.value(var_y[i][j])) for j in range(trans_num)] for i in range(k + 1)]}
#         return dict1['heuristic'], np.array(dict1['var_x']).sum(axis=0) + np.array(dict1['var_y']).sum(axis=0)
#     else:
#         print("sb了")
#         return 0, 0
#
def get_ini_heuristic(ini_vec, fin_vec, cost, split_lst,
                      incidence_matrix,
                      consumption_matrix,
                      t_index, p_index,
                      trace_lst_sync,
                      trace_lst_log):
    # Define problem
    m = gp.Model()
    m.Params.LogToConsole = 0

    # Create two 2-D arrays of integer variables X and Y, 0 to k+1
    k = len(split_lst)
    x = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)
    y = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)

    # Set objective
    m.setObjective(sum(np.array(cost) @ x[i, :] + np.array(cost) @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)

    # Add constraint 1
    cons_one = np.array([0 for i in range(len(p_index))])
    for i in range(k + 1):
        sum_x = incidence_matrix @ x[i, :]
        sum_y = incidence_matrix @ y[i, :]
        cons_one += sum_x + sum_y
    m.addConstr(cons_one == fin_vec - ini_vec)

    # Add constraint 2
    cons_two_temp = incidence_matrix @ x[0, :]
    # cons_two = ini_vec
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                cons_two_temp += incidence_matrix @ x[b, :] + incidence_matrix @ y[b, :]
        ct2 = cons_two_temp + consumption_matrix @ y[a, :]
        m.addConstr(ct2+ini_vec >= 0)

    # Add constraint 5 and 6:
    y_col = 1
    m.addConstr(y[0, :].sum() == 0)
    for i in split_lst:
        if len(trace_lst_sync[i]) > 0 and trace_lst_log[i] is not None:
            y_index = 0
            for j in trace_lst_sync[i]:
                y_index += y[y_col, j]
            y_index += y[y_col, trace_lst_log[i]]
            m.addConstr(y_index == 1)
        else:
            k_1 = trace_lst_log[i]
            m.addConstr(y[y_col, k_1] == 1)
        m.addConstr(y[y_col, :].sum() == 1)
        y_col += 1

    # optimize model
    m.optimize()
    # print("solution vec:", np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0))
    return m.objVal, np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0)


# compute the exact heuristic of marking m
def get_exact_heuristic(marking_diff, incidence_matrix, cost_vec):
    m = gp.Model()
    m.Params.LogToConsole = 0
    x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
    z = np.array(incidence_matrix) @ x[0, :]
    m.addConstr(z == marking_diff)
    m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
    m.optimize()
    # print("compute exact h", m.objVal)
    return m.objVal, np.array(x.X.sum(axis=0))


# compute the exact heuristic of marking m
def get_exact_heuristic_new(marking, split_lst, marking_diff, ini, incidence_matrix, cost_vec, max_rank, trace_len):
    insert_position = 1
    rank = max_rank + 1
    if marking.m != ini:
        if rank + 1 not in split_lst:
            insert_position = -1
    if marking.m == ini or insert_position > 0 or rank + 1 >= trace_len:
        m = gp.Model()
        m.Params.LogToConsole = 0
        x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
        z = np.array(incidence_matrix) @ x[0, :]
        m.addConstr(z == marking_diff)
        m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
        m.optimize()
        r = get_max_events(marking)
        if r > max_rank:
            max_rank = r
        if m.status == 2:
            return m.objVal, np.array(x.X.sum(axis=0)), True, split_lst, max_rank
        else:
            return "HEURISTICINFINITE", [], "Infeasible", split_lst, max_rank
    else:
        split_lst.append(rank + 1)
        return -1, [0 for i in range(len(cost_vec))], 0, split_lst, max_rank


def get_max_events(marking):
    if marking.t is None:
        return -2
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
    return get_max_events(marking.p)
