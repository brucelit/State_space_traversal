import re
import sys
import timeit
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.util.lp import solver as lp_solver
from cvxopt import matrix

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
    x = m.addMVar((k + 1, len(t_index)), lb=0)
    y = m.addMVar((k + 1, len(t_index)), lb=0)

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
    cons_two = ini_vec
    for a in range(1, k + 1):
        for b in range(1, a):
            if b == a - 1:
                cons_two_temp += incidence_matrix @ x[b, :] + incidence_matrix @ y[b, :]
        ct2 = cons_two_temp + consumption_matrix @ y[a, :]
        m.addConstr(ct2 >= -1 * cons_two)

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
    return m.objVal, np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0)

# def get_ini_heuristic(ini_vec, fin_vec, cost, split_lst,
#                       incidence_matrix,
#                       consumption_matrix,
#                       t_index, p_index,
#                       trace_lst_sync,
#                       trace_lst_log):
#     # Define problem
#     m = gp.Model()
#     m.Params.LogToConsole = 0
#
#     # Create two 2-D arrays of integer variables X and Y, 0 to k+1
#     k = len(split_lst)
#     x = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)
#     y = m.addMVar((k + 1, len(t_index)), vtype=GRB.INTEGER, lb=0)
#
#     # Set objective
#     m.setObjective(sum(np.array(cost) @ x[i, :] + np.array(cost) @ y[i, :] for i in range(k + 1)), GRB.MINIMIZE)
#
#     # Add constraint 1
#     cons_one = np.array([0 for i in range(len(p_index))])
#     for i in range(k + 1):
#         sum_x = incidence_matrix @ x[i, :]
#         sum_y = incidence_matrix @ y[i, :]
#         cons_one += sum_x + sum_y
#     m.addConstr(cons_one == fin_vec - ini_vec)
#
#     # Add constraint 2
#     cons_two_temp = incidence_matrix @ x[0, :]
#     cons_two = ini_vec
#     for a in range(1, k + 1):
#         for b in range(1, a):
#             if b == a - 1:
#                 cons_two_temp += incidence_matrix @ x[b, :] + incidence_matrix @ y[b, :]
#         ct2 = cons_two_temp + consumption_matrix @ y[a, :]
#         m.addConstr(ct2 >= -1 * cons_two)
#
#     # Add constraint 5 and 6:
#     y_col = 1
#     m.addConstr(y[0, :].sum() == 0)
#     for i in split_lst:
#         if len(trace_lst_sync[i]) > 0 and trace_lst_log[i] is not None:
#             y_index = 0
#             for j in trace_lst_sync[i]:
#                 y_index += y[y_col, j]
#             y_index += y[y_col, trace_lst_log[i]]
#             m.addConstr(y_index == 1)
#         else:
#             k_1 = trace_lst_log[i]
#             m.addConstr(y[y_col, k_1] == 1)
#         m.addConstr(y[y_col, :].sum() == 1)
#         y_col += 1
#
#     # optimize model
#     m.optimize()
#     return m.objVal, np.array(x.X).sum(axis=0) + np.array(y.X).sum(axis=0)


# compute the exact heuristic of marking m
def get_exact_heuristic(marking_diff, incidence_matrix, cost_vec):
    m = gp.Model()
    m.Params.LogToConsole = 0
    x = m.addMVar((1, len(cost_vec)), lb=0)
    z = np.array(incidence_matrix) @ x[0, :]
    m.addConstr(z == marking_diff)
    m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
    m.optimize()
    # print("compute exact h", m.objVal)
    return m.objVal, np.array(x.X.sum(axis=0))

# compute the exact heuristic of marking m
# def get_exact_heuristic(marking_diff, incidence_matrix, cost_vec):
#     m = gp.Model()
#     m.Params.LogToConsole = 0
#     x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
#     z = np.array(incidence_matrix) @ x[0, :]
#     m.addConstr(z == marking_diff)
#     m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
#     m.optimize()
#     # print("compute exact h", m.objVal)
#     return m.objVal, np.array(x.X.sum(axis=0))

# def get_exact_heuristic(incidence_matrix, cost_vec, trans_len, fin_vec, m_vec):
#     a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
#     h_cvx = np.matrix(np.zeros(trans_len)).transpose()
#     g_matrix = -np.eye(trans_len)
#     cost_vec = [x * 1.0 for x in cost_vec]
#
#     use_cvxopt = True
#
#     if use_cvxopt:
#         a_matrix = matrix(a_matrix)
#         g_matrix = matrix(g_matrix)
#         h_cvx = matrix(h_cvx)
#         cost_vec = matrix(cost_vec)
#
#     h, x = __compute_exact_heuristic_new_version(trans_len, a_matrix, h_cvx, g_matrix, cost_vec,
#                                                  m_vec, fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT)
#     return h, x

# compute the exact heuristic of marking m
def get_exact_heuristic_new(marking, split_lst, marking_diff, ini, incidence_matrix, cost_vec, max_rank, trace_len):
    insert_position = 1
    rank = max_rank + 1
    if marking.m != ini:
        if rank+1 not in split_lst:
            insert_position = -1
    if marking.m == ini or insert_position > 0 or rank+1 >= trace_len:
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
        split_lst.append(rank+1)
        return -1, [0 for i in range(len(cost_vec))], 0, split_lst, max_rank

# def get_exact_heuristic_new(marking, m_vec, split_lst, fin_vec, ini, incidence_matrix, cost_vec, max_rank, trace_len, trans_len):
#     insert_position = 1
#     rank = max_rank + 1
#     if marking.m != ini:
#         if rank+1 not in split_lst:
#             insert_position = -1
#     if marking.m == ini or insert_position > 0 or rank+1 >= trace_len:
#         a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
#         h_cvx = np.matrix(np.zeros(trans_len)).transpose()
#         g_matrix = -np.eye(trans_len)
#         cost_vec = [x * 1.0 for x in cost_vec]
#
#         a_matrix = matrix(a_matrix)
#         g_matrix = matrix(g_matrix)
#         h_cvx = matrix(h_cvx)
#         cost_vec = matrix(cost_vec)
#
#         h,x = __compute_exact_heuristic_new_version(trans_len, a_matrix, h_cvx, g_matrix, cost_vec,
#                                           m_vec, fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT)
#         # m = gp.Model()
#         # m.Params.LogToConsole = 0
#         # x = m.addMVar((1, len(cost_vec)), vtype=GRB.INTEGER, lb=0)
#         # z = np.array(incidence_matrix) @ x[0, :]
#         # m.addConstr(z == marking_diff)
#         # m.setObjective(sum(cost_vec @ x[i, :] for i in range(1)), GRB.MINIMIZE)
#         # m.optimize()
#         # print("use normal A*")
#         # r = get_max_events(marking)
#         # if r > max_rank:
#         #     max_rank = r
#         # if m.status == 2:
#         return h, x, True, split_lst, max_rank
#         # else:
#         #     return "HEURISTICINFINITE", [0 for i in range(len(cost_vec))], "Infeasible", split_lst, max_rank
#     else:
#         split_lst.append(rank+1)
#         return -1, [0 for i in range(len(cost_vec))], 0, split_lst, max_rank


def __compute_exact_heuristic_new_version(trans_len, a_matrix, h_cvx, g_matrix, cost_vec,
                                          m_vec, fin_vec, variant):
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()
    # not available in the latest version of PM4Py
    b_term = matrix(b_term)
    parameters_solving = {"solver": "glpk"}

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * trans_len

    return prim_obj, points


def get_max_events(marking):
    if marking.t is None:
        return -2
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
    return get_max_events(marking.p)
