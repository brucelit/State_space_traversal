import re

import numpy as np
import gurobipy as gp
from gurobipy import GRB


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

    # optimise model
    m.optimize()
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
    return m.objVal, np.array(x.X.sum(axis=0))


# compute the exact heuristic of marking m
def get_exact_heuristic_new(marking, split_lst, marking_diff, ini, incidence_matrix, cost_vec, max_rank, trace_len):
    insert_position = 1
    if marking.m != ini:
        if max_rank + 1 not in split_lst:
            insert_position = -1

    if marking.m == ini or insert_position >= 0 or max_rank >= trace_len - 1:
        print("max rank",max_rank)
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
            return "HEURISTICINFINITE", [0 for i in range(len(cost_vec))], "Infeasible", split_lst, max_rank
    else:
        split_lst.append(max_rank + 1)
        print("append rank", max_rank)
        return -1, [0 for i in range(len(cost_vec))], 0, split_lst, max_rank


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


def get_max_events(marking):
    if marking.t is None:
        return -1
    if marking.t.label[0] != ">>":
        return int(re.search("(\d+)(?!.*\d)", marking.t.name[0]).group())
    return get_max_events(marking.p)
