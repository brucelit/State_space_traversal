import time
import pulp
import numpy as np


def compute_ini_heuristic(ini_vec, fin_vec, cost_vec, incidence_matrix,
                          consumption_matrix, split_dict, t_index, p_index,
                          trace_lst_log, trace_lst_sync, set_model_move):
    k = len(split_dict) - 1

    split_dict = dict(sorted(split_dict.items(), key=lambda item: item[1]))
    split_lst = list(split_dict.values())[1:]
    split_lst.append(len(trace_lst_log))
    place_num = len(p_index)
    trans_num = len(t_index)

    # define problem
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)

    # define x_i from x_0 to x_k, with k+1 x_i
    var_x = np.array([[pulp.LpVariable(f'x{i}_{j}', lowBound=0, upBound=255, cat=pulp.LpInteger)
                       for j in range(trans_num)] for i in range(k + 1)])

    # define y_i from 0 to k
    var_y = np.array([[pulp.LpVariable(f'y{i}_{j}', lowBound=0, upBound=1, cat=pulp.LpInteger)
                       for j in range(trans_num)] for i in range(k + 1)])

    # constraint 1
    marking_diff = np.array(fin_vec) - np.array(ini_vec)
    ct1 = np.dot(incidence_matrix, var_x.sum(axis=0)) + np.dot(incidence_matrix, var_y.sum(axis=0))
    for j in range(place_num):
        # prob += ct1[i] == marking_diff[i]
        prob.addConstraint(
            pulp.LpConstraint(
                e=ct1[j],
                sense=pulp.LpConstraintEQ,
                rhs=marking_diff[j]))

    # c2
    a = 1
    b = 0
    var_temp2 = var_x[0]
    while a < k + 1:
        if b > 0 and b < a:
            var_temp2 = var_temp2 + var_x[b] + var_y[b]
        # ct2 = np.matmul(incidence_matrix, var2) + np.matmul(consumption_matrix, var_y[a])
        # start_time = time.time()
        ct2 = np.array(ini_vec) + \
              np.dot(incidence_matrix, var_temp2) + np.dot(consumption_matrix, var_y[a])
        # print("config time of matrix", time.time() - start_time)

        for j in range(place_num):
            prob.addConstraint(
                pulp.LpConstraint(
                    e=ct2[j],
                    sense=pulp.LpConstraintGE,
                    rhs=0))
        b += 1
        a += 1

    # constraint 5,6
    y_col = 1
    prob += np.sum(var_y[0]) == 0
    for i in split_lst[:-1]:
        if trace_lst_sync[i] is not None and trace_lst_log[i] is not None:
            prob += pulp.LpAffineExpression([(var_y[y_col][trace_lst_sync[i]], 1),
                                             (var_y[y_col][trace_lst_log[i]], 1)]) == 1
        elif trace_lst_sync[i] is None and trace_lst_log[i] is not None:
            prob += pulp.LpAffineExpression([(var_y[y_col][trace_lst_log[i]], 1)]) == 1
        prob += pulp.lpSum(var_y[y_col]) == 1
        y_col += 1

    # add objective
    costs = np.array([cost_vec for i in range(2 * k + 2)])
    x = np.concatenate((var_x, var_y), axis=0)
    objective = pulp.lpSum(x[i, j] * costs[i, j]
                           for j in range(trans_num)
                           for i in range(2 * k + 2))
    prob.setObjective(objective)
    # path_to_cplex = r"E:\Program Files\IBM\ILOG\CPLEX_Studio201\cplex\bin\x64_win64\cplex.exe"
    # solver = pulp.CPLEX_CMD(path=path_to_cplex)
    # prob.solve(solver)
    prob.solve()
    if pulp.LpStatus[prob.status] == "Optimal":
        dict1 = {'heuristic': int(pulp.value(prob.objective)),
                 'var_x': [[int(pulp.value(var_x[i][j])) for j in range(trans_num)] for i in range(k + 1)],
                 'var_y': [[int(pulp.value(var_y[i][j])) for j in range(trans_num)] for i in range(k + 1)]}
        return dict1['heuristic'], np.array(dict1['var_x']).sum(axis=0) + np.array(dict1['var_y']).sum(axis=0), \
               pulp.LpStatus[prob.status]
    else:
        return 0, 0, "Infeasible"


def compute_ini_heuristic2(ini_vec, fin_vec, cost_vec, incidence_matrix,
                           consumption_matrix, split_dict, t_index, p_index,
                           trace_lst_log, trace_lst_sync, prob):
    k = len(split_dict) - 1

    split_dict = dict(sorted(split_dict.items(), key=lambda item: item[1]))
    split_lst = list(split_dict.values())[1:]
    split_lst.append(len(trace_lst_log))
    place_num = len(p_index)
    trans_num = len(t_index)

    # define problem
    # prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)

    # define x_k, y_k
    d1 = {"var_x{0}".format(k): np.array([[pulp.LpVariable(f'x{i}_{j}', lowBound=0, upBound=255, cat=pulp.LpInteger)
                                           for j in range(trans_num)] for i in range(k + 1)])}

    d2 = {"var_y{0}".format(k): np.array([[pulp.LpVariable(f'y{i}_{j}', lowBound=0, upBound=1, cat=pulp.LpInteger)
                                           for j in range(trans_num)] for i in range(k + 1)])}

    # constraint 1
    marking_diff = np.array(fin_vec) - np.array(ini_vec)
    for k, v in d1.items():
        x = str(k)
    for k, v in d2.items():
        y = str(k)
    ct1 = ct1 + np.dot(incidence_matrix, d1[x]) + np.dot(incidence_matrix, d2[y])
    for j in range(place_num):
        prob.addConstraint(
            pulp.LpConstraint(
                e=ct1[j],
                sense=pulp.LpConstraintEQ,
                rhs=marking_diff[j]))

    # constraint 2
    a = 1
    b = 0
    ct2 = ct2 + np.dot()

    for j in range(place_num):
        prob.addConstraint(
            pulp.LpConstraint(
                e=ct2[j],
                sense=pulp.LpConstraintGE,
                rhs=0))
    temp2 = d1[x] + d1[y]
    ct2 = ct2 + np.dot(incidence_matrix, temp2)

    # constraint 5,6
    y_col = 1
    if trace_lst_sync[k] is not None and trace_lst_log[k] is not None:
        prob += pulp.LpAffineExpression([(d2[y][y_col][trace_lst_sync[k]], 1),
                                         (d2[y][y_col][trace_lst_log[k]], 1)]) == 1
    elif trace_lst_sync[k] is None and trace_lst_log[k] is not None:
        prob += pulp.LpAffineExpression([(d2[y][y_col][trace_lst_log[k]], 1)]) == 1
    prob += pulp.lpSum(d2[y][y_col]) == 1
    y_col += 1

    # add objective
    objective = prob.objective + pulp.lpSum(d1[x][i] * cost_vec[i]
                                            for i in range(trans_num))
    prob.setObjective(objective)

    prob.solve()
    if pulp.LpStatus[prob.status] == "Optimal":
        dict1 = {'heuristic': int(pulp.value(prob.objective)),
                 'var_x': [int(pulp.value(d1[x][j])) for j in range(trans_num)],
                 'var_y': [int(pulp.value(d1[y][j])) for j in range(trans_num)]}
        return dict1['heuristic'], np.array(dict1['var_x']).sum(axis=0) + np.array(dict1['var_y']).sum(axis=0), \
               pulp.LpStatus[prob.status], prob, ct1, ct2
    else:
        return 0, 0, "Infeasible"


# compute heuristic of marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    var = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger)
           for i in range(len(cost_vec))]
    prob += pulp.lpDot(cost_vec, var)
    var1 = np.dot(inc_matrix, np.array(var))
    for j in range(len(ini_vec)):
        prob.addConstraint(
            pulp.LpConstraint(
                e=var1[j],
                sense=pulp.LpConstraintEQ,
                rhs=marking_diff[j]))
    prob.solve()
    dict1 = {'heuristic': int(pulp.value(prob.objective)),
             'var': [int(pulp.value(var[i])) for i in range(len(cost_vec))]}
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
