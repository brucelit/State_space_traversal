import pulp
import numpy as np
from scipy.optimize import linprog


def compute_ini_heuristic(ini_vec,fin_vec,cost_vec,incidence_matrix,consumption_matrix,split_lst,t_index):
    k = len(split_lst)-1
    if k == 0:
        return ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec)
    split_lst = sorted(split_lst[1:], key=lambda k: k.label)

    place_num = len(incidence_matrix)
    trans_num = len(incidence_matrix[0])
    marking_diff = np.array(fin_vec) - np.array(ini_vec)

    # 1. define problem
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    #
    # 定义变量x_1和y_1放到列表中,用var表示, 0-k-1是y的，k-2k-2是x的
    var = np.array([[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger)
                    for j in range(trans_num)]
                    for i in range(2*k)])
    # print(type(var))
    # 0-k是给x用的，k+1到2k-1是给y用的
    costs = np.array([cost_vec for i in range(k)])

    # add objective定义目标函数
    prob += pulp.lpDot(costs.flatten(), var[0:k].flatten())

    for i in range(0, k):
        temp = split_lst[i]
        # print(temp)
        trans_index = t_index[temp]
        var[k + i, :] = 0
        var[k + i][trans_index] = 1

    # rule 1
    var1 = np.array(var).sum(axis=0)
    # print("\nvar1", var1)
    ct1 = np.dot(incidence_matrix, var1)
    # print("\nct1", ct1)

    for i in range(place_num):
        prob += (pulp.lpSum(ct1[i]) == marking_diff[i])

    #rule 2
    for a in range(0, k):
        var2 = np.array([0 for i in range(trans_num)])
        for b in range(0, a):
            # -1是为了映射到对应的变量上
            var2 = var2 + var[b] + var[b+k]
        ct3 = np.dot(incidence_matrix, var2) + np.dot(consumption_matrix, var[a+k])
        for i in range(place_num):
            prob += (pulp.lpSum(ct3[i]) >= -ini_vec[i])
    prob.solve()
    print("计算结果：", pulp.value(prob.objective))
    dict1 = {'heurstic': pulp.value(prob.objective),
            'var': [[pulp.value(var[i][j]) for j in range(trans_num)] for i in range(k*2)]}
    # print([[pulp.value(var[i][j]) for j in range(trans_num)] for i in range(k*2)])
    # print("!!!solution vector:", np.array(dict1['var']).sum(axis=0))
    # print("var",dict1['var'])
    # print("solution in func:", np.array(dict1['var']).sum(axis=0))
    print("costs", costs)
    print("var", dict1['var'])
    return dict1['heurstic'], np.array(dict1['var']).sum(axis=0)


def ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec):
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
    dict1 = {'heuristic': pulp.value(prob.objective),
             'var': [pulp.value(var[i]) for i in range(trans_num)]}
    print("init h when no split point",dict1["heuristic"])
    print("solution vector when no split point:", [pulp.value(var[i]) for i in range(trans_num)])
    return dict1['heuristic'], np.array(dict1['var'])

# compute heuristic marking m' from marking m
def compute_exact_heuristic(ini_vec, fin_vec, inc_matrix, cost_vec):
    marking_diff = np.asarray(fin_vec) - np.asarray(ini_vec)
    prob = pulp.LpProblem('Heuristic', sense=pulp.LpMinimize)
    var = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger)
                    for i in range(len(cost_vec))]
    prob += pulp.lpDot(cost_vec, var)
    var1 = np.dot(inc_matrix, np.array(var))
    for i in range(len(ini_vec)):
        prob += (pulp.lpSum(var1[i]) == marking_diff[i])
    prob.solve()
    dict1 = {'heurstic': pulp.value(prob.objective),
             'var': [pulp.value(var[i]) for i in range(len(cost_vec))]}
    print("compute exact heuristic", dict1['var'], dict1['heurstic'])
    return dict1['heurstic'], dict1['var']


def compute_estimate_heuristic(h_score, solution_x, t_index, cost_vec):
    result_aux = [0 for x in cost_vec]
    not_trust = 1
    if solution_x[t_index] >= 1:
        not_trust = 0
    result_aux[t_index] = 1
    new_solution_x = np.array(solution_x) - np.array(result_aux)
    new_h_score = max(0, h_score - cost_vec[t_index])
    return new_h_score, new_solution_x, not_trust