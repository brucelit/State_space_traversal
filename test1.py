from astar_implementation import utilities, thesis_one
from pm4py.objects import petri
import numpy as np

model_net, model_im, model_fm = utilities.construct_model_net()
trace_net, trace_im, trace_fm = utilities.construct_trace_net()
sync_net, sync_im, sync_fm = utilities.construct_sync_net(trace_net, trace_im, trace_fm, model_net, model_im, model_fm,
                                                          ">>")
incidence_matrix = petri.incidence_matrix.construct(sync_net)
a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)

enabled_trans = []
for trans in incidence_matrix.transitions:
    if trans.label[1] == ">>":
        enabled_trans.append(trans)
trans_index = []
for trans in enabled_trans:
    print(trans)
    t_index = incidence_matrix.transitions[trans]
    print(a_matrix[t_index])
# n = 10 # maximum number of bars
# L = 250 # bar length
# m = 4 # number of requests
# w = [187, 119, 74, 90] # size of each item
# b = [1, 2, 2, 1] # demand for each item
#  # creating the model
# model = Model()
# x = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="x[%d ,%d ]" % (i, j))
# for i in range(m) for j in range(n)}
# y = {j: model.add_var(obj=1, var_type=BINARY, name="y[%d ]" % j)
# for j in range(n)}
# # constraints
# for i in range(m):
#     model.add_constr(xsum(x[i, j] for j in range(n)) >= b[i])
# for j in range(n):
#     model.add_constr(xsum(w[i] * x[i, j] for i in range(m)) <= L * y[j])
#
# # optimizing the model
# model.optimize()
#
# # printing the solution
# print('')
# print('Objective value: {model.objective_value:.3} '.format(**locals()))
# print('Solution: ', end='')
# for v in model.vars:
#     if v.x > 1e-5:
#         print('{v.name} = {v.x} '.format(**locals()))
#         print(' ', end='')
#
