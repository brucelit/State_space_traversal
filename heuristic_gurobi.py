import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def ini_heuristic_without_split(ini_vec, fin_vec, incidence_matrix, cost_vec):
    start_time = time.time()
    try:
        marking_diff = np.array(fin_vec) - np.array(ini_vec)

        # Create a new model
        m = gp.Model('h1')

        # Create Vars
        trans_num = len(cost_vec)
        var = m.addMVar(trans_num, lb=0, vtype=GRB.INTEGER, name="variable")
        ct1 = np.asarray(incidence_matrix) @ var

        # Set obj
        m.setObjective(np.asarray(cost_vec) @ var, GRB.MINIMIZE)

        # Add constraints
        m.addConstr(ct1 == marking_diff)

        # Optimize model
        m.optimize()
        print(time.time() - start_time)
        if GRB.OPTIMAL == 2:
            return m.objVal, np.asarray(m.x), "Optimal"
        else:
            return m.objVal, np.asarray(m.x), "Infeasible"


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')