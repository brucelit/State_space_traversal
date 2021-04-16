import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

import gurobipy as gp
import numpy as np

try:
    m = gp.Model("model")
    x = m.addMVar(shape=(10), name="x")
    y = m.addMVar(shape=(5), name="y")
    A = np.random.rand(5, 10)
    print("y")
    m.addConstrs((A[i,:] @ x - (y[i]@y[i]) <= A[i,0] for i in range(5)),
    name="const")
    m.setObjective(x.sum() - y @ y, gp.GRB.MAXIMIZE)
    print("y@y", y@y)
    m.optimize()
except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
        print('Encountered an attribute error')

