from scipy.optimize import linprog

obj = [-1,-2]
lhs_ineq = [[2,1],[-4,5],[1,-2]]
rhs_ineq = [20,10,2]
lhs_eq = [[-1,5]]
rhs_eq = [15]
bnd = [(0,float("inf")),
       (0,float("inf"))]
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
      A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,method="revised simplex")
print(opt.success)
