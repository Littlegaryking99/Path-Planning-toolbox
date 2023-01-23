How to solve the model
======================

In this section we will show how to solve the model.

After the user have specified the variables, objective 
function and constraints, they can solve the 
QP problem::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")
    prob += x ** 2 + y ** 2 + 2 * x
    prob += x + y >= 3
    prob.solve()

Then unisolver will transform the problem into suitable  
format for corresponding solver and solve the problem.

User can input coefficients to specify what kind 
of output they want for further use.