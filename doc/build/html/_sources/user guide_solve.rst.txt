How to solve the model
======================

After user have input the variables, objective function and constraints, they can solve the model in
a vary simple way. In this section we will show how to solve the model.

For most cases, the model can be solved by *solve* function::

    import unisolver
    prob = QpProblem("myProblem", "quadprog")
    x = QpVariable("x", 0, 3)
    y = QpVariable("y", 0, 1)
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob += y + x <= 3
    prob += x + y >= 2
    prob.solve()
    #[[2.]
    # [0.]]

For different solvers, unisolver will provide different data types due to corresponding solver's settings::

    prob = QpProblem("myProblem", "Gurobi")
    x = QpVariable("x", 0, 3)
    y = QpVariable("y", 0, 1)
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob += y + x <= 3
    prob += x + y >= 2
    prob.solve()
    #[1.9999558636043262, 4.413643994956694e-05]

If the problem is infeasible, it will print out default error massage for each solver.

After user understands the fundemental functions of unisolver, let's see some case studies about 
how unisolver can be used to solve real-life problems.