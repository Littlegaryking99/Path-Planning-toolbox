Configure a solver in unisolver
===============================

After user have initialized the QP problem, 
they should also state the name and the 
specific solver used in the corresponding QP 
problem. 

There are several solvers included in unisolver, user 
can get the name of solvers like this::

    import unisolver
    prob = unisolver.QpProblem()
    prob.solvers
    # ['Gurobi','quadprog']

If the solver is not contained in the unisolver, 
it will print out error message::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "cvxopt")
    # This is not a valid solver in unisolver

For different solver, the unisolver will  
accept input in the same format, and adjust
the input into corresponding format for different 
solvers.


