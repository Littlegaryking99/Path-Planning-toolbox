Initialize a QP problem by unisolver
====================================

By import the unisolver model and initialize
a problem with name and specific solver, a 
QP problem can be initialized::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")

If the name of problem is not specified, it will 
automatically named by **NoName**::

    import unisolver
    prob = unisolver.QpProblem(solver = "quadprog")
    prob.name
    # NoName

If the solver is not contained in the unisolver, 
it will print out error message::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "cvxopt")
    # This is not a valid solver in unisolver

