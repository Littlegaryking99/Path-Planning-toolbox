Initialize a QP problem by unisolver
====================================

To solve a QP problem, user should initialize a model by stating the name of the model and the solver 
they want to use for the model. To get the available solvers of unisolver, user can use *listsolver* 
function::

    import unisolver
    unisolver.listsolver()
    #["quadprog", "Gurobi"]

By import the unisolver model and acclaiming
a model with name and specific solver, a 
QP model can be initialized::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")

If the name of problem is not specified, it will 
automatically named by **NoName**::

    import unisolver
    prob = unisolver.QpProblem(solver = "quadprog")
    prob.name
    # NoName

