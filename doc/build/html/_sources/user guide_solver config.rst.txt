Configure a solver in unisolver
===============================

After user have initialized the QP problem, 
they should also state the name and the 
specific solver used in the corresponding QP 
problem. 

For solver installation, please refer to :doc:`usage_installation`

There are several solvers included in unisolver, user 
can get the list of solvers by the following method::

    import unisolver
    prob = unisolver.QpProblem()
    prob.solvers
    # ['Gurobi','quadprog']

If the solver is not contained in the unisolver, 
it will print out error message::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "cvxopt")
    # This is not a valid solver in unisolver

For different solvers, they may require different formats of input. However, for some light solvers,
they do not have detailed documentation, which create barrier for users. Therefore, in unisolver we 
accept same format of input for every included specific solver. For detailed input format, please refer to 
:doc:`user guide`.

After user config the solver of the model, they can plug in detailed infomation about the model, for instance,  
:doc:`user guide_variables`, :doc:`user guide_objective` and :doc:`user guide_constraint`.


