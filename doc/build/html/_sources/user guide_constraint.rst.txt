How to add a constraint 
=======================

While objective function is the performance standard of the system,
in most cases, not every value is applicable for variables. In that case, 
we use constraints to describe which value of variables is applicable.
In this section we will show how to add constraints to the model.

In unisolver, we support constraints in 
linear form. By using **+=** to add constraints::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")
    prob += 2 * x + 3 * y <= 5

If the right part of **+=** is recognized as 
**QpConstraint**, which means it contains **QpExpression**,
sign and right hand side constant, then the model will add the constraint.

When the constraint is added to the model, it will
automatically given a name *ci*, where *i* means this is the ith constraint. Therefore, if 
the contraint is mistakenly input by user, user can modify it by the name of constraint::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")
    prob += y + x <= 3
    prob.constraints["c0"]
    # x + y <= 3

The constraint can also be displayed in the format of a list of dictionary::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")
    prob += y + x <= 3
    prob.constraints["c0"].toDict()
    #{'constant': -3, 'coefficients': [{'name': 'y', 'value': 1}, {'name': 'x', 'value': 1}]}

Therefore, user can modify the constraint by two ways, which is similar to modify the objective function.
For more details about how to modify objective function, please refer to :doc:`user guide_objective`.

After we input the objectives and contraints, we can solve the model in a very simple way. Please move to next part for further information.

