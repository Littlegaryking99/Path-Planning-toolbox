How to add a constraint 
=======================

In this section we will show how to add a constraint to the model.

In unisolver, we support constraint in 
linear form. By using **+=** to add constraints::

    import unisolver
    prob = unisolver.QpProblem("myProblem", "quadprog")
    prob += 2 * x + 3 * y <= 5

If the right part of **+=** is recognized as 
**QpConstraint**, which means it contains **QpExpression**,
sign and right hand side constant.

When the constraint is added to the model, it will
automatically given a name *ci*. Therefore, if 
the contraint is mistakenly input by user, they 
can revise them by the name of constraint.

