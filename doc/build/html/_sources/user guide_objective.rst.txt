Objective function
==================

In this section we will show how to add objective
function to the QP problem. For linear programming
problems, the cost function is linear, while for  
QP, the cost function is in quadratic form.

In general, we use **+=** to add objective function::

    import unisolver
    prob = unisolver.QpProblem("myproblem", "quadprog")
    prob += x ** 2 + y ** 2 + 2 * x

The right part of **+=** will be recognized as 
a QpExpression element. And by separating them into 
parts and transform to matrix format.
