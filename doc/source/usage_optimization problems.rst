Optimization problems
=====================

The general formulation of the optimization problem in engineering design is to select a group of 
parameters (variables) to make the design index (target) reach the optimal value under a series 
of related constraints (constraints). Therefore, optimization problems can usually be expressed as 
problems in the form of mathematical programming.

Linear Programming problems
---------------------------
**Linear programming (LP)** is an important branch of operational research with early research, rapid 
development, wide application and mature methods. It is a mathematical method to assist people 
in scientific management. It is a mathematical theory and method to study the extreme value problem 
of linear objective function under linear constraints. Its standard form is as follows:

.. math::
    minimize \ \ c^Tx &+ d  \\
    subject \ to \ \ Gx &\leqslant h\\
                Ax &= b\\

Quadratic Programming problems
------------------------------
**Quadratic programming (QP)** is the process of solving some mathematical optimization problems 
involving quadratic functions. Specifically, we seek to optimize (minimize or maximize) 
multivariate quadratic functions subject to linear constraints of variables. Quadratic 
programming is a nonlinear programming. It can be written as 

.. math::
    minimize \ \ \frac{1}{2}x^TPx &+ q^Tx + r \\
    subject \ to \ \ Gx &\leqslant h\\
                Ax &= b\\

