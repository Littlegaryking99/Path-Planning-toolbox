Specify variables for QP problems
==================================

In this section, we will show how to 
specify variables for the model.

After we initialize the QP problem and declare 
the solver that used to solve the QP, we can 
add variables to the problem. 

There are many ways to define the variables. 
The simplest way is directly specify **QpVariable** 
class::

    import unisolver
    x = QpVariable("x", 0, 3)

If lowbound or upbound is not specified, it 
would be initialized by None. And its initial
value is set to be 0 if not initialized.

