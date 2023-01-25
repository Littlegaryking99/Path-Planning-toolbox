Case Study
==========

In this section, we are going to show some practical examples

Control Lyapunov Function (CLF) - Control Barrier
Function (CBF) - Quadratic Programming (QP) is an 
approach for motion planning. For different
control input, CLF and CBF constraint may have different relative 
degrees. Although there are no specific Barrier
function which are suitable for every situations,
we can somehow use energy-like form to test whether 
it satisfy the requirements of Lyapunov or Barrier function.

In the following part, we illustrate an example:

For an adaptive cruise-control problem,
whose general form of dynamics of the systems are given by:

.. math::
    \frac{d}{dt} \begin{bmatrix}v \\ d \end{bmatrix} = 
    \begin{bmatrix}0 \\ v_0 - v \end{bmatrix} - 
    \frac{1}{m} \begin{bmatrix} 1 & v & v^2 \\ 0 & 0 & 0 \end{bmatrix}
    \begin{bmatrix} f_0 \\ f_1 \\ f_2 \end{bmatrix} + 
    \begin{bmatrix} \frac{1}{m} \\ 0 \end{bmatrix} u

**Example 1**: 

.. image:: Ex1.png

We define corresponding parameters:

- State \ :math:`x = \begin{bmatrix} p \\ v \\ z \end{bmatrix} \in \mathbb{R}^3` 
- Control input \ :math:`u \in \mathbb{R}`
- Dynamics \ :math:`\dot x = \begin{bmatrix} v \\ -\frac{1}{m}F_r(v) \\ v_0 - v\end{bmatrix} + \begin{bmatrix} 0 \\ \frac{1}{m} \\ 0\end{bmatrix}u`, where :math:`F_r(v) = f_0 + f_1v  + f_2v^2` is the resistance.
- Input constraints \ :math:`-mc_dg \leq u \leq mc_ag`
- Stability objective \ :math:`v \to v_d` (:math:`v_d`: desired velocity)
- Safety objective \ :math:`z \geq T_hv` (:math:`T_h``: lookahead time)
- Lyapunov function \ :math:`V(x) =  (v - v_d)^2` 
- CLF constraint \ :math:`(v - v_d)\{\frac{2}{m}(u - F_r) + \lambda(v - v_d)\} \leq \delta`
- Barrier function \ :math:`h(x) = z - T_hv - \frac{\frac{1}{2}(v-v_0)^2}{c_dg}`
- CBF constraint \ :math:`\frac{1}{m}(T_h + \frac{v - v_0}{c_dg})(F_r(v) - u) + \gamma(v_0 - v + z - T_hv - \frac{(v - v_0)^2}{2c_dg}) \geq 0`
- QP :math:`1/2\begin{bmatrix} u \\ \delta \end{bmatrix}^T \begin{bmatrix} \frac{4}{m^2} & 0 \\ 0 & 3\end{bmatrix}\begin{bmatrix} u \\ \delta \end{bmatrix} + \begin{bmatrix} -\frac{2F_r}{m^2} & 0\end{bmatrix}\begin{bmatrix} u \\ \delta \end{bmatrix}`

Then we can set up some initial values of parameters:

- Initial state :math:`x = \begin{bmatrix} 0 \\ 20 \\ 100 \end{bmatrix}`
- Lead vehicle velocity :math:`v_0 = 14`
- Desired velocity :math:`v_d  = 24`
- Weight :math:`m = 1650`, :math:`g = 9.81`
- Friction :math:`f_0 = 0.1`, :math:`f_1 = 5.0`, :math:`f_2 = 0.25`
- Input constraints :math:`c_a = 0.3`, :math:`c_d = 0.3`
- Lookahead time :math:`T_h = 0.8`
- CLF parameter :math:`\lambda = 5`
- CBF parameter :math:`\gamma = 5`