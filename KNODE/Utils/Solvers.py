"""## solvers"""
def RK(z0, n_steps, f, h):
    '''
    4th Order Runge Kutta Numerical Solver
    Input:
      z0: initial condition
      t0: initial time (not actual time, but the index of time)
      n_steps: the number of steps to integrate
      f: vector field
      h: step size
    Return:
      z: the state after n_steps
    '''
    z = z0
    for i in range(int(n_steps)):
        k1 = h * f(z)
        k2 = h * f(z + 0.5 * k1)
        k3 = h * f(z + 0.5 * k2)
        k4 = h * f(z + k3)

        z = z + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z