import matplotlib.pyplot as plt
import solver

# Define Fitzhugh-Nagumo model
def fitzhugh_nagumo(x, y):
    alpha = 0.1
    gamma = 0.5
    epsilon = 0.01
    eq1 = (y[0] * (1 - y[0]) * (y[0] - alpha) - y[1] + 0.026) / epsilon
    eq2 = y[0] - gamma * y[1]

    return [eq1, eq2]


# Define initialisation value for numerical method
x_min = 0
x_max = 0.43
initial_value = [0.01, 0.01]
# mesh_points = 10

problem = solver.AdaptiveMethod(
    fitzhugh_nagumo, x_min, x_max, initial_value, initial_mesh=0.1)

# Solve with adaptive method, Runge-Kutta 45 method
mesh, soln = problem.ode45(abs_tol=0.5, rel_tol=1e-1)

# mesh, soln = problem.ode23()
# print(mesh)

v = []
w = []
for i in range(len(soln)):
    v.append(soln[i][0])
    w.append(soln[i][1])

plt.figure()
plt.scatter(v,w)
plt.show()