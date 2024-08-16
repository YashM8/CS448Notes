import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Gradient norm change to stop optimization.
tolerance = 1e-5


# Function to optimize.
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


# Gradient of function to optimize.
def rosenbrock_gradient(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    df_dy = 2 * b * (y - x ** 2)
    return np.array([df_dx, df_dy])


def rosenbrock_hessian_inv(x, y):
    h11 = 2 - 400 * (y - 3 * x ** 2)
    h12 = -400 * x
    h21 = -400 * x
    h22 = 200
    determinant = h11 * h22 - h12 * h21
    hessian_inv = np.array([[h22, -h12], [-h21, h11]]) / determinant
    return hessian_inv


def gradient_descent(grad_f, f, initial_point, learning_rate, num_iterations):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*point)
        point -= learning_rate * gradient
        path.append(point.copy())
        k += 1
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k, f(*point)


def stochastic_gradient_descent(grad_f, f, initial_point, learning_rate, num_iterations):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    for k in range(num_iterations):
        noise = np.random.normal(scale=10, size=2)
        gradient = grad_f(*point)
        gradient = gradient + noise

        point -= learning_rate * gradient
        path.append(point.copy())
        k += 1
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k, f(*point)


def gradient_descent_armijo(grad_f, f, initial_point, initial_learning_rate, num_iterations, beta=0.5, sigma=0.001):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    learning_rate = initial_learning_rate
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*point)
        while f(*(point - learning_rate * gradient)) > f(*point) - sigma * learning_rate * np.linalg.norm(
                gradient) ** 2:
            learning_rate *= beta
        point -= learning_rate * gradient
        path.append(point.copy())
        if np.linalg.norm(gradient) < 1e-4:
            break

    return point, path, k + 1, f(*point)


def gradient_descent_momentum(grad_f, f, initial_point, learning_rate, num_iterations, momentum=1):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*point)
        point = point - learning_rate * gradient + momentum * (point - path[-2] if len(path) > 1 else 0)
        path.append(point.copy())
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k + 1, f(*point)


def gradient_descent_nesterov(grad_f, f, initial_point, learning_rate, num_iterations, momentum=1):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    momentum_term = np.zeros_like(point)
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*(point - momentum * momentum_term))
        momentum_term = momentum * momentum_term + learning_rate * gradient
        point = point - momentum_term
        path.append(point.copy())
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k + 1, f(*point)


def gradient_descent_nesterov_with_restart(grad_f, f, initial_point, learning_rate, num_iterations, momentum=0.9,
                                           restart_threshold=1e-6):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    momentum_term = np.zeros_like(point)
    prev_f_value = f(point[0], point[1])
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*(point - momentum * momentum_term))
        momentum_term = momentum * momentum_term + learning_rate * gradient
        point = point - momentum_term
        f_value = f(point[0], point[1])
        path.append(point.copy())
        if f_value - prev_f_value > restart_threshold:
            momentum_term = np.zeros_like(point)
        prev_f_value = f_value
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k + 1, f(*point)


def newtons_method(grad_f, f, initial_point, _, num_iterations):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*point)
        point -= rosenbrock_hessian_inv(*point) @ gradient
        path.append(point.copy())
        k += 1
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k, f(*point)


def newtons_method_damped(grad_f, f, initial_point, learning_rate, num_iterations):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    for k in range(num_iterations):
        gradient = grad_f(*point)
        point -= (learning_rate * rosenbrock_hessian_inv(*point)) @ gradient
        path.append(point.copy())
        k += 1
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k, f(*point)


def coordinate_descent(grad_f, f, initial_point, learning_rate, num_iterations):
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    k = None
    n = len(point)

    for k in range(num_iterations):
        gradient = grad_f(*point)
        for i in range(n):
            point[i] -= learning_rate * gradient[i]
            path.append(point.copy())
        k += 1
        if np.linalg.norm(gradient) < tolerance:
            break

    return point, path, k, f(*point)


def plot_optimizer(run_optim_func, lr, max_iter):
    np.random.seed(1975)
    initial_point = np.random.uniform(low=-2, high=2, size=2)
    minimum_point, path, iterations, point_reached = run_optim_func(rosenbrock_gradient, rosenbrock, initial_point,
                                                                    lr, max_iter)

    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock(X, Y)

    path = np.array(path)

    sns.set(style="darkgrid")
    plt.style.use('dark_background')
    plt.scatter(initial_point[0], initial_point[1], color='red', marker='x', label='Start', s=300)
    contour = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 40), cmap='viridis')
    plt.colorbar(contour)
    plt.plot(path[:, 0], path[:, 1], 'g-', label='Path')
    plt.title(f'{run_optim_func.__name__} \n Max Iterations: {iterations} \n Point reached {point_reached}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)


optimizers = [
    (gradient_descent, 0.001, 1000),
    (stochastic_gradient_descent, 0.001, 1000)
    # (gradient_descent_momentum, 0.0001, 15),
    # (gradient_descent_nesterov, 0.0001, 15),
    # (gradient_descent_nesterov_with_restart, 0.001, 100),
    # (gradient_descent_armijo, 1, 100),
    # (newtons_method, -1, 10),
    # (newtons_method_damped, 0.999, 10),
    # (coordinate_descent, 0.001, 10000)
]

for optimizer, param1, param2 in optimizers:
    plot_optimizer(optimizer, param1, param2)
    plt.show()


def save_plot(optimizer_name):
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{optimizer_name}.png")
    plt.close()
#
#
# for optimizer, param1, param2 in optimizers:
#     plot_optimizer(optimizer, param1, param2)
#     optimizer_name = optimizer.__name__
#     save_plot(optimizer_name)
