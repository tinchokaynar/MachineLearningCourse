from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
# ys = np.array([5, 6, 4, 4, 6, 5, 7, 5, 6], dtype=np.float64)


def dataset_create(points, variance, step=2, correlation=False):
    val = 1
    xs = [i for i in range(points)]
    ys = []
    for i in range(points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / (pow(mean(xs), 2) - mean(pow(xs, 2))))
    b = mean(ys) - m * mean(xs)
    return m, b


def square_error(ys_orig, ys_line):
    return np.sum(np.power(ys_line-ys_orig, 2))


def determination_coef(ys_orig, ys_line):
    square_error_rgr = square_error(ys_orig, ys_line)
    square_error_mean = square_error(ys_orig, np.mean(ys_orig))
    return 1 - (square_error_rgr/square_error_mean)


xs, ys = dataset_create(points=50, variance=10, correlation='neg')
m, b = best_fit_slope_and_intercept(xs, ys)
regression = [(m*x) + b for x in xs]

predict_x = 12
predict_y = m*predict_x + b
cd = determination_coef(ys, regression)

print(f'Slope: {m}')
print(f'Interception: {b}')
print(f'Determination Coef: {cd}')

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(regression)
plt.show()
