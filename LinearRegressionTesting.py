from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype = np.float64)
#zs = np.array([5,4,6,5,6,7], dtype = np.float64)

# code to make the axes visible on a graph if needed
# plt.plot(xs, zs)
# plt.show()

def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	zs = []
	for i in range(hm):
		z = val + random.randrange(-variance, variance)
		zs.append(z)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step

	xs = [i for i in range(len(zs))]

	return np.array(xs, dtype=np.float64), np.array(zs, dtype=np.float64)

def best_fit_slope_and_intercept(xs, zs):
	m = ( ((mean(xs) * mean (zs)) - mean(xs*zs)) /
		 ((mean(xs)**2) - mean(xs**2)))
	b = mean(zs) - m*mean(xs)
	return m, b


def squared_error(zs_orig, zs_line):
	return sum((zs_line - zs_orig)**2)
	
def coeffecient_of_determination(zs_orig, zs_line):
	z_mean_line = [mean(zs_orig) for z in zs_orig]
	squared_error_reqr = squared_error(zs_orig, zs_line)
	squared_error_z_mean = squared_error(zs_orig, z_mean_line)
	return 1 - (squared_error_reqr / squared_error_z_mean)


xs, zs = create_dataset(40,10,2, correlation = 'pos')
m,b = best_fit_slope_and_intercept(xs, zs)


regression_line = [(m*x) + b for x in xs]
print('Line coeffecients are')
print(m, b)

#simple prediction
predict_x = 8
predict_z = (m*predict_x)+b

r_squared = coeffecient_of_determination(zs, regression_line)
print('Coeffecient of Determination')
print(r_squared)


plt.scatter(xs,zs)
plt.scatter(predict_x, predict_z, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()