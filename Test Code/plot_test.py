import numpy as np
import matplotlib.pyplot as plt


x = np.array([0, 0, 0, 0, .01, .02, .04, .06, .08])
y = np.array([0, .02, .04, .06, .08, .10, .12, .13, .14])

# object
x_ob_1 = np.array([-.2, .06])
y_ob_1 = np.array([.08, .08])

x_ob_2 = np.array([.06, .06])
y_ob_2 = np.array([.08, .2])





plt.plot(x, y, '-')
plt.plot(x_ob_1, y_ob_1, 'r-')
plt.plot(x_ob_2, y_ob_2, 'r-')


#idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
#plt.plot(x[idx], f[idx], 'ro')
plt.show()