from numpy import array, cross
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define lines A and B by two points
X = [0, 0]
Y = [0, 0]
Z = [1, 0]
XA0 = array([1, 0, 0])
XA1 = array([1, 1, 0])
XB0 = array([.5, 0, 0])
XB1 = array([0, .5, 0])

# compute unit vectors of directions of lines A and B
UA = (XA1 - XA0) / norm(XA1 - XA0)
UB = (XB1 - XB0) / norm(XB1 - XB0)
# find unit direction vector for line C, which is perpendicular to lines A and B
UC = cross(UB, UA); UC /= norm(UC)

# solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
RHS = XB0 - XA0
LHS = array([UA, -UB, UC]).T
print(solve(LHS, RHS))
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X, Y, Z)
plt.plot(XA0, XA1)
plt.plot(XB0, XB1)
plt.show()


# prints "[ 0. -0.  1.]"