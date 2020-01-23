import numpy as np
import matplotlib.pyplot as pt

n_state = 1000
n_step = 100

x = np.zeros((n_state, n_state))
y = np.zeros(n_state)

for i in range(n_state):
    x[i, i] = 2
    x[max(0, i-n_step):i, i] = -1/n_step
    x[i+1: i+1+n_step, i] = -1/n_step
    if i<n_step:
        y[i] = (i-n_step)/n_step
    elif (n_state-i-1)<n_step:
        y[i] = 1-(n_state-i-1)/n_step
    else:
        y[i] = 0

v = np.linalg.solve(x, y)

pt.figure(1)
pt.xlim(1, 1000)
pt.ylim(-1, 1)
pt.plot(np.arange(len(v))+1, v)
pt.show()
