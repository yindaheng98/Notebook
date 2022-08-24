import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0., 1., 1e-6)
x_q = np.round(np.log(x-np.min(x)+1)/np.max(np.log(x-np.min(x)+1))*255)
bin = np.bincount(x_q.astype('int'))[1:-1]
levels = np.arange(0, 256, 1)[1:-1]
plt.plot(levels, np.sum(bin)/bin)
plt.xlabel('levels')
plt.ylabel('density')
plt.show()
