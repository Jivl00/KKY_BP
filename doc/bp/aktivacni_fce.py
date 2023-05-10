import matplotlib.pyplot as plt
import numpy as np

# plot relu, sigmoid, tanh, and softmax
x = np.arange(-5, 5, 0.1)
fontsize = 14
title = 'Activation functions'
plt.rcParams.update({'font.size': fontsize})
plt.title(title)
plt.plot(x, np.maximum(x, 0), label='relu', color='red')
plt.plot(x, 1 / (1 + np.exp(-x)), label='sigmoid', color='green')
plt.plot(x, np.tanh(x), label='tanh', color='blue')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.savefig('aktivacni_fce.pdf')
plt.show()

