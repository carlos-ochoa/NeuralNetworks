import math
import numpy as np
import random
import matplotlib.pyplot as plt

# First we plot a 3-cycle sine function
x = np.arange(0,9.5,0.1)
freq = 2
amp = 0.5
sine = amp * np.sin(freq * x)
#sine = amp * x + freq
plt.plot(x,sine)
plt.show()
# Add gaussian-noise
e = 0.1
noise = np.random.randn(len(sine))
sine = sine + noise * e
plt.scatter(x,sine)
plt.show()
# Start gradient descent to optimize parameter freq and amp
epochs = 100
alpha = 0.2
# Initialize random parameter freq and amp
amp_hat = random.random()
freq_hat = random.random()
errors = []
for e in range(epochs):
    # Calculate the lms error
    sine_hat = amp_hat * np.sin(freq_hat * x)
    #sine_hat = amp_hat * x + freq_hat
    error = np.mean(0.5 * ((sine - sine_hat) ** 2))
    errors.append(error)
    # Partial derivatives of J for amp_hat and freq_hat
    d_amp_hat = np.mean((sine - sine_hat) * np.sin(freq_hat * x)) * -1
    d_freq_hat = np.mean((sine - sine_hat) * (amp_hat * x * np.cos(freq_hat * x))) * -1
    #d_amp_hat = np.mean(-1 * x * (sine - sine_hat))
    #d_freq_hat = np.mean(-1 * (sine - sine_hat))
    # Update values
    amp_hat -= alpha * d_amp_hat
    freq_hat -= alpha * d_freq_hat
for error in errors:
    print(error)

res = sine - sine_hat
# Print standard deviation
print('Desviación estándar: ',np.std(res))
#plt.distplt(res)
#plt.show()

print('amplitude ',amp_hat,' freq ',freq_hat)
plt.plot(x,sine_hat)
plt.show()
plt.plot(range(epochs),errors)
plt.show()
