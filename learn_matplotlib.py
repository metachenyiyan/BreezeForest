import matplotlib.pyplot as plt
import random
prefix = 6.18

rx = [prefix+(0.001*random.random()) for i in range(100)]
ry = [prefix+(0.001*random.random()) for i in range(100)]
plt.plot(rx,ry,'ko')

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.show()