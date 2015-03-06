import cPickle
import sys
from matplotlib import pyplot as plt

plt.plot(cPickle.load(open(sys.argv[1], 'rb')))
plt.show()