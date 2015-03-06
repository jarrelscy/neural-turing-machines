#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tasks
import sys
import run_model

def plot(seq_length):
    i,o = tasks.repeat_copy(9,seq_length,2)
    weights,outputs = do_task(i)
    print i, outputs
    plt.figure(1,figsize=(7,7))
    plt.subplot(311)
    plt.imshow(i.T,interpolation='nearest')
    plt.subplot(312)
    plt.imshow(o.T,interpolation='nearest')
    plt.subplot(313)
    plt.imshow(outputs.T,interpolation='nearest')
    plt.show()


def plot_weights(seq_length):
    i,o = tasks.repeat_copy(9,seq_length,2)
    weights,outputs = do_task(i)
    plt.figure(1,figsize=(20,7))
    plt.imshow(weights.T[0:1023],interpolation='nearest',cmap=cm.gray)
    plt.show() 

if __name__ == "__main__":
    P,do_task = run_model.make_model(input_size = 9,
		mem_size   = 1024,
		mem_width  = 20,
		output_size = 9
	)
    P.load('l2_repeat_2.mdl')
    plot(10)
    #plot_weights(8)