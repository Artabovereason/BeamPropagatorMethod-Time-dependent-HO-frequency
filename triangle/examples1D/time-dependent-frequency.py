import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

'''
	omega 		  : the frequency of oscillation of time-dependent potential.
	Nx, Ny		  : grid points.
	dt			  : evolution step.
	ntp 		  : number of periods.
	tmax 		  : end of propagation.
	xmax 		  : x-window size.
	ymax 		  : y-window size.
	images    	  : number of .png images.
	absorb_coeff  : 0 = periodic boundary
	output_choice : If 1, it plots on the screen but does not save the images.
					If 2, it saves the images but does not plot on the screen.
					If 3, it saves the images and plots on the screen.

	fixmaximum	  : fixes a maximum scale of |psi|**2 for the plots.
				    If 0, it does not fix it.

	psi_0		  : initial wavefunction.
	V			  : the potential.

'''

omega         = float(input('Enter the value of omega : '))
Nx            = 600   # 600
Ny            = Nx
dt            = 0.01 # 0.001
ntp           = 3


if omega==0 :
	tmax 	  = ntp*2*np.pi
else:
	tmax      = ntp*2*np.pi/omega

xmax          = 5
ymax 	      = xmax
images 		  = 300
absorb_coeff  = 2000000
output_choice = 1

fixmaximum    = 0

#steps_image   = int(tmax/dt/images )
steps_image   = int(tmax/dt)
t_values      = []
triangle_function  = []

#for j in range(steps_image*images+1):
for j in range(steps_image+1):
    t_values.append(j*dt)



for i in range(len(t_values)):
    triangle_function.append(2+signal.sawtooth(t_values[i]*omega, 0.5))

#triangle_function = 2+signal.sawtooth(t_values*omega, 0.5)

#plt.plot(t_values ,[signal.sawtooth(w*omega, 0.5) for w in t_values] )
#plt.show()


def psi_0(x,y):

	f = 0.j+np.exp(-( (x**2))/2)/np.sqrt(np.sqrt(np.pi))   # A Gaussian

	return f;

def V(x,y,t,psi,omega):
	#V = triangle_function[int(t/dt)]*(x**2)/2.0
	V = triangle_function[t_values.index(t)]*(x**2)/2.0
	return V;
