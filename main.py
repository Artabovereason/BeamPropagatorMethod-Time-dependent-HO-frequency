import numpy as np
import sys
import os
import importlib
import glob
import matplotlib.pyplot as plt
import matplotlib        as mpl
import seaborn           as sns
import math
import cmath
import time
import multiprocessing
from   scipy.integrate import simps
from   joblib          import Parallel
from   joblib          import delayed
from   matplotlib      import cm
from   scipy.optimize  import curve_fit

sns.set(rc={'axes.facecolor':'whitesmoke'})

start = time.clock()

'''=========================================================================='''

def psi_0(x,y):
	f = 0.j+np.sqrt(np.sqrt(np.sqrt(3)))*np.exp(-( np.sqrt(3)*(x**2))/2)/np.sqrt(np.sqrt(np.pi))   # A Gaussian
	return f;

'''=========================================================================='''

def V(x,y,t,psi,omega):
	V = (2+np.cos(omega*t))*(x**2)/2.0
	return V;

'''=========================================================================='''

def grid(Nx,Ny,xmax,ymax):
	x = np.linspace(-xmax, xmax-2*xmax/Nx, Nx)
	y = 0
	return x,y;

'''=========================================================================='''

def L(Nx,Ny,xmax,ymax):
	kx = np.linspace(-Nx/4/xmax, Nx/4/xmax-1/2/xmax, Nx)
	return (2*np.pi*1.j*kx)**2

'''=========================================================================='''

def absorb(x,y,xmax,ymax,dt,absorb_coeff):
	wx = xmax/20
	return np.exp(-absorb_coeff*(2-np.tanh((x+xmax)/wx)+np.tanh((x-xmax)/wx))*dt);

'''=========================================================================='''

def savepsi(Ny,psi):
	return abs(psi)**2

'''=========================================================================='''

def function(x):
    #return [1/ ((i) ** 2) for i in x]
    return [1/ ((i) ** 2) for i in x]

'''=========================================================================='''

'''
    omega                               : frequency of oscillation of the potential.
    Nx                                  : grid point.
    Ny                                  : grid point.
    dt                                  : time evolution step.
    xmax                                : x-window size.
    ymax 	                            : y-window size.
    dx                                  : position x step.
    ntp                                 : number of periods.
    images 		                        : number of .png images.
    absorb_coeff                        : 0 = periodic boundary

    output_choice                       : If 1, it plots on the screen but does not save the images.
        				                  If 2, it saves the images but does not plot on the screen.
        				                  If 3, it saves the images and plots on the screen.

    fixmaximum                          : fixes a maximum scale of |psi|**2 for the plots.
				                          If 0, it does not fix it.

    tmax                                : end of propagation.
    x, y                                : builds spatial grid.
    psi0                                : initial wavefunction.
    L                                   : Laplacian in Fourier space.
    linear_phase                        : linear phase in Fourier space (including point swap).
    border                              : absorbing shell at the border of the computational window.
    steps_image                         : number of computational steps between consecutive graphic outputs.
    savepsi_list                        : creates a vector to save the data of |psi|^2 for the final plot.
    psi_dictionnary                     : dictionnary to store the wavefunction at a time t and point x.
    sigma_gaussian_width                : array of the width of the gaussian at space x = 0.
    t_values                            : array of all the time values.
    beta_phase                          : the variation of the phase according to the integration of the square inverse of sigma.
    second_derivative_psi_relative_to_x : second derivative of psi relative to space x.
    first_derivative_psi_relative_to_t  : first derivative of psi relative to time t.
    mean_hamiltonian                    : mean value of the hamiltonian.
    invariant_action                    : invariant action.
    dw                                  : step in omega(t).
    list_omegas                         : all values of omega through time.
    berry_connection                    : bery connection.
    berry_phase                         : the integration of the berry connection.
    aharonov_phase                      :
    dynamical_phase                     :
    total_phase_change                  :
    normalisation_liste                 :
    diffrence_in_phase                  :
    test_chi2                           :

'''

omega                               = float(input('Enter the value of omega : '))
Nx                                  = 600   # 600
Ny                                  = Nx
dt                                  = 0.01 # 0.001
xmax                                = 10
ymax 	                            = xmax
dx                                  = 2*xmax/Nx
ntp                                 = 10
images 		                        = 300
absorb_coeff                        = 2000000
output_choice                       = 1
fixmaximum                          = 0

if omega==0 :
	tmax 	  = ntp*2*np.pi
else:
	tmax      = ntp*2*np.pi/omega

x, y                                = grid(Nx,Ny,xmax,ymax)
psi                                 = psi_0(x,y)
L                                   = L(Nx,Ny,xmax,ymax)
linear_phase                        = np.fft.fftshift(np.exp(1.j*L*dt/2))
border                              = absorb(x,y,xmax,ymax,dt,absorb_coeff)
steps_image                         = int(tmax/dt)
#savepsi_list                        = np.zeros((Nx,images+1))
savepsi_list                        = np.zeros((Nx,steps_image+1))
#steps_image                         = int(tmax/dt/images)
psi_dictionnary                     = {}
sigma_gaussian_width                = []
t_values                            = []
beta_phase                          = []
second_derivative_psi_relative_to_x = []
first_derivative_psi_relative_to_t  = []
mean_hamiltonian                    = []
invariant_action                    = []
dw                                  = []
list_omegas                         = []
berry_connection                    = []
berry_phase                         = []
aharonov_phase                      = []
dynamical_phase                     = []
total_phase_change                  = []
normalisation_liste                 = []
diffrence_in_phase                  = []
test_chi2                           = []

'''=========================================================================='''

for j in range(steps_image+1):
    t_values.append(j*dt)

    #if j%steps_image == 0:
    #    savepsi_list[:,int(j)] = savepsi(Ny,psi)
    savepsi_list[:,int(j)] = savepsi(Ny,psi)
    psi_dictionnary[j] = psi
    psi *= np.exp(-1.j*dt*V(x,y,j*dt,psi,omega))
    psi  = np.fft.fft(psi)
    psi *= linear_phase
    psi  = border*np.fft.ifft(psi)

    sigma_gaussian_width.append(   1/((np.sqrt(math.pi)* (abs(psi_dictionnary[j][int(Nx/2)])**2) ) ) )


'''=========================================================================='''


"""
plt.plot(t_values, sigma_gaussian_width , color='red')
plt.plot(t_values, function(sigma_gaussian_width) , color='blue')
plt.show()
plt.clf()
"""
for i in range(len(t_values)):
    if i == 0 :
        beta_phase.append(0)
    else :
        beta_phase.append( -0.5*simps( function(sigma_gaussian_width[:i]) , t_values[:i] , dt ) )

for i in range(len(t_values)):
    cache = []
    for w in range(len(x)):
        if w == 0 or w==len(x)-1:
            cache.append( 0 )
        else :
            cache.append( (psi_dictionnary[i][int(w-1)]-2*psi_dictionnary[i][w]+psi_dictionnary[i][int(w+1)])/(dx*dx) )
    second_derivative_psi_relative_to_x.append(cache)


for i in range(len(t_values)):
    mean_hamiltonian.append( ((simps( -0.5*np.conj(psi_dictionnary[i])* second_derivative_psi_relative_to_x[i]+ np.conj(psi_dictionnary[i])* (2+np.cos(omega*t_values[i]))*(x**2)* psi_dictionnary[i]*0.5 , x , dx ))).real )
    invariant_action.append(  ( (mean_hamiltonian[i]/np.sqrt((2+np.cos(omega*t_values[i])))).real ) )

for i in range(len(t_values)):
    if i==0 or i== len(t_values)-1:
        first_derivative_psi_relative_to_t.append( 0 )
    else:
        first_derivative_psi_relative_to_t.append( ((((psi_dictionnary[i+1][int(Nx/2)]/abs(psi_dictionnary[i+1][int(Nx/2)])))-((psi_dictionnary[i-1][int(Nx/2)]/abs(psi_dictionnary[i-1][int(Nx/2)]))))/(2*dt))*abs(psi_dictionnary[i][int(Nx/2)])   + (psi_dictionnary[i][int(Nx/2)]/abs(psi_dictionnary[i][int(Nx/2)]))*(abs(psi_dictionnary[i+1][int(Nx/2)])-abs(psi_dictionnary[i-1][int(Nx/2)]))/(2*dt) )



for i in range(len(t_values)):
    dw.append(abs( -np.sin(omega*t_values[i])*omega*dt/(2*np.sqrt(2+np.cos(omega*t_values[i]))) ) )

for i in range(len(t_values)):
    if i == 0 :
        berry_connection.append(0)
    else :
        berry_connection.append( np.conj(psi_dictionnary[i][int(Nx/2)])*first_derivative_psi_relative_to_t[i] )

for i in range(len(t_values)):
    list_omegas.append( np.sqrt(2+np.cos(omega*t_values[i])) )

for i in range(len(t_values)):
    if i == 0 :
        berry_phase.append( 0 )
    else :
        berry_phase.append(  simps( berry_connection[:i] , t_values[:i], dt)  )
"""
plt.plot(t_values, berry_connection)
plt.show()
"""

prop_coefficient = -0.5*(np.sqrt(3)-1/ sigma_gaussian_width[0]**2 )

#print(berry_phase)
dynamical_phase_other = []
for i in range(len(t_values)):
    if i == 0 :
        dynamical_phase.append(0)
        dynamical_phase_other.append( 0 )
    else :
        dynamical_phase.append( -simps( mean_hamiltonian[:i] , t_values[:i] , dt) )
        dynamical_phase_other.append( - mean_hamiltonian[i]*t_values[i] )

for i in range(len(t_values)):
    total_phase_change.append(berry_phase[i]+dynamical_phase[i] )

for i in range(len(t_values)):
    aharonov_phase.append(beta_phase[i] - dynamical_phase[i])


#plt.plot(t_values, beta_phase      , '--'  , color = 'red'       )
#plt.plot(t_values, true_phase_wf   , '--'   , color = 'green'      )
#plt.plot(t_values, dynamical_phase , '-.' , color = 'orange'     )
"""
plt.plot(t_values, berry_phase     , color = 'blue'       )

plt.plot(t_values, aharonov_phase  , color = 'aquamarine' )
plt.plot(t_values, test_calcul     , color = 'purple'     )
plt.show()
"""
print(' ')
print(' ')
print(omega)
print(' ')
print(' ')
print(aharonov_phase[int(len(t_values)-1)])
print(' ')
print(' ')
print(' ')

'''
    | Omega | aharonov |
    | 0     |  0.0000  |
    | 0.250 |  0.0059  |
    | 0.5   |  0.0159  |
    | 0.75  |  0.0549  |
    | 1.00  |  0.1393  |
    | 1.250 |  0.3366  |
    | 1.500 |  0.7355  |




'''
'''=========================================================================='''


for i in range(len(t_values)):
    diffrence_in_phase.append( (aharonov_phase[i].real-berry_phase[i].real)**2 )

for i in range(len(t_values)):
    if i == 0 :
        test_chi2.append(0)
    else :
        test_chi2.append( simps( diffrence_in_phase[:i] , t_values[:i],dt ) )

'''=========================================================================='''
for i in range(len(t_values)):
    normalisation_liste.append( simps( abs(psi_dictionnary[i])**2 , x, dx) )





plt.plot(t_values , beta_phase     , '--' , color = 'aquamarine' , label = 'Beta phase'    )

plt.plot(t_values , [-np.sqrt(3/4.0)*w for w in t_values ] , '--' , color = 'orange' , label = 'fit constant')
plt.plot(t_values , [-np.sqrt(1/4.0)*w for w in t_values ] , '--' , color = 'orange' )

plt.plot(t_values , berry_phase    , color = 'blue' , label = 'Berry phase'    )
plt.plot(t_values , aharonov_phase , color = 'red'  , label = 'Aharonov phase' )

plt.plot(t_values , dynamical_phase              , color = 'pink'   , label = 'Dynamical phase'        )
plt.plot(t_values , dynamical_phase_other , '--' , color = 'purple' , label = 'Dynamical approx phase' )



period_time = []
for i in range(ntp+1):
    if i == 0 :
        period_time.append(0)
        plt.vlines( period_time[i] , -0.5, +0.5 )
    else :
        period_time.append(i*tmax/ntp)
        plt.vlines( period_time[i] , -0.5, +0.5 )

plt.legend(loc="lower left", prop={'size': 9})
figname = "omage=%.3f"%omega+'.png'
plt.savefig(figname , dpi=600)

'''
plt.clf()
plt.plot(t_values , normalisation_liste)
plt.ylim(-0.1,1.1)
plt.show()
'''
'''=========================================================================='''
'''

  _______________________________________________
 | omega0 |  eta   |  bphase |  aphase |  chi2   |
 |¯¯¯¯¯¯¯¯|¯¯¯¯¯¯¯¯|¯¯¯¯¯¯¯¯¯|¯¯¯¯¯¯¯¯¯|¯¯¯¯¯¯¯¯¯|
 | 0.000  | 0.000  |  0      |  0      |   0     |
 | 0.250  | 0.061  |  0      |  0.0059 |         |
 | 0.500  | 0.121  |  0      |  0.0159 |         |
 | 0.750  | 0.182  |  0      |  0.0549 |         |
 | 1.000  | 0.242  |  0      |  0.1393 |         |
 | 1.250  | 0.303  |  0      |  0.3366 |         |
 | 1.500  | 0.363  |  0      |  0.7355 |         |
 | 1.750  | 0.424  |  0      |  0.     |         |
 | 2.000  | 0.484  |  0      |  0.     |         |
 ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯


'''
'''
    uncomment this section to plot the Berry and Aharonov phases
'''
list_eta            = [0 , 0.061  , 0.121  , 0.182  , 0.242  , 0.303  , 0.363  ]
list_aharonov_phase = [0 , 0.0059 , 0.0159 , 0.0549 , 0.1393 , 0.3366 , 0.7355 ]
fit_parameters = np.polyfit( list_eta[1:],np.log(list_aharonov_phase[1:]) , 1, w=np.sqrt(list_aharonov_phase[1:]) )

list_eta_num = np.linspace(0,0.37, num=50)

plt.clf()
plt.rcParams["figure.figsize"] = (15,15)
sns.set(rc={'axes.facecolor':'whitesmoke'})
plt.title('Representation of the Berry and Aharonov phase in function of the adiabatic coefficient $\eta$')

plt.plot(list_eta_num , [0*w for w in list_eta_num]                                                             , color = 'blue'       , label = 'Berry phase'    )
plt.plot(list_eta_num , [ np.exp(fit_parameters[1]) * np.exp(fit_parameters[0]*w)  for w in list_eta_num]       , color = 'red'        , label = 'Aharonov phase' )
plt.plot(list_eta     ,  list_aharonov_phase                                                              , 'x' , color = 'lightcoral' , label = 'data'           )

plt.axvspan(0, 0.01, facecolor='green', alpha=0.5, label='Adiabatic validity')
plt.ylabel('Phase in radians')
plt.xlabel('Adiabatic coefficient $\eta$')

plt.legend(loc="upper left", prop={'size': 9})
plt.savefig('berry_vs_aharonov.png', dpi=600)


'''=========================================================================='''



end = time.clock()
print('time taken to run the full program is ',end-start,' seconds')
