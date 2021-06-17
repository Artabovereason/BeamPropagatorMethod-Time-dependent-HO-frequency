# Integrating a 1+1D or 1+2D NLSE with different initial conditions and for different potentials.


import numpy as np
import sys
import os
import importlib
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'axes.facecolor':'whitesmoke'})
import math
from scipy.integrate   import simps

dt = 0.001                    # Evolution step
dx = 1/60.0

# Preliminaries (handling directories and files)
sys.path.insert(0, './examples'+sys.argv[2])          # adds to path the directory with examples
output_folder = './examples'+sys.argv[2]+'/'+sys.argv[1]  # directory for images and video output
if not os.path.exists(output_folder):                     # creates folder if it does not exist
    os.makedirs(output_folder)

try:              # Erase all image files (if exist) before starting computation and generating new output
    for filename in glob.glob(output_folder+'/*.png') :
        os.remove( filename )
except:
    pass

my = importlib.__import__(sys.argv[1])             # imports the file with the details for the computation
build = importlib.__import__(sys.argv[2])         # selects 1D or 2D


'''=========================================================================='''
'''

Initialization of the computation

    x , y                : builds spatial grid
    psi                  : loads initial condition
    L                    : Laplacian in Fourier space
    linear_phase         : linear phase in Fourier space (including point swap)
    border               : Absorbing shell at the border of the computational window
    savepsi              : Creates a vector to save the data of |psi|^2 for the final plot
    steps_image          : Number of computational steps between consecutive graphic outputs
    omega                : frequency of oscillation of the potential
    psi_dictionnary      : array to store the psi values at all time and all x
    sigma_gaussian_width : array of the width of the gaussian at space x=0
    t_values             : array of all the time values
'''

x, y                 = build.grid(my.Nx,my.Ny,my.xmax,my.ymax)
psi                  = my.psi_0(x,y)
L                    = build.L(my.Nx,my.Ny,my.xmax,my.ymax)
linear_phase         = np.fft.fftshift(np.exp(1.j*L*my.dt/2))
border               = build.absorb(x,y,my.xmax,my.ymax,my.dt,my.absorb_coeff)
savepsi              = np.zeros((my.Nx,my.images+1))
steps_image          = int(my.tmax/my.dt/my.images)
omega                = float(input('Enter the value of omega : '))
psi_dictionnary      = {}
sigma_gaussian_width = []
t_values             = []

for j in range(steps_image*my.images+1):
    t_values.append(j*my.dt)

# Main computational loop
print("calculating", end="", flush=True)
for j in range(steps_image*my.images+1):        # propagation loop
    if j%steps_image == 0:  # Generates image output
        #build.output(x,y,psi,omega,int(j/steps_image),j*my.dt,output_folder,my.output_choice,my.fixmaximum)
        #savepsi[:,int(j/steps_image)]=build.savepsi(my.Ny,psi)
        print(".", end="", flush=True)
    V = my.V(x,y,j*my.dt,psi,omega)            # potential operator
    psi *= np.exp(-1.j*my.dt*V)            # potential phase
    if sys.argv[2] == "1D":
        psi = np.fft.fft(psi)            # 1D Fourier transform
        psi *=linear_phase                # linear phase from the Laplacian term
        psi = border*np.fft.ifft(psi)    # inverse Fourier transform and damping by the absorbing shell
        psi_dictionnary[j] = psi
        sigma_gaussian_width.append((1/(np.sqrt(math.pi)*max(psi_dictionnary[j].real))))

    """
    elif sys.argv[2] == "2D":
        psi = np.fft.fft2(psi)            # 2D Fourier transform
        psi *=linear_phase                # linear phase from the Laplacian term
        psi = border*np.fft.ifft2(psi)    # inverse Fourier transform and damping by the absorbing shell
    """
    else:
        print("Not implemented")

# Final operations
# Generates some extra output after the computation is finished and save the final value of psi:


'''=========================================================================='''
'''
    sigma_gaussian_width_log_dot : array with the logarithmic derivative relative
                                   to time of the width of the gaussian.
'''
sigma_gaussian_width_log_dot = []


for i in range(len(t_values)):
    if i == 0 or i == len(t_values)-1:
        sigma_gaussian_width_log_dot.append(0)
    else:
        sigma_gaussian_width_log_dot.append(  ( np.log(sigma_gaussian_width[i-1])- np.log(sigma_gaussian_width[i+1]))/(4*dt)     )

def function(x):
    return [-1/ ((i) ** 2) for i in x]

berry_phase = []
new_new_t_values = []
berry_phase.append(0)
new_new_t_values.append(0)
for i in range(1,len(t_values),10):
    new_new_t_values.append(t_values[i])
    #print(i)
    berry_phase.append(simps( function(sigma_gaussian_width[:i]) , t_values[:i] , dx=dx ) )


seconde_derivative_psi = []
for i in range(0,int(len(t_values))):
    #print('i=%.1f'%i)
    derivee_seconde_prems = []
    derivee_seconde_prems.append(0)
    for w in range(int(1),int(len(x)-1)):
        derivee_seconde_prems.append( (psi_dictionnary[i][int(w-1)]-2*psi_dictionnary[i][w]+psi_dictionnary[i][int(w+1)])/(dx*dx)    )
    derivee_seconde_prems.append(0)
    seconde_derivative_psi.append(derivee_seconde_prems)

mean_hamiltonian = []
new_t_values = []
invariant_action=[]
for i in range(0,len(t_values),10):
    new_t_values.append(t_values[i])
    mean_hamiltonian.append( (simps( -0.5*np.conj(psi_dictionnary[i])* seconde_derivative_psi[i]+ np.conj(psi_dictionnary[i])* (2+np.cos(omega*t_values[i]))*(x**2)* psi_dictionnary[i]*0.5 , x , dx )).real )
    invariant_action.append(   (mean_hamiltonian[int(i/10)]/(2+np.cos(omega*t_values[i]))).real  )




'''=========================================================================='''

plt.rcParams["figure.figsize"] = (13,13)             #size of the output picture

# plot the line chart
fig, axs = plt.subplots(3,3)      #define the picture to be a grid of 3*3
gs       = fig.add_gridspec(3, 3) # i don't know
st       = fig.suptitle("Quantum Harmonic Oscillator with time-dependent frequency, $\omega$=%.2f"%omega , fontsize=20) # name the whole 3*3 plot by a big suptitle

'''=========================================================================='''
''' the name to plot here is : f_ax1
—————————————
| XXXXXXXXX |
—————————————
|   |   |   |
—————————————
|   |   |   |
—————————————

    Eigenvalue : the plot of the eigenvalues of the system
    <H>        : mean value of the energy
    fit        : correct theory potential energy
    potential  : numerical plot of the potential

It is intended that fit en potential tend to the same curve.
But, eigenvalue and <H> could differ in some cases.

'''
f_ax1 = fig.add_subplot(gs[0, :]) # used to make a big plot of 3 colon-wide
'''
    This whole section allows the grid [0,0], [0,1] and [0,2] to be unseen and
    only the f_ax1 to be seen.
'''
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])

f_ax1.set_title('Total and potential energy fluctuations over time')
f_ax1.set_ylabel(' ')
f_ax1.set_xlabel('time $t$ in s')
#f_ax1.plot(t_val            , Eigenvalue_stock                                                 , color='black'      , label = 'Eigenvalue' )
f_ax1.plot(new_t_values , mean_hamiltonian                            , color='aquamarine' , label = '<$H$>'     )
#f_ax1.plot(coordonnee_temps , [(2+0.1*np.cos(omega*w))*0.5 for w in coordonnee_temps]  , '--'  , color='green'      , label = 'fit'        )
f_ax1.plot(t_values     , [(2+np.cos(omega*w))*0.5 for w in t_values] , color='blue'       , label = 'potential' )
f_ax1.legend(loc="upper right", prop={'size': 9})

'''======================================================================'''
''' the name to plot here is : axs[1,0]
—————————————
|   |   |   |
—————————————
| X |   |   |
—————————————
|   |   |   |
—————————————

This plot will always be the same, since it's based on static array,
it is used to show how-adiabatic our case is.
'''

"""axs[1,0].set_title('Adiabatic coefficient')
axs[1,0].set_ylabel(' ')
axs[1,0].set_xlabel('Frequency $\omega$')"""


'''======================================================================'''
''' the name to plot here is : axs[1,1]
—————————————
|   |   |   |
—————————————
|   | X |   |
—————————————
|   |   |   |
—————————————

    potential       : plot a representative of the potential well acting on the wavefunction
    wavefunction gs : wavefunction groundstate representation

The main idea here is to show at t=0 the situation. A gif could be outputed to
see the evolution through time but it is not time-efficient.
'''
axs[1,1].set_title('System representation at $t=0$')
axs[1,1].set_ylabel(' ')
axs[1,1].set_xlabel('space $x$')
axs[1,1].plot(x, abs(psi_dictionnary[0])**2                    , color = 'blue' , label = 'wavefunction groundstate' )

axs[1,1].plot(x, (2+1)*(x**2)/2.0 , color = 'red'  , label = 'potential' )
axs[1,1].set_ylim(-0.5,2.5)
axs[1,1].legend(loc="best", prop={'size': 9})

'''======================================================================'''
''' the name to plot here is : axs[1,2]
—————————————
|   |   |   |
—————————————
|   |   | X |
—————————————
|   |   |   |
—————————————

    sigma_width : widt of the gaussian through time

'''
axs[1,2].set_title('Width of the Gaussian function through time')
axs[1,2].set_ylabel(' ')
axs[1,2].set_xlabel('time $t$ in s')
axs[1,2].plot(t_values, sigma_gaussian_width           , color = 'blue' , label = 'sigma' )
axs[1,2].legend(loc="upper right", prop={'size': 9})

'''======================================================================'''
''' the name to plot here is : axs[2,0]
—————————————
|   |   |   |
—————————————
|   |   |   |
—————————————
| X |   |   |
—————————————
    berry_phase : plot the berry phase through time
'''

axs[2,0].set_title('Berry phase through time')
axs[2,0].set_ylabel('$beta(t)$ in degrees')
axs[2,0].set_xlabel('time $t$ in s')
axs[2,0].plot(new_new_t_values   , berry_phase         , color='blue' , label ='phase'  )
axs[2,0].legend(loc="best", prop={'size': 9})

'''======================================================================'''
''' axs[2,1]
—————————————
|   |   |   |
—————————————
|   |   |   |
—————————————
|   | X |   |
—————————————

    alpha(t) : plot the alpha coefficient defined as the quotient of the derivative
    relative to time of sigma divided by sigma.

This plot is pretty noisy, but on it can be seen a periodc
'''

axs[2,1].set_title('alpha coefficient through time')
axs[2,1].set_ylabel(' ')
axs[2,1].set_xlabel('time $t$ in s')
axs[2,1].plot(t_values[1:int(len(t_values)-1)], sigma_gaussian_width_log_dot[1:int(len(t_values)-1)]                    , color = 'blue' , label = 'alpha(t)')  # makes the plot
axs[2,1].legend(loc="upper right", prop={'size': 9})


'''======================================================================'''
'''axs[2,2]
—————————————
|   |   |   |
—————————————
|   |   |   |
—————————————
|   |   | X |
—————————————
'''

axs[2,2].set_title('Invariant plot through time')
axs[2,2].set_ylabel(' ')
axs[2,2].set_ylim(0,3)
axs[2,2].set_xlabel('time $t$ in s')
#axs[2,2].plot(t_val[2:len(t_val)-1] , Omega_invar[1:len(Omega_invar)]  , color = 'black'   , label = '$\Omega$'                                     )
#axs[2,2].plot(t_val[1:len(t_val)-1] , valeur_moyenne_I[1:len(t_val)-1] , color = 'red'     , label = '$I(t)$'                                       )
#axs[2,2].plot(t_val[1:len(t_val)-1] , mean_I[1:len(t_val)-1]           , color = 'yellow'  , label = '$mean I(t)=$%.3f'%Average(valeur_moyenne_I  ) )
axs[2,2].plot(new_t_values                 , invariant_action                 , color='orange'    , label = '$<H>/\omega$'                                         )
axs[2,2].legend(loc="best", prop={'size': 9})


'''======================================================================'''
plt.subplots_adjust(
                    left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3 # 0.7
                    )

filename = 'omega=%.3f'%omega+'.png'
    # save frame
plt.savefig(filename,transparent=True)
plt.close()




#build.final_output(output_folder,x,steps_image*my.dt,psi,savepsi,my.output_choice,my.images,my.fixmaximum)
