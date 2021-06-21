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
import time
start = time.clock()
dt   = 0.01                    # Evolution step
Nx   = 600
xmax = 5
dx   = 2*xmax/Nx
'''=========================================================================='''
'''

Preliminaries (handling directories and files)

    my    : imports the file with the details for the computation
    build : selects 1D or 2D
'''

sys.path.insert(0, './examples'+sys.argv[2])          # adds to path the directory with examples
output_folder = './examples'+sys.argv[2]+'/'+sys.argv[1]  # directory for images and video output
if not os.path.exists(output_folder):                     # creates folder if it does not exist
    os.makedirs(output_folder)

try:              # Erase all image files (if exist) before starting computation and generating new output
    for filename in glob.glob(output_folder+'/*.png') :
        os.remove( filename )
except:
    pass

my    = importlib.__import__(sys.argv[1]) #
build = importlib.__import__(sys.argv[2]) # 1D


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
    if j%steps_image == 0:            # Generates image output
        #build.output(x,y,psi,omega,int(j/steps_image),j*my.dt,output_folder,my.output_choice,my.fixmaximum)
        savepsi[:,int(j/steps_image)]=build.savepsi(my.Ny,psi)
        print(".", end="", flush=True)
    V = my.V(x,y,j*my.dt,psi,omega)   # potential operator
    psi *= np.exp(-1.j*my.dt*V)       # potential phase
    if sys.argv[2] == "1D":
        psi = np.fft.fft(psi)         # 1D Fourier transform
        psi *=linear_phase            # linear phase from the Laplacian term
        psi = border*np.fft.ifft(psi) # inverse Fourier transform and damping by the absorbing shell
        psi_dictionnary[j] = psi
        sigma_gaussian_width.append((1/(np.sqrt(math.pi)*(np.conj(psi_dictionnary[j][int(Nx/2)])*psi_dictionnary[j][int(Nx/2)]).real ) ) )
    else:
        print("Not implemented")

# Final operations
# Generates some extra output after the computation is finished and save the final value of psi:


'''=========================================================================='''
'''
    sigma_gaussian_width_log_dot : array with the logarithmic derivative relative
                                   to time of the width of the gaussian.
    normalisation_liste          : list of the integral of |psi|^2 over x

    Average                      : return average of a list
'''
sigma_gaussian_width_log_dot = []
normalisation_liste          = []
berry_phase                  = []
new_new_t_values             = []
seconde_derivative_psi       = []
mean_hamiltonian             = []
new_t_values                 = []
invariant_action             = []

def Average(lst):
    return sum(lst) / len(lst)

for i in range(len(t_values)):
    normalisation_liste.append( (simps( np.conj(psi_dictionnary[i])*(psi_dictionnary[i]) ,x , dx)).real )

for i in range(len(sigma_gaussian_width)):
    sigma_gaussian_width[i]=sigma_gaussian_width[i]/normalisation_liste[i]


for i in range(len(t_values)):
    if i == 0 or i == len(t_values)-1:
        sigma_gaussian_width_log_dot.append(0)
    else:
        sigma_gaussian_width_log_dot.append(  ( np.log(sigma_gaussian_width[i-1])- np.log(sigma_gaussian_width[i+1]))/(4*dt)     )

def function(x):
    return [1/ ((i) ** 2) for i in x]


new_new_t_values.append(0)
for i in range(len(t_values)):
    if i == 0 :
        berry_phase.append(0)
    else :
        new_new_t_values.append(t_values[i])
        berry_phase.append( -0.5*simps( function(sigma_gaussian_width[:i]) , t_values[:i] , dt ) )



for i in range(0,int(len(t_values))):
    derivee_seconde_prems = []
    derivee_seconde_prems.append(0)
    for w in range(int(1),int(len(x)-1)):
        derivee_seconde_prems.append( (psi_dictionnary[i][int(w-1)]-2*psi_dictionnary[i][w]+psi_dictionnary[i][int(w+1)])/(dx*dx)    )
    derivee_seconde_prems.append(0)
    seconde_derivative_psi.append(derivee_seconde_prems)

'''=========================================================================='''
'''
    <H>
    <H>/omega
'''

for i in range(len(t_values)):
    new_t_values.append(t_values[i])
    mean_hamiltonian.append( ((simps( -0.5*np.conj(psi_dictionnary[i])* seconde_derivative_psi[i]+ np.conj(psi_dictionnary[i])* (2+np.cos(omega*t_values[i]))*(x**2)* psi_dictionnary[i]*0.5 , x , dx ))).real )
    invariant_action.append(   (mean_hamiltonian[int(i)]/np.sqrt((2+np.cos(omega*t_values[int(i)])))).real  )



'''=========================================================================='''
'''
    Quantum mean of (1/sigma2+dot(sigma)2)<x2>

    first_mean_value : array to store that mean.

'''

first_mean_value = []


for i in range(len(t_values)):
    if i==0 or i==int(len(t_values)-1):
        first_mean_value.append( simps( np.conj(psi_dictionnary[i])*( (x**2)/(sigma_gaussian_width[i]**2)  )*psi_dictionnary[i] , x, dx) )
    else:
        first_mean_value.append( simps( np.conj(psi_dictionnary[i])*( ((x**2)/(sigma_gaussian_width[i]**2))+(x**2)*(( (sigma_gaussian_width[i-1]-sigma_gaussian_width[i+1])/(2*dt) )**2  )  )*psi_dictionnary[i] , x, dx) )

'''
    Quantum mean of sigma2 <second derivative relative to x>

    secnd_mean_value      : array to store that mean.
    secnd_derivative_psi  : array to store the second derivative relative to x of psi.
    derivee_seconde_prems : a cache-value in order to append the second derivative of psi.
'''
secnd_mean_value     = []
secnd_derivative_psi = []

for i in range(int(len(t_values))):
    derivee_seconde_prems = []
    derivee_seconde_prems.append(0)
    for w in range(int(1),int(len(x)-1)):
        derivee_seconde_prems.append(  (psi_dictionnary[i][int(w-1)]-2*psi_dictionnary[i][int(w)]+psi_dictionnary[i][int(w+1)])/(dx*dx)    )
    derivee_seconde_prems.append(0)
    secnd_derivative_psi.append(derivee_seconde_prems)

for i in range(len(t_values)):
    if i==0 or i==int(len(t_values)-1):
        secnd_mean_value.append(0)
    else:
        secnd_mean_value.append( (sigma_gaussian_width[i]**2)*simps( np.conj(psi_dictionnary[i])*secnd_derivative_psi[i]  , x, dx)  )

'''
    Quantum mean of i sigma dot(sigma) (<x derivative x> +1 )

    third_mean_value : array to store that mean.
    derivative_sigma : array to store the derivative of sigma relative to x.
    derivative_psi   : array to store the derivative of psi relative to x.

'''
third_mean_value = []
derivative_sigma = []
derivative_psi   = []

for i in range(int(len(t_values))):
    if i==0 or i==int(len(t_values)-1):
        derivative_sigma.append(0)
    else:
        derivative_sigma.append(  (sigma_gaussian_width[i-1]-sigma_gaussian_width[i+1])/(2*dt) )

for i in range(int(len(t_values))):
    derivee_premier_prems = []
    derivee_premier_prems.append(0)
    for w in range(int(1),int(len(x)-1)):
        derivee_premier_prems.append(  (psi_dictionnary[i][int(w-1)]-psi_dictionnary[i][int(w+1)])/(2*dx)    )
    derivee_premier_prems.append(0)
    derivative_psi.append(derivee_premier_prems)



for i in range(int(len(t_values))):
    if i==0 or i==int(len(t_values)-1):
        third_mean_value.append(0)
    else:
        third_mean_value.append( 1j*simps(sigma_gaussian_width[i]*derivative_sigma[i]*(2*np.conj(psi_dictionnary[i])*x*(derivative_psi[i])+np.conj(psi_dictionnary[i])*psi_dictionnary[i] ) , x ,dx ) )


'''
    Quantum mean of the Ermakov's invariant I(t)

    mean_value_I : array to store that mean.

'''
mean_value_I = []

for i in range(int(len(t_values))):
    mean_value_I.append( 0.5*(first_mean_value[i]-secnd_mean_value[i]+third_mean_value[i]) )


print(' ')
print('Mean of invariant action is :%.3f'%Average(invariant_action))
#print(mean_value_I)
print('Mean of I is :%.3f'%Average(mean_value_I))

dynamical_phase = []
values_time     = []
for i in range(1,int(len(t_values))):
    values_time.append(t_values[i])
    dynamical_phase.append( -simps( mean_hamiltonian[:i] , t_values[:i], dt))

alternate_berry = []
print(len(t_values))
print(len(berry_phase))
print(len(dynamical_phase))
for i in range(1,int(len(t_values)-1)):
    alternate_berry.append(berry_phase[i]-dynamical_phase[i])


#total_phase = []
#for i in range(1,int(len(t_values)-2)):
#    total_phase.append(dynamical_phase[i]+berry_phase[i])

adiabatic_coefficient_eta = []
for i in range(len(t_values)):
    if i == 0 or i == int(len(t_values)-1):
        adiabatic_coefficient_eta.append(0)
    else:
        adiabatic_coefficient_eta.append( ((np.sqrt(2+np.cos(omega*t_values[i-1]))-np.sqrt(2+np.cos(omega*t_values[i+1]))  )/(2*dt))/(2+np.cos(omega*t_values[i]))    )

print('The adiabatic coefficient for omega=%.3f'%omega+' is =%.3f'% max(adiabatic_coefficient_eta))
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
'''
f_ax1 = fig.add_subplot(gs[0, :]) # used to make a big plot of 3 colon-wide
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])

f_ax1.set_title('Total and potential energy fluctuations over time')
f_ax1.set_ylabel(' ')
f_ax1.set_xlabel('time $t$ in s')
f_ax1.plot(new_t_values , mean_hamiltonian                            , color='aquamarine' , label = '<$H$>'     )
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
'''

axs[1,0].set_title('normalisation')
axs[1,0].set_ylabel(' ')
axs[1,0].set_xlabel('time $t$')
axs[1,0].plot(t_values, normalisation_liste , color = "red")


'''======================================================================'''
''' the name to plot here is : axs[1,1]
—————————————
|   |   |   |
—————————————
|   | X |   |
—————————————
|   |   |   |
—————————————
'''
axs[1,1].set_title('System representation at $t=0$')
axs[1,1].set_ylabel(' ')
axs[1,1].set_xlabel('space $x$')
axs[1,1].plot(x , abs(psi_dictionnary[0])**2 , color = 'blue' , label = 'wavefunction groundstate' )
axs[1,1].plot(x , (2+1)*(x**2)/2.0           , color = 'red'  , label = 'potential'                )
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
'''

axs[2,0].set_title('Berry phase through time')
axs[2,0].set_ylabel('$beta(t)$ in degrees')
axs[2,0].set_xlabel('time $t$ in s')
axs[2,0].plot(new_new_t_values , berry_phase     , color='blue'   , label ='berry phase'     )
axs[2,0].plot(values_time      , dynamical_phase , color='orange' , label ='dynamical phase' )
axs[2,0].plot(values_time[:int(len(values_time)-1)]      , alternate_berry , color='green' , label =' ' )
#axs[2,0].plot(values_time[0:int(len(values_time)-1)]      , total_phase     , color='green'  , label ='total phase'     )
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
axs[2,2].plot(new_t_values , invariant_action , color = 'orange' , label = '$<H>/\omega$=%.3f'%Average(invariant_action) )
axs[2,2].plot(t_values[1:int(len(t_values)-1)]     , mean_value_I[1:int(len(t_values)-1)]      , color = 'blue'   , label = '$<I(t)>=$%.3f'    %Average(mean_value_I)     )
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
#plt.savefig(filename,transparent=True)
plt.savefig(filename)
plt.close()

build.final_output(output_folder,x,steps_image*my.dt,psi,savepsi,my.output_choice,my.images,my.fixmaximum)

end = time.clock()
print('time taken to run the full program is ',end-start,' seconds')
