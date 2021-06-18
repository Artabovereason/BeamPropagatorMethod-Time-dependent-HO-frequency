import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import shutil
import platform
import seaborn as sns
sns.set(rc={'axes.facecolor':'whitesmoke'})

'''
	grid 	: define a grid.
	x       : x variable.
	y 		: set to 0, not used but a value must be given.
	L		: builds the Laplacian in Fourier space.
	kx		: x variable.

	asborb  : introduces an absorbing shell at the border of the computational window.

	savepsi : saves the data of abs(psi)**2 at different values of t.

	output  : defines graphic output: |psi|^2 is depicted.

'''

def grid(Nx,Ny,xmax,ymax):
	x = np.linspace(-xmax, xmax-2*xmax/Nx, Nx)
	y = 0
	return x,y;

def L(Nx,Ny,xmax,ymax):
	kx = np.linspace(-Nx/4/xmax, Nx/4/xmax-1/2/xmax, Nx)
	return (2*np.pi*1.j*kx)**2

def absorb(x,y,xmax,ymax,dt,absorb_coeff):
	wx = xmax/20
	return np.exp(-absorb_coeff*(2-np.tanh((x+xmax)/wx)+np.tanh((x-xmax)/wx))*dt);


def savepsi(Ny,psi):
	return abs(psi)**2

def output(x,y,psi,omega,n,t,folder,output_choice,fixmaximum):
	# Number of figure
	if (output_choice==2) or (output_choice==3):
		num =str(int(n))
		if n < 100:
			num ='0'+str(int(n))
		if n < 10:
			num ='00'+str(int(n))

	# plot the line chart
	plt.rcParams["figure.figsize"] = (13,13)
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
	#f_ax1.plot(t_val            , energy_moyenne                                                   , color='aquamarine' , label = '<$H$>'      )
	#f_ax1.plot(coordonnee_temps , [(2+0.1*np.cos(omega*w))*0.5 for w in coordonnee_temps]  , '--'  , color='green'      , label = 'fit'        )
	#f_ax1.plot(t_val            , [(2+0.1*np.cos(omega*w))*0.5 for w in t_val]                     , color='blue'       , label = 'potential'  )
	#f_ax1.legend(loc="upper right", prop={'size': 9})

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

	axs[1,0].set_title('Adiabatic coefficient')
	axs[1,0].set_ylabel(' ')
	axs[1,0].set_xlabel('Frequency $\omega$')
	#axs[1,0].plot(list_frequency_omega , list_adiabatic_coefficient  , color='blue' )

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
	axs[1,1].set_title('System representation at $t=$%.3f' %t)
	axs[1,1].set_ylabel(' ')
	axs[1,1].set_xlabel('space $x$')
	axs[1,1].plot(x, abs(psi)**2                    , color = 'blue' , label = 'wavefunction groundstate' )
	axs[1,1].plot(x, (2+np.cos(omega*t))*(x**2)/2.0 , color = 'red'  , label = 'potential'                )
	axs[1,1].set_ylim(-0.5,2.5)
	#axs[1,1].legend(loc="best", prop={'size': 9})

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
	#axs[1,2].plot(t    , sigma_width           , color='blue' , label= 'width' )
	#axs[1,2].legend(loc="upper right", prop={'size': 9})

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
	#axs[2,0].plot(t_val   , berry_phase         , color='blue' , label ='phase'  )
	#axs[2,0].legend(loc="best", prop={'size': 9})

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
	#axs[2,1].plot(t_val[1:len(t_val)-1], autre_derivee[1:len(t_val)-1], color='black'  , label ='$alpha(t)$'  )
	#axs[2,1].legend(loc="upper right", prop={'size': 9})


	'''======================================================================'''
	''' axs[2,2]
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
	#axs[2,2].plot(t_val                 , autretest_valeur                 , color='orange'    , label = 'I(t)'                                         )
	#axs[2,2].legend(loc="best", prop={'size': 9})


	'''======================================================================'''
	plt.subplots_adjust(
	                    left=0.1,
	                    bottom=0.1,
	                    right=0.9,
	                    top=0.9,
	                    wspace=0.3,
	                    hspace=0.3 # 0.7
	                    )

	# Saves figure
	if (output_choice==2) or (output_choice==3):
		figname = folder+'/fig'+num+'.png'
		plt.savefig(figname)

	# Displays on screen
	if (output_choice==1) or (output_choice==3):
		plt.show(block=False)
		fig.canvas.flush_events()

	return;

# Some operations after the computation is finished: save the final value of psi, generate videos and builds
# the final plot: a contour map of the y=0 cut as a function of x and t

def final_output(folder,x,Deltat,psi,savepsi,output_choice,images,fixmaximum):


	np.save(folder,psi)		# saves final wavefunction

	if (output_choice==2) or (output_choice==3):
		movie(folder)	                        # creates video

	# Now we make a plot of the evolution depicting the 1D cut at y=0
	tvec=np.linspace(0,Deltat*images,images+1)
	tt,xx=np.meshgrid(tvec,x)
	figtx = plt.figure("Evolution of |psi(x)|^2")              # figure
	plt.clf()                # clears the figure
	figtx.set_size_inches(8,6)



    # Generates the plot
	toplot=savepsi
	if fixmaximum>0:
		toplot[toplot>fixmaximum]=fixmaximum

	plt.contourf(xx, tt, toplot, 100, cmap=cm.jet, linewidth=0, antialiased=False)
	cbar=plt.colorbar()               # colorbar
	plt.xlabel('$x$')                 # axes labels, title, plot and axes range
	plt.ylabel('$t$')
	cbar.set_label('$|\psi|^2$',fontsize=14)

	figname = folder+'/sectx.png'
	plt.savefig(figname)    # Saves the figure
	plt.show()      # Displays figure on screen


# Generates video from the saved figures. This function is called by final_output
def movie(folder):
	folder.replace('.','')
	examplename=folder[13:]


	video_options='vbitrate=4320000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq'

	if platform.system() == 'Windows':
		try:
			shutil.copyfile('mencoder.exe', folder+'/mencoder.exe')
			os.chdir(folder)
			command ='mencoder "mf://fig*.png" -mf w=800:h=600:fps=25:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o movie_'+examplename+'.avi'
			os.system(command)
			try:
				os.remove('mencoder.exe')
			except:
				print("Could not delete mencoder.exe in examlpe directory")

			os.chdir('../../')
		except:
			print("Error making movie with mencoder in windows")

	else:
		try:
			command1 ='mencoder "mf://'+folder+'/fig*.png" -mf fps=25 -o /dev/null -ovc lavc -lavcopts vcodec=mpeg4:vpass=1:'+video_options
			command2 ='mencoder "mf://'+folder+'/fig*.png" -mf fps=25 -o ./'+folder+'/movie_'+examplename+'.avi -ovc lavc -lavcopts vcodec=mpeg4:vpass=2:'+video_options
			os.system(command1)
			os.system(command2)
		except:
			print("Error making movie with mencoder in Linux")




	## delete temporary files:
	try:
		os.remove('divx2pass.log')
	except:
		pass

	try:
		shutil.rmtree('__pycache__')
	except:
		pass

	try:
		shutil.rmtree('examples1D/__pycache__/')
	except:
		pass
