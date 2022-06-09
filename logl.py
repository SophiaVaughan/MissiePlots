import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

from scipy import stats

######################################################################################
############################# Private Functions ######################################
######################################################################################
				
def _plotadmin(ax, kwargs):

	'''
	This is a helper function to add a title and axis labels to a plot or subplot as well as alter the axis ticks and lims. 
	
	Parameters
	----------
	ax : matplotlib.pyplot.axis
		The matplotlib axis on to add the labels.
	kwargs : dict
		This function only uses the following kwargs.
		font_scale : int or float, default=1
			A scale factor by which to increase all the fontsizes of the labels in the plot.
		title : str, optional
			The title of the plot. If this keyword isn't used, no title will be added.
		xlabel : str, optional
			The label for the x axis of the plot. If this keyword isn't used, no label will be added.
		ylabel : str, optional
			The label for the y axis of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
	'''

	### Add a title to the axis is the 'title' and 'font_scale' keywords are set ###
	_adjustplot( lambda kwargs: ax.set_title(kwargs['title'], fontsize=16*kwargs['font_scale']) , kwargs, ['title', 'font_scale'], 'setting title')
	
	### Add a label/change the axis ticks/set the limits of the x axis if the and 'xlabel' and 'font_scale'/'xticks' and 'font_scale'/'xlim' keywords are set ###
	_adjustplot( lambda kwargs: ax.set_xlabel(kwargs['xlabel'], fontsize=12*kwargs['font_scale']) , kwargs, ['xlabel', 'font_scale'], 'setting x label')
	_adjustplot( lambda kwargs: ax.set_xticks(kwargs['xticks'], fontsize=12*kwargs['font_scale']) , kwargs, ['xticks', 'font_scale'], 'setting x ticks')
	_adjustplot( lambda kwargs: ax.set_xlim(kwargs['xlim']) , kwargs, ['xlim', 'font_scale'], 'setting x lim')
	
	### Add a label/change the axis ticks/set the limits of the y axis if the and 'ylabel' and 'font_scale'/'yticks' and 'font_scale'/'ylim' keywords are set ###
	_adjustplot( lambda kwargs: ax.set_ylabel(kwargs['ylabel'], fontsize=12*kwargs['font_scale']) , kwargs, ['ylabel', 'font_scale'], 'setting y label')
	_adjustplot( lambda kwargs: ax.set_yticks(kwargs['yticks'], fontsize=12*kwargs['font_scale']) , kwargs, ['yticks', 'font_scale'], 'setting y ticks')
	_adjustplot( lambda kwargs: ax.set_ylim(kwargs['ylim']) , kwargs, ['ylim', 'font_scale'], 'setting y lim')


######################################################################################
	
def _makecbar(fig, func, logl, kwargs):

	'''
	This is a helper function to add a colorbar to the side of the image.
	
	Parameters
	----------
	fig : matplotlib.pyplot.figure
		The figure to add the colorbar to.
	func : callable
		A function that takes the array and converts the log likelihood values to significance. Is called like this: sig = func(arry).
	logl : np.ndarray or list
		A 2D array of delta log likelihoods equivelent to significances.
	kwargs : dict
		This function only uses the following kwargs.
		font_scale : int or float, default=1
			A scale factor by which to increase all the fontsizes of the labels in the plot.
		siglabel : str, optional
			The label for the 'significance' side of the colorbar of the plot. If this keyword isn't used, the default "Significance" label will be added.
		logllabel : str, optional
			The label for the 'Log Likelihood' side of the colorbar of the plot. If this keyword isn't used, the default "Log Likelihood" label will be added.
		sigticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		loglticks : list or np.ndarray, optional
			A list/array of ticks in loglikelihhod to add to the log likelihood colorbar. If this keyword isn't used, the default ticks will be used.
		siglim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
	'''
	
	### Move plot to side to make room for colorbar ###
	fig.subplots_adjust(right=0.725)
	### Make a new axis for the colorbar ###
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	
	### Conver logl to sig ###
	sig = func(logl)
	
	### Set up the cmap properly if norm is set or not ###
	if kwargs['norm'] is not None: cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N), cmap=kwargs['cmap']), cax=cbar_ax)
	else: cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(sig), vmax=np.max(sig)), cmap=kwargs['cmap']), cax=cbar_ax)

	### Add a label/change the axis ticks/set the limits of the colorbar if the and 'siglabel' and 'font_scale'/'sigticks' and 'font_scale'/'siglim' keywords are set ###
	_adjustplot( lambda kwargs: cbar.set_label(kwargs['siglabel'], fontsize=12*kwargs['font_scale'], rotation=-90, labelpad=15) , kwargs, ['siglabel', 'font_scale'], 'setting sig label')
	_adjustplot( lambda kwargs: cbar.set_ticks(kwargs['sigticks'], fontsize=12*kwargs['font_scale']) , kwargs, ['sigticks', 'font_scale'], 'setting sig ticks')
	_adjustplot( lambda kwargs: cbar.set_lim(kwargs['siglim'], fontsize=12*kwargs['font_scale']) , kwargs, ['siglim', 'font_scale'], 'setting sig lim')
	
	### Get the position of the colorbar and it's axis object ###
	pos = cbar.ax.get_position()
	ax1 = cbar.ax
	ax1.set_aspect('auto')

	### Dulicate the axis ###
	ax2 = ax1.twinx()
	### Match the limits of the new axis to the old one ###
	ax2.set_ylim(ax1.get_ylim())
	
	### Add the mark points as axis ticks ###
	ax2.set_yticks(func(np.array(kwargs['loglticks'])))
	### Add the labels for the marks as tick labels ###
	ax2.set_yticklabels(kwargs['loglticks'], fontsize=10*kwargs['font_scale'])
	
	### set the positions of the two axis to match the colorbar
	ax1.set_position(pos)
	ax2.set_position(pos)
	### Set the label positions to be on the correct side of the colorbar
	ax1.yaxis.set_ticks_position('right')
	ax1.yaxis.set_label_position('right')
	ax2.yaxis.set_ticks_position('left')
	ax2.yaxis.set_label_position('left')

	_adjustplot( lambda kwargs: ax2.set_ylabel(kwargs['logllabel'], fontsize=12*kwargs['font_scale']) , kwargs, ['logllabel', 'font_scale'], 'setting logl label')

	
######################################################################################

def _addslicemarker(ax, markpoints, x, y):

	'''
	This is a helper function to add crosshairs onto the 2d slices.
	
	Parameters
	----------
	ax : matplotlib.pyplot.axis
		The matplotlib axis on to add the labels.
	markpoints : list or np.ndarray
		A list/array of 2 length tuples specifying the coordinates to add crosshairs to. The format of the tuples is (a,b) where a is the x coordinate in data coordinates to mark, b is the y coordinate in data coordinates to mark.
	x : list or np.ndarray
		A list/array of x axis values.
	y : list or np.ndarray
		A list/array of y axis values.
	'''

	### Assuming we have something to plot ###
	if markpoints is not None:
		### For each markpoint ###
		for i in range(len(markpoints)):
			### Add the crosshairs using vertical and horizontal lines with gaps in the middle ###
			sm1 = (markpoints[i][0] - np.min(x)) / (np.max(x) - np.min(x))
			sm2 = (markpoints[i][1] - np.min(y)) / (np.max(y) - np.min(y))
			ax.axvline(markpoints[i][0], ymin=sm2+0.05, ls='--', c='r')
			ax.axvline(markpoints[i][0], ymax=sm2-0.05, ls='--', c='r')
			ax.axhline(markpoints[i][1], xmin=sm1+0.05, ls='--', c='r')
			ax.axhline(markpoints[i][1], xmax=sm1-0.05, ls='--', c='r')

######################################################################################
############################### Input Handelers ######################################
######################################################################################
	
### defaults for some of the kwargs ###
kwarg_defaults = {'cmap':plt.cm.bone,
		   'norm':None,
		   'saveloc':'./', 
		   'savename':None,
		   'figsize': (8,8),
		   'font_scale':1
		   }	
	
######################################################################################
	
def _maininputhandeler(arry, func, scales, kwargs):

	'''
	This takes the main inputs and performs some checks to make sure they are valid so the code doesn't fall over. No checks are done on the keywords as this occours later but the defaults are added if that keyword isn't used.
	
	Parameters
	----------
	cube : np.ndarray or list
		A 3D array to be plotted.
	func : callable
		A function that takes the array and converts the log likelihood values to significance. Is called like this: sig = func(arry)
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y scales. Must be of the format {0:a, 1:b} where a,b are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the array along the axis given by the corresponding key.  
	kwargs : dict
		See one of the main plotting functions (Cube, Slice or CubeSlice) for the full list of kwargs. 
	'''

	### Adding default kwargs if needed ###
	for key in kwarg_defaults.keys():
		if key not in kwargs.keys():
			kwargs[key] = kwarg_defaults[key]

	### Check the arry is 2d and make it a nupmy array if it isn't ###
	if isinstance(arry, (np.ndarray, list))==False or len(arry.shape)!=2: raise ValueError('Need a 2d np.array or list for a 2d plot')
	arry = np.array(arry)
	
	### Check the func is a function ###
	if callable(func)==False: raise ValueError('func needs to be a function')
	arry = np.array(arry)
	
	### Convert scales if None to the correct form with the arry indexes ###
	if scales is None: scales = {0:np.arange(arry.shape[0]), 1:np.arange(arry.shape[1])}
	## Check scales is a dictionary with keys 0,1 and nothing else ###
	if  isinstance(scales, dict)==False or np.any([k not in [0, 1] for k in scales.keys()]) or np.all([k in [0, 1] for k in scales.keys()])==False: raise ValueError('Scales must be a dict with only the keys 0 and 1')
	
	### Shuffle the arry axis and scales dict so they are in order x,y ###
	arry = np.moveaxis(arry, list(scales.keys()), [0, 1])
	scales = dict(sorted(scales.items()))
	
	## Check the values in the scales dict and convert as needed ###
	for k in scales.keys():
		if isinstance(scales[k], (type(None), np.ndarray, list))==False: raise ValueError('Scales must be an np.ndarray, list of None')
		if isinstance(scales[k], type(None)): scales[k] = np.arange(arry.shape[k])
		scales[k] = np.array(scales[k]).flatten()
		if len(scales[k])!=arry.shape[int(k)]: raise ValueError('Scales mush have same length as array along axis '+str(k))
	
	if 'loglticks' not in kwargs.keys(): kwargs['loglticks'] = np.array([round(a,1) for a in np.linspace(np.max(arry), np.min(arry), 5)])
	if 'siglabel' not in kwargs.keys(): kwargs['siglabel'] = 'Significance'
	if 'logllabel' not in kwargs.keys(): kwargs['logllabel'] = 'Log Likelihood'


	return arry, func, scales
	
######################################################################################
	
def _adjustplot(func, kwargs, req, errmesg, noneshallpass=False):

	'''
	This is the main error handeler for the keywords. It doesn't check if the keyword has been given a sensible input but it will tell you where the code fell over and which keywords you need to check.
	
	Parameters
	----------
	func : function
		A function to try and run.
	kwargs : dict
		A dict of keywords to give to func.
	req : list
		A list of keys to check are present in keywords.
	errmesg : str
		A short str indicating what the code was trying to do when it fell over.
	noneshallpass : bool, default=False
		Whether to throw an error if a key in req is not in kwargs. 
	'''

	### Decide whether to throw an error if a keyword is missing or just do nothing ###
	if np.any([r not in kwargs.keys() for r in req]): 
		if noneshallpass==True: raise KeyError('Missing some keyword arguments for '+str(errmesg))
		else: return None
		
	### Try calling the function and if it fails print out the custom error message (along with the full traceback) ###
	try: func(kwargs)
	except: raise ValueError('Something went wrong when calling '+str(errmesg)+', check inputs for keywords '+str(req))

######################################################################################
############################## Public Functions ######################################
######################################################################################

def logl_to_sig(dof, zero_point):

	'''
	This returns a function that converts the log likelihoods to significance. zero_point is the log likelihood of the null model and DOF is the difference in the number of degrees of freedom between the null model and the alternative model.
	
	Parameters
	----------
	dof : int
		The difference in the number of degrees of freedom between the null model and the alternative model.
	zero_point : float
		The log likelihood of the null model.
	'''

	return lambda logl : stats.norm.isf((1 - stats.chi2.cdf(2*(logl-zero_point), dof))/2)

######################################################################################

def SigmaMap(arry, func, scales=None, **kwargs):

	"""
	Plot a 2D log likelihood array with accompanying significances with ease. This function takes the 2D array and does the rest for you, just specify a conversion function for the loglikelihhod to significance! See below or the tutorial in https://github.com/SophiaVaughan/MissiePlots for more information.

	Parameters
	----------
	array: np.ndarray, list 
		The array to be plotted. Must have exactly 2 axis.
	func : callable
		A function that takes the array and converts the log likelihood values to significance. Is called like this: sig = func(arry)
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y scales. Must be of the format {0:a, 1:b} where a,b are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the array along the axis given by the corresponding key. 
	kwargs : dict or keywords
		saveloc : str, default='./' 
			The filepath to the folder in which to save the plot if savename is not None.
		savename : str or None, default=None
			If not None, the plot is saved as a png file with this filename in the folder specified by saveloc.
		figsize : tuple, default=(8,8)
			A length 2 tuple specifying the size of the figure to be created.
		font_scale : int or float, default=1
			A scale factor by which to increase all the fontsizes of the labels in the plot.
		title : str, optional
			The title of the plot. If this keyword isn't used, no title will be added.
		xlabel : str, optional
			The label for the x axis of the plot. If this keyword isn't used, no label will be added.
		ylabel : str, optional
			The label for the y axis of the plot. If this keyword isn't used, no label will be added.
		siglabel : str, optional
			The label for the 'significance' side of the colorbar of the plot. If this keyword isn't used, the default "Significance" label will be added.
		logllabel : str, optional
			The label for the 'Log Likelihood' side of the colorbar of the plot. If this keyword isn't used, the default "Log Likelihood" label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		sigticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		loglticks : list or np.ndarray, optional
			A list/array of ticks in loglikelihhod to add to the log likelihood colorbar. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		siglim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		markpoints : dict
			Used to add crosshairs to points on the 2d plot. Must be a list of length 2 tuples. The tuples are of the form (d,e) where d is the point to mark on the x axis in data coordinates and e is the same for the y axis.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
	"""
	
	### Check main inputs ###
	arry, func, scales = _maininputhandeler(arry, func, scales, kwargs)

	### Create the figure and make the axis 3D ###
	fig = plt.figure(figsize=kwargs['figsize'])
	fig.patch.set_facecolor('white')
	ax = fig.add_subplot(1, 1, 1)

	### Plot the array and add the labels/ticks/lims ###
	sig = func(arry)
	
	### Plot the array making sure to set up the colorscale correclty ###
	if kwargs['norm'] is not None: ax.pcolormesh(scales[0], scales[1], sig.T, cmap=kwargs['cmap'],  norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N))
	else: ax.pcolormesh(scales[0], scales[1], sig.T, vmin=np.nanmin(sig), vmax=np.nanmax(sig),  cmap=kwargs['cmap'],  norm=kwargs['norm'])
	_plotadmin(ax, kwargs)
	plt.tight_layout()
	
	### Make the colorbar ###
	_makecbar(fig, func, arry, kwargs)
	
	### Add crosshair marks ###
	_adjustplot(lambda kwargs: _addslicemarker(ax, kwargs['markpoints'], scales[0], scales[1]), kwargs, ['markpoints'], 'adding slice marker points')
	
	### Add a watermark ###
	fig.suptitle('Plot made with code written by Sophia Vaughan \n https://github.com/SophiaVaughan/MissiePlots', y=0.06, fontsize=10)
	fig.subplots_adjust(bottom=0.15)
	
	### Save or show the figure ###
	if isinstance(kwargs['savename'], str): _adjustplot( lambda kwargs: plt.savefig(kwargs['saveloc']+'/'+kwargs['savename']+'.png'), kwargs, ['saveloc', 'savename'], 'saving plot')
	else: plt.show()
	plt.close()

######################################################################################
################################ Code Ends ###########################################
######################################################################################
