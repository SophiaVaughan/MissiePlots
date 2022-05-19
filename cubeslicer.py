import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

######################################################################################
############################# Private Functions ######################################
######################################################################################

def _plot3dslice(face, ax, x, y, zval, axis, zorder, cmap):

	'''
	This is the code that actually plots the slices in the 3d plot.
	It has been put in a separate function so that it can be extracted if the user so wishes to write their own code.
	
	Parameters
	----------
	face : np.ndarray or list
		A 2D array of the face values for the colormap of the slice.
	ax : matplotlib.pyplot.axis
		The matplotlib axis on which to plot the slice.
	x : list, np.ndarray
		The values of the slice along the axis of the lowest axis of the cube that is not the slice axis.
	y: list, np.ndarray
		The values of the slice along the axis of the other axis of the cube that is not the slice axis.
	zval : int
		The cube index of the slice.
	axis : int 
		The cube axis along which the slice is taken. Must be one of 0,1,2.
	zorder : int or float
		A number specifying the order to plot the slices.
	'''

	### Creates the coordinates of each point of the face in the two non-constant directions ###
	xx, yy = np.meshgrid(x,y, indexing='ij')
	### Adds in the constant coordinates on the correct axis ###
	args = [xx, yy]; args.insert(axis, zval*np.ones(xx.shape))
	### Plots the surface with the facecolors defined by the values of the array ###
	ax.plot_surface(*tuple(args), facecolors = cmap.to_rgba(face), zorder=zorder)

######################################################################################

def _cube3D(cube, ax, slices, scales, kwargs):

	'''
	This splits up the slices into sub slices along the lines where other slices intersect the slice (so that the slices appear to pass through each other).
	It then plots these slices in 3D using _plot3dslice
	
	Parameters
	----------
	cube : np.ndarray or list
		A 3D array cube to plot slices of.
	ax : matplotlib.pyplot.axis
		The matplotlib axis on which to plot the slice.
	slices : dict
		The slices to be plotted. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, an integer, a list of integers or a numpy array of integers. The integers specify the index of the slice to be plotted along the axis specified by the key. The order in the dict specifies which axis of the cube maps to which axis of the plot, the first maps the the x axis, the second to y and so on.
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y,z scales (this does not alter how slices works). Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the cube along the axis given by the corresponding key. 
	kwargs : dict
		This function only uses the following kwargs.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
		zorder_function : function, default= an internal function
			DO NOT USE UNLESS NEEDED. If the slices appear ontop of one another when they shouldn't then use the function to map the center x,y,z value of the slice to a number. The silces will then be plotted in ascending order of that number. 
	'''

	### Create a list of the numbers defining the plotting order of the subslices ###
	zorderlist = []
	
	### Set up the cmap properly if norm is set or not ###
	if kwargs['norm'] is not None: cmap = plt.cm.ScalarMappable(norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N), cmap=kwargs['cmap'])
	else: cmap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(cube), vmax=np.max(cube)), cmap=kwargs['cmap'])
	cmap.set_array([])
    
	### For each axis ##
	for i in range(3):
		### Get the remaining axes to look for intersecting slices ###
		remidx = [a for a in np.arange(3) if a!=i]
        
		for slc in slices[i]:
			### For each slice along axis i, get the cube along that slice ###
			fullface = np.take(cube, slc, axis=int(i))
			### And find the intersections ###
			xedge = [0] + [int(a) for a in slices[remidx[0]]] + [len(scales[remidx[0]])-1]
			yedge = [0] + [int(a) for a in slices[remidx[1]]] + [len(scales[remidx[1]])-1]

			### Use booleans to split the slice into sections where other slices intersect ###
			for j in range(len(xedge)-1):
				xbool = (scales[remidx[0]] >= scales[remidx[0]][xedge[j]]) & (scales[remidx[0]] <= scales[remidx[0]][xedge[j+1]])
				for k in range(len(yedge)-1):
					ybool = (scales[remidx[1]] >= scales[remidx[1]][yedge[k]]) & (scales[remidx[1]] <= scales[remidx[1]][yedge[k+1]])
                    
                    			### Find the x,y and face of each sub-section ###
					x = scales[remidx[0]][xbool]
					y = scales[remidx[1]][ybool]
					face = fullface[np.outer(xbool, ybool)].reshape(len(x), len(y))
                    
                    			### Find the number for the plotting order (zorder) ###
					zorder = kwargs['zorder_function'](np.average(x),np.average(y),scales[i][slc])
					### Check no preivious slice has that number, if there is keep adding 1 until a unique number is obtained ###
					while zorder in zorderlist: zorder += 1
					zorderlist += [zorder]
                    			
                    			### Add the subsection to the plot ###
					_plot3dslice(face, ax, x, y, scales[i][slc], int(i), zorder, cmap)

######################################################################################
					
def _plotadmin(ax, kwargs, d3=False):

	'''
	This is a helper function to add a title and axis labels to a plot or subplot as well as alter the axis ticks and lims. 
	CAUTION: Altering the limits of a 3D plot will not clip the slices and so slices may appear to spill beyond the edges of the plot. Please clip the cube instead.
	
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
		zlabel : str, optional
			The label for the z axis of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		zticks : list or np.ndarray, optional
			A list/array of ticks to include on the z axis of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		zlim : list, optional
			A length 2 list containing the [min, max] range for the z axis of the plot. If this keyword isn't used, the default range will be used.
	d3 : bool, default=False
		Whether or not to include the z axis when adding labels, ticks and lims.
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
	
	### If the plot is 3D, add a label/change the axis ticks/set the limits of the z axis if the and 'zlabel' and 'font_scale'/'zticks' and 'font_scale'/'zlim' keywords are set ###
	if d3==True:
		_adjustplot( lambda kwargs: ax.set_zlabel(kwargs['zlabel'], fontsize=12*kwargs['font_scale']) , kwargs, ['zlabel', 'font_scale'], 'setting z label')
		_adjustplot( lambda kwargs: ax.set_zticks(kwargs['zticks'], fontsize=12*kwargs['font_scale']) , kwargs, ['zticks', 'font_scale'], 'setting z ticks')
		_adjustplot( lambda kwargs: ax.set_zlim(kwargs['zlim']) , kwargs, ['zlim', 'font_scale'], 'setting z lim')

######################################################################################

def _maketempkwarg(kwargs, ax1, ax2, i):

	'''
	This is a helper function to convert the input kwargs for a 3D plot to those for the 2D slice plots. 
	
	Parameters
	----------
	kwargs : dict
		This function only uses the following kwargs.
		font_scale : int or float, default=1
			A scale factor by which to increase all the fontsizes of the labels in the plot.
		title : str or list, optional
			The title of the plot. If this keyword isn't used, no title will be added.
		xlabel : str, optional
			The label for the x axis of the plot. If this keyword isn't used, no label will be added.
		ylabel : str, optional
			The label for the y axis of the plot. If this keyword isn't used, no label will be added.
		zlabel : str, optional
			The label for the z axis of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		zticks : list or np.ndarray, optional
			A list/array of ticks to include on the z axis of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		zlim : list, optional
			A length 2 list containing the [min, max] range for the z axis of the plot. If this keyword isn't used, the default range will be used.
	ax1 : int
		The index of the 3D cube to be mapped to the x axis of the 2D slice.
	ax2 : int
		The index of the 3D cube to be mapped to the y axis of the 2D slice.
	i : int
		The number of the plot so that if title is a list of titles (one for each plot) then the correct on is added.
	'''

	### Create a temporary list of kwargs ###
	tempkwargs = {}
	### To map the axis number to the labels ###
	lab = ['x','y','z']

	### Add the font_scale to the temporary keywords ###
	tempkwargs['font_scale'] = kwargs['font_scale']

	### Add the correct title if it is a list ###
	if 'title' in kwargs.keys() and isinstance(kwargs['title'], (np.ndarray, list)): tempkwargs['title'] = kwargs['title'][i]

	### Map ax1 keywords to the x axis keywords in the temporary dict ###
	if lab[ax1]+'label' in kwargs.keys(): tempkwargs['xlabel'] = kwargs[lab[ax1]+'label']
	if lab[ax1]+'ticks' in kwargs.keys(): tempkwargs['xticks'] = kwargs[lab[ax1]+'ticks']
	if lab[ax1]+'lim' in kwargs.keys(): tempkwargs['xlim'] = kwargs[lab[ax1]+'lim']
	
	### Map ax2 keywords to the y axis keywords in the temporary dict ###
	if lab[ax2]+'label' in kwargs.keys(): tempkwargs['ylabel'] = kwargs[lab[ax2]+'label']
	if lab[ax2]+'ticks' in kwargs.keys(): tempkwargs['yticks'] = kwargs[lab[ax2]+'ticks']
	if lab[ax2]+'lim' in kwargs.keys(): tempkwargs['ylim'] = kwargs[lab[ax2]+'lim']
	
	return tempkwargs

######################################################################################
	
def _makecbar(fig, cube, kwargs):

	'''
	This is a helper function to add a colorbar to the side of the image.
	
	Parameters
	----------
	fig : matplotlib.pyplot.figure
		The figure to add the colorbar to.
	cube : np.ndarray or list
		A 3D array cube to plot slices of.
	kwargs : dict
		This function only uses the following kwargs.
		font_scale : int or float, default=1
			A scale factor by which to increase all the fontsizes of the labels in the plot.
		cbarlabel : str, optional
			The label for the colorbar of the plot. If this keyword isn't used, no label will be added.
		cbarticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		cbarlim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		cbar_marks : list or np.ndarray, optional
			A list/array of points along the colorbar to add marks to. If this keyword isn't used, no marks will be added.
		cbar_mark_labels : list or np.ndarray, optional
			A list/array of strings to label the corresponding marks specified with cbar_marks. If this keyword isn't used, no marks will be added.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
	'''
	
	### Move plot to side to make room for colorbar ###
	fig.subplots_adjust(right=0.8)
	### Make a new axis for the colorbar ###
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	
	### Set up the cmap properly if norm is set or not ###
	if kwargs['norm'] is not None: cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N), cmap=kwargs['cmap']), cax=cbar_ax)
	else: cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(cube), vmax=np.max(cube)), cmap=kwargs['cmap']), cax=cbar_ax)

	### Add a label/change the axis ticks/set the limits of the colorbar if the and 'cbarlabel' and 'font_scale'/'cbarticks' and 'font_scale'/'cbarlim' keywords are set ###
	_adjustplot( lambda kwargs: cbar.set_label(kwargs['cbarlabel'], fontsize=12*kwargs['font_scale']) , kwargs, ['cbarlabel', 'font_scale'], 'setting cbar label')
	_adjustplot( lambda kwargs: cbar.set_ticks(kwargs['cbarticks'], fontsize=12*kwargs['font_scale']) , kwargs, ['cbarticks', 'font_scale'], 'setting cbar ticks')
	_adjustplot( lambda kwargs: cbar.set_lim(kwargs['cbarlim'], fontsize=12*kwargs['font_scale']) , kwargs, ['cbarlim', 'font_scale'], 'setting cbar lim')
	
	### Mark points on the colorbar by adding axis ticks to the right hand side of the colorbar and labelling them ###
	_adjustplot( lambda kwargs: _addcbarmarker(cbar, kwargs['cbar_marks'], kwargs['cbar_mark_labels'], kwargs['font_scale']) , kwargs, ['cbar_marks', 'cbar_mark_labels', 'font_scale'], 'setting cbar marks')

######################################################################################

def _addcbarmarker(cbar, markpoints, marklabels, fontscale):

	'''
	This is a helper function to add marks on the right side of the colorbar.
	
	Parameters
	----------
	cbar : matplotlib.pyplot.figure.colorbar
		The colorbar to add the marks on.
	markpoints : list or np.ndarray
		A list/array of points along the colorbar to add marks to.
	marklabels : list or np.ndarray, optional
		A list/array of strings to label the corresponding marks specified with markpoints.
	font_scale : int or float, default=1
		A scale factor by which to increase all the fontsizes of the labels in the plot.
	'''

	### Get the position of the colorbar and it's axis object ###
	pos = cbar.ax.get_position()
	ax1 = cbar.ax
	ax1.set_aspect('auto')

	### Dulicate the axis ###
	ax2 = ax1.twinx()
	### Match the limits of the new axis to the old one ###
	ax2.set_ylim(ax1.get_ylim())
	### Add the mark points as axis ticks ###
	ax2.set_yticks(markpoints)
	### Add the labels for the marks as tick labels ###
	ax2.set_yticklabels(marklabels, color='r', fontsize=12*fontscale)
	### set the positions of the two axis to match the colorbar
	ax1.set_position(pos)
	ax2.set_position(pos)
	### Set the label positions to be on the correct side of the colorbar
	ax1.yaxis.set_ticks_position('right')
	ax1.yaxis.set_label_position('right')
	ax2.yaxis.set_ticks_position('left')
	ax2.yaxis.set_label_position('left')
	
######################################################################################

def _addslicemarker(ax, markpoints, x, y, slcidx):

	'''
	This is a helper function to add crosshairs onto the 2d slices.
	
	Parameters
	----------
	ax : matplotlib.pyplot.axis
		The matplotlib axis on to add the labels.
	markpoints : list or np.ndarray
		A list/array of 2/3 length tuples specifying the coordinates to add crosshairs to. The format of the tuples is (c,a,b) or (a,b) where a is the x coordinate in data coordinates to mark, b is the y coordinate in data coordinates to mark and c is the index of the slice to mark it on. If c is not set, it is marked on all slices.
	x : list or np.ndarray
		A list/array of x axis values.
	x : list or np.ndarray
		A list/array of y axis values.
	slcidx : int
		The index of the slice being plotted.
	'''

	### Assuming we have something to plot ###
	if markpoints is not None:
		### For each markpoint ###
		for i in range(len(markpoints)):
			### If it is a 3 length mark point, check the slice too ###
			if (len(markpoints[i])==3 and slcidx==markpoints[i][0]): 
				### Add the crosshairs using vertical and horizontal lines with gaps in the middle ###
				sm1 = (markpoints[i][1] - np.min(x)) / (np.max(x) - np.min(x))
				sm2 = (markpoints[i][2] - np.min(y)) / (np.max(y) - np.min(y))
				ax.axvline(markpoints[i][1], ymin=sm2+0.05, ls='--', c='r')
				ax.axvline(markpoints[i][1], ymax=sm2-0.05, ls='--', c='r')
				ax.axhline(markpoints[i][2], xmin=sm1+0.05, ls='--', c='r')
				ax.axhline(markpoints[i][2], xmax=sm1-0.05, ls='--', c='r')
			## If it a length 2 mark point ###
			elif len(markpoints[i])==2: 
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
		   'zorder_function':lambda x, y, z: x*abs(x) + y*abs(y) + z*abs(z) ,
		   'saveloc':'./', 
		   'savename':None,
		   'figsize': (8,8),
		   'font_scale':1
		   }	
	
######################################################################################
	
def _maininputhandeler(cube, slices, scales, kwargs):

	'''
	This takes the main inputs and performs some checks to make sure they are valid so the code doesn't fall over. No checks are done on the keywords as this occours later but the defaults are added if that keyword isn't used.
	
	Parameters
	----------
	cube : np.ndarray or list
		A 3D array to be plotted.
	slices : dict
		The slices to be plotted. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, an integer, a list of integers or a numpy array of integers. The integers specify the index of the slice to be plotted along the axis specified by the key. The order in the dict specifies which axis of the cube maps to which axis of the plot, the first maps the the x axis, the second to y and so on.
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y,z scales (this does not alter how slices works). Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the cube along the axis given by the corresponding key. 
	kwargs : dict
		See one of the main plotting functions (Cube, Slice or CubeSlice) for the full list of kwargs. 
	'''

	### Adding default kwargs if needed ###
	for key in kwarg_defaults.keys():
		if key not in kwargs.keys():
			kwargs[key] = kwarg_defaults[key]

	### Check the cube is 3d and make it a nupmy array if it isn't ###
	if isinstance(cube, (np.ndarray, list))==False or len(cube.shape)!=3: raise ValueError('Need a 3d np.array or list for a 3d plot')
	cube = np.array(cube)

	### Check slices is a dictionary with keys 0,1,2 and nothing else ###
	if isinstance(slices, dict)==False or np.any([k not in [0, 1, 2] for k in slices.keys()]) or np.all([k in [0, 1, 2] for k in slices.keys()])==False: raise ValueError('Slices must be a dict with only the keys 0, 1 and 2')
	### Shuffle the cube axis and slices dict so they are in order x,y,z ###
	cube = np.moveaxis(cube, list(slices.keys()), [0, 1, 2])
	slices = dict(sorted(slices.items()))
	
	### Check the values in the slices dict and convert as needed ###
	for k in slices.keys():
		if isinstance(slices[k], (int, type(None), np.ndarray, list))==False: raise ValueError('Slices must be an np.ndarray, list, int or None')
		if isinstance(slices[k], type(None)): slices[k] = np.array([], dtype=int)
		elif isinstance(slices[k], int): slices[k] = np.array([slices[k]], dtype=int)
		else: slices[k] = np.array(slices[k], dtype=int)
		
	### Convert scales if None to the correct form with the cube indexes ###
	if scales is None: scales = {0:np.arange(cube.shape[0]), 1:np.arange(cube.shape[1]), 2:np.arange(cube.shape[2])}
	## Check scales is a dictionary with keys 0,1,2 and nothing else ###
	if  isinstance(scales, dict)==False or np.any([k not in [0, 1, 2] for k in scales.keys()]) or np.all([k in [0, 1, 2] for k in scales.keys()])==False: raise ValueError('Scales must be a dict with only the keys 0, 1 and 2')
	scales = dict(sorted(scales.items()))
	
	## Check the values in the scales dict and convert as needed ###
	for k in scales.keys():
		if isinstance(scales[k], (type(None), np.ndarray, list))==False: raise ValueError('Scales must be an np.ndarray, list of None')
		if isinstance(scales[k], type(None)): scales[k] = np.arange(cube.shape[k])
		scales[k] = np.array(scales[k]).flatten()
		if len(scales[k])!=cube.shape[int(k)]: raise ValueError('Scales mush have same length as cube along axis '+str(k))
		
	return cube, slices, scales
	
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


def Cube(cube, slices, scales=None, **kwargs):

	"""
	Plot the slices of a cube in 3D with ease. This function takes the 3D cube and does the rest for you, just specify the slices you want! See below or the tutorial in https://github.com/SophiaVaughan/MissiePlots for more information.

	Parameters
	----------
	cube : np.ndarray, list 
		The cube to be plotted. Must have exactly 3 axis.
	slices : dict
		The slices to be plotted. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, an integer, a list of integers or a numpy array of integers. The integers specify the index of the slice to be plotted along the axis specified by the key. The order in the dict specifies which axis of the cube maps to which axis of the plot, the first maps the the x axis, the second to y and so on.
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y,z scales (this does not alter how slices works). Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the cube along the axis given by the corresponding key. 
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
		zlabel : str, optional
			The label for the z axis of the plot. If this keyword isn't used, no label will be added.
		cbarlabel : str, optional
			The label for the colorbar of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		zticks : list or np.ndarray, optional
			A list/array of ticks to include on the z axis of the plot. If this keyword isn't used, the default ticks will be used.
		cbarticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		zlim : list, optional
			A length 2 list containing the [min, max] range for the z axis of the plot. If this keyword isn't used, the default range will be used.
		cbarlim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		cbar_marks : list or np.ndarray, optional
			A list/array of points along the colorbar to add marks to. If this keyword isn't used, no marks will be added.
		cbar_mark_labels : list or np.ndarray, optional
			A list/array of strings to label the corresponding marks specified with cbar_marks. If this keyword isn't used, no marks will be added.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
		zorder_function : function, default= an internal function
			DO NOT USE UNLESS NEEDED. If the slices appear ontop of one another when they shouldn't then use the function to map the center x,y,z value of the slice to a number. The silces will then be plotted in ascending order of that number. 
	"""
	
	### Check main inputs ###
	cube, slices, scales = _maininputhandeler(cube, slices, scales, kwargs)

	### Create the figure and make the axis 3D ###
	fig = plt.figure(figsize=kwargs['figsize'])
	fig.patch.set_facecolor('white')
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	### Plot the slices and add the labels/ticks/lims ###
	_cube3D(cube, ax, slices, scales, kwargs)
	_plotadmin(ax, kwargs, d3=True)
	plt.tight_layout()
	
	### Make the colorbar ###
	_makecbar(fig, cube, kwargs)
	
	### Add a watermark ###
	fig.suptitle('Plot made with code written by Sophia Vaughan \n https://github.com/SophiaVaughan/MissiePlots', y=0.04, fontsize=10)
	fig.subplots_adjust(bottom=0.1)
	
	### Save or show the figure ###
	if isinstance(kwargs['savename'], str): _adjustplot( lambda kwargs: plt.savefig(kwargs['saveloc']+'/'+kwargs['savename']+'.png'), kwargs, ['saveloc', 'savename'], 'saving plot')
	else: plt.show()
	plt.close()
	
######################################################################################

def Slice(cube, slices, scales=None, **kwargs):

	"""
	Plot a range of 2D slices of a cube with ease. This function takes the 3D cube and does the rest for you, just specify the slices you want! See below or the tutorial in https://github.com/SophiaVaughan/MissiePlots for more information.

	Parameters
	----------
	cube : np.ndarray, list 
		The cube to be plotted. Must have exactly 3 axis.
	slices : dict
		The slices to be plotted. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, an integer, a list of integers or a numpy array of integers. The integers specify the index of the slice to be plotted along the axis specified by the key. The order in the dict specifies which axis of the cube maps to which axis of the plot, the first maps the the x axis, the second to y and so on.
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y,z scales (this does not alter how slices works). Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the cube along the axis given by the corresponding key. 
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
		zlabel : str, optional
			The label for the z axis of the plot. If this keyword isn't used, no label will be added.
		cbarlabel : str, optional
			The label for the colorbar of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		zticks : list or np.ndarray, optional
			A list/array of ticks to include on the z axis of the plot. If this keyword isn't used, the default ticks will be used.
		cbarticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		zlim : list, optional
			A length 2 list containing the [min, max] range for the z axis of the plot. If this keyword isn't used, the default range will be used.
		cbarlim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		cbar_marks : list or np.ndarray, optional
			A list/array of points along the colorbar to add marks to. If this keyword isn't used, no marks will be added.
		cbar_mark_labels : list or np.ndarray, optional
			A list/array of strings to label the corresponding marks specified with cbar_marks. If this keyword isn't used, no marks will be added.
		markpoints : dict
			Used to add crosshairs to points on the 2d slices. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None or a list of 2/3 length tuples. The tuples are of the form (f,d,e) or (d,e) where d is the point to mark on the x axis in data coordinates and e is the same for the y axis. If f is used it is the index of the slice to plot the crosshairs on. The x axis maps to the lowest axis that is not the slices axis and the y aixs is the other axis that is not the slice axis.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
		zorder_function : function, default= an internal function
			DO NOT USE UNLESS NEEDED. If the slices appear ontop of one another when they shouldn't then use the function to map the center x,y,z value of the slice to a number. The silces will then be plotted in ascending order of that number. 
	"""

	### Check main inputs ###
	cube, slices, scales = _maininputhandeler(cube, slices, scales, kwargs)
	
	### Count how many slices there are to plot and set the grid size ###
	nslice = np.sum([len(slices[s]) for s in slices.keys()])
	if nslice<=3: n1=1; n2=nslice
	elif nslice==4: n1=2; n2=2
	else: 
		n1 = int(nslice/3)
		if nslice%3!=0: n1+=1 
		n2 = 3

	### Create the figure and start a plot index counter ###
	fig = plt.figure(figsize=kwargs['figsize'])
	fig.patch.set_facecolor('white')
	pltidx = 1
	
	### For each axis ###
	idx = [key for key in slices.keys()]
	for i in idx:
		### Get the remaining axes for axis scales ###
		remidx = [a for a in idx if a!=i]
		for slc in slices[i]:
			### For each slice along axis i, get the cube along that slice ###
			fullface = np.take(cube, slc, axis=int(i))
	
			### Create an axis to plot it on ###
			ax = fig.add_subplot(n1, n2, pltidx)
			
			### Plot the slice making sure to set up the colorscale correclty ###
			if kwargs['norm'] is not None: ax.pcolormesh(scales[remidx[0]], scales[remidx[1]], fullface.T, cmap=kwargs['cmap'],  norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N))
			else: ax.pcolormesh(scales[remidx[0]], scales[remidx[1]], fullface.T, vmin=np.nanmin(cube), vmax=np.nanmax(cube), cmap=kwargs['cmap'],  norm=kwargs['norm'])
			
			### Make the temp keywords for the axis labels/ticks/lim and title and add them to the axis ###
			tempkwargs = _maketempkwarg(kwargs, remidx[0], remidx[1], pltidx-1)
			_plotadmin(ax, tempkwargs)
			
			### Add slice marks ###
			_adjustplot(lambda kwargs: _addslicemarker(ax, kwargs['markpoints'][i], scales[remidx[0]], scales[remidx[1]], slc), kwargs, ['markpoints'], 'adding slice marker points')
			
			### Make the plots square and iterate the plot marker ###
			ax.set_aspect((np.max(scales[remidx[0]] - np.min(scales[remidx[0]])))/(np.max(scales[remidx[1]] - np.min(scales[remidx[1]]))))
			pltidx += 1
	
	### Add main title if only one is supplied
	if 'title' in kwargs.keys() and isinstance(kwargs['title'], str): _adjustplot( lambda kwargs: fig.suptitle(kwargs['title'], fontsize=16*kwargs['font_scale']), kwargs, ['title', 'font_scale'], 'adding title')
	plt.tight_layout()
	
	### Add colorbar ###
	_makecbar(fig, cube, kwargs)
	
	### Add watermark ###
	fig.suptitle('Plot made with code written by Sophia Vaughan \n https://github.com/SophiaVaughan/MissiePlots', y=0.04, fontsize=10)
	fig.subplots_adjust(bottom=0.1)	
	
	### Save or show the figure ###
	if isinstance(kwargs['savename'], str): _adjustplot( lambda kwargs: plt.savefig(kwargs['saveloc']+'/'+kwargs['savename']+'.png'), kwargs, ['saveloc', 'savename'], 'saving plot')
	else: plt.show()
	plt.close()	
	
######################################################################################

def CubeSlice(cube, slices, scales=None, **kwargs):

	"""
	Want to plot the slices of a cube in 3D and display those slices in 2D on the same plot? Look no further, this function takes the 3D cube and does the rest for you, just specify the slices you want! See below or the tutorial in https://github.com/SophiaVaughan/MissiePlots for more information.

	Parameters
	----------
	cube : np.ndarray, list 
		The cube to be plotted. Must have exactly 3 axis.
	slices : dict
		The slices to be plotted. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, an integer, a list of integers or a numpy array of integers. The integers specify the index of the slice to be plotted along the axis specified by the key. The order in the dict specifies which axis of the cube maps to which axis of the plot, the first maps the the x axis, the second to y and so on.
	scales : dict or None, default=None 
		Maps the index of the cube to the x,y,z scales (this does not alter how slices works). Must be of the format {0:a, 1:b, 2:c} where a,b,c are None, a list or a numpy array. If None, the index is kept as the scale, if a list or array are supplied then they must be the same length as the cube along the axis given by the corresponding key. 
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
		zlabel : str, optional
			The label for the z axis of the plot. If this keyword isn't used, no label will be added.
		cbarlabel : str, optional
			The label for the colorbar of the plot. If this keyword isn't used, no label will be added.
		xticks : list or np.ndarray, optional
			A list/array of ticks to include on the x axis of the plot. If this keyword isn't used, the default ticks will be used.
		yticks : list or np.ndarray, optional
			A list/array of ticks to include on the y axis of the plot. If this keyword isn't used, the default ticks will be used.
		zticks : list or np.ndarray, optional
			A list/array of ticks to include on the z axis of the plot. If this keyword isn't used, the default ticks will be used.
		cbarticks : list or np.ndarray, optional
			A list/array of ticks to include on the colorbar of the plot. If this keyword isn't used, the default ticks will be used.
		xlim : list, optional
			A length 2 list containing the [min, max] range for the x axis of the plot. If this keyword isn't used, the default range will be used.
		ylim : list, optional
			A length 2 list containing the [min, max] range for the y axis of the plot. If this keyword isn't used, the default range will be used.
		zlim : list, optional
			A length 2 list containing the [min, max] range for the z axis of the plot. If this keyword isn't used, the default range will be used.
		cbarlim : list, optional
			A length 2 list containing the [min, max] range for the colorbar of the plot. If this keyword isn't used, the default range will be used.
		cbar_marks : list or np.ndarray, optional
			A list/array of points along the colorbar to add marks to. If this keyword isn't used, no marks will be added.
		cbar_mark_labels : list or np.ndarray, optional
			A list/array of strings to label the corresponding marks specified with cbar_marks. If this keyword isn't used, no marks will be added.
		markpoints : dict
			Used to add crosshairs to points on the 2d slices. Must be of the format {0:a, 1:b, 2:c} where a,b,c are None or a list of 2/3 length tuples. The tuples are of the form (f,d,e) or (d,e) where d is the point to mark on the x axis in data coordinates and e is the same for the y axis. If f is used it is the index of the slice to plot the crosshairs on. The x axis maps to the lowest axis that is not the slices axis and the y aixs is the other axis that is not the slice axis.
		cmap : matplotlib.pyplot.cm.<colormap>, default=matplotlib.pyplot.cm.bone
			The colormap to use for the plot.
		norm : list, np.ndarray or None, default=None
			If not None then this specifies the boundaries along which to discretize the colormap.
		zorder_function : function, default= an internal function
			DO NOT USE UNLESS NEEDED. If the slices appear ontop of one another when they shouldn't then use the function to map the center x,y,z value of the slice to a number. The silces will then be plotted in ascending order of that number. 
	"""

	### Check main inputs ###
	cube, slices, scales = _maininputhandeler(cube, slices, scales, kwargs)
	
	### Count how many slices there are to plot and set the grid size ###
	nslice = np.sum([len(slices[s]) for s in slices.keys()])
	if nslice<=3: n1=1; n2=nslice
	elif nslice==4: n1=2; n2=2
	else: 
		n1 = int(nslice/3)
		if nslice%3!=0: n1+=1 
		n2 = 3
		
	### Create the figure and start a plot index counter ###
	fig = plt.figure(figsize=kwargs['figsize'])
	fig.patch.set_facecolor('white')

	pltidx = n2*n2 + 1
	
	### Make the 3D plot at the top ###
	ax = fig.add_subplot(n1+n2, n2, (1,n2*n2), projection='3d')
	_cube3D(cube, ax, slices, scales, kwargs)
	
	### Add lables/ticks/lims and title to the 3D plot ###
	tempkwargs = kwargs.copy()
	if 'title' in tempkwargs.keys() and isinstance(tempkwargs['title'], (np.ndarray, list)): tempkwargs['title'] = tempkwargs['title'][0]
	_plotadmin(ax, tempkwargs, d3=True)
	
	### For each axis ###
	idx = [key for key in slices.keys()]
	for i in idx:
		### Get the remaining axes for axis scales ###
		remidx = [a for a in idx if a!=i]
		for slc in slices[i]:
			### For each slice along axis i, get the cube along that slice ###
			fullface = np.take(cube, slc, axis=int(i))
	
			### Create an axis to plot it on ###
			ax = fig.add_subplot(n1+n2, n2, pltidx)
			
			### Plot the slice making sure to set up the colorscale correclty ###
			if kwargs['norm'] is not None: ax.pcolormesh(scales[remidx[0]], scales[remidx[1]], fullface.T, cmap=kwargs['cmap'],  norm=colors.BoundaryNorm(kwargs['norm'], kwargs['cmap'].N))
			else: ax.pcolormesh(scales[remidx[0]], scales[remidx[1]], fullface.T, vmin=np.nanmin(cube), vmax=np.nanmax(cube), cmap=kwargs['cmap'],  norm=kwargs['norm'])
			
			### Make the temp keywords for the axis labels/ticks/lim and title and add them to the axis ###
			tempkwargs = _maketempkwarg(kwargs, remidx[0], remidx[1], pltidx-n2*n2)
			_plotadmin(ax, tempkwargs)
			
			### Add slice marks ###
			_adjustplot(lambda kwargs: _addslicemarker(ax, kwargs['markpoints'][i], scales[remidx[0]], scales[remidx[1]], slc), kwargs, ['markpoints'], 'adding slice marker points')
			
			### Make the plots square and iterate the plot marker ###
			ax.set_aspect((np.max(scales[remidx[0]] - np.min(scales[remidx[0]])))/(np.max(scales[remidx[1]] - np.min(scales[remidx[1]]))))
			pltidx += 1
	
	### Add main title if only one is supplied
	if 'title' in kwargs.keys() and isinstance(kwargs['title'], str): _adjustplot( lambda kwargs: fig.suptitle(kwargs['title'], fontsize=16*kwargs['font_scale']), kwargs, ['title', 'font_scale'], 'adding title')
	plt.tight_layout()
	
	### Add colorbar ###
	_makecbar(fig, cube, kwargs)
	
	### Add watermark ###
	fig.suptitle('Plot made with code written by Sophia Vaughan \n https://github.com/SophiaVaughan/MissiePlots', y=0.04, fontsize=10)
	fig.subplots_adjust(bottom=0.1)

	### Save or show the figure ###
	if isinstance(kwargs['savename'], str): _adjustplot( lambda kwargs: plt.savefig(kwargs['saveloc']+'/'+kwargs['savename']+'.png'), kwargs, ['saveloc', 'savename'], 'saving plot')
	else: plt.show()
	plt.close()

######################################################################################
################################ Code Ends ###########################################
######################################################################################
