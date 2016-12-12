from img2vis import img2vis
import glob
import numpy as np
import os
from astropy.io import fits

import pdb

##
## allmodels.py
##
## PURPOSE
##    wrapper function for img2vis; parses all FITS file headers in given 
##    directory and calls img2vis for 12 micron images of Circinus/NGC1068
##

modeldir = "../models/Bernd_2016-12-12/"
models = glob.glob(modeldir+"*.fits")

scalefactors = [1.]
#scalefactors=[0.75,1.,1.25] ## give array of multiple values if you want to test different scalefactors

for m in models:
	hdr=fits.getheader(m)
	
	obj = hdr["NAMEBV"].strip() ## not sure if the strip() is necessary, but it doesn't hurt either
	lam = np.float(hdr["LAMBDABV"])
	inc = np.float(hdr["ANGLEBV"]) ## not used at the moment
	L_scale = np.float(hdr["LBV"])
	
	##
	## convert lam to wavelength in meters
	#lam=1.e-6*lam
	##
	## wavelength currently fixed to 12 micron (highest quality of visibilities)
	lam=12.e-6
	oifits=False
	phot=False
	
	if obj=="n1068":
		mas_per_pc = 1000/71
		delta_pa=25
		##
		## PA of disk in NGC 1068: -45 deg = 135 deg (Lopez Gonzaga+ 2014)
		pa_init=110
		pa_max=160
		if np.isclose(lam,1.2e-5):
			oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits" ## contains correlated fluxes in Jansky
#			phot = 10.0 ## 10 micron flux in Jansky from Raban+ 2009
			phot = 16.5 ## 12 micron flux in Jansky from Raban+ 2009
	elif obj=="circinus":
		##
		## PA of disk in Circinus: 44 deg (Tristram+ 2014)
		pa_init=20
		pa_max=70
		mas_per_pc = 1000/20
		delta_pa=15
		if np.isclose(lam,1.2e-5):
			oifits = "../MIDI_data/Circinus_clean.oifits"
	else:
		raise ValueError("Object {0} not known".format(obj))
	
	pxscale_pc = 0.04 * np.sqrt(L_scale)
	pxscale_mas = pxscale_pc * mas_per_pc

	for scale in scalefactors:
		this_pxscale_mas=pxscale_mas*scale
		i=img2vis(m, this_pxscale_mas, lam, oifits=oifits, phot=phot, delta_pa=delta_pa)
		if scale != 1.0:
			i.f_plot=i.f_plot.split(".png")[0]+"_"+"{0:.2f}".format(scale)+".png"
			
		i.f_plot="/Users/leo/Desktop/circinus_bild_10_25L0.0370370.png"

		if oifits:
			i.optimize_pa(step=2,pa_init=pa_init,pa_max=pa_max)
		i.make_plot()