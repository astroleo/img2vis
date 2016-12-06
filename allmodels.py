from img2vis import img2vis
import glob
import numpy as np
import pdb
import os

import pdb

models=glob.glob("../models/Bernd_2016-10-31/circinus_bild_10_*.fits")
#models+=glob.glob("../models/Bernd_2016-10-12/n1068_bild_12102016/n1068_bild_10_*.fits")

phot_1068_10mu = 10.0

#scales=np.array([0.50,0.75,1.00,1.25,1.50])
scales=np.array([0.70,1.00])
#scales=np.array([0.50])

for m in models:
#	L_scale = np.float(m.split("spiral.fits")[0].split("L")[1])
	#L_scale = np.float(m.split(".fits")[0].split("L")[1])
	L_scale=np.float(m.split(".fits")[0].split("L")[1].split("size")[0])
#	obj = m.split("/")[-1].split("_")[0]
	obj="circinus"
#	lam = np.round(np.float(m.split("/")[-1].split("_")[2])) * 1e-6
	lam=1e-5
	oifits=False
	phot=False
	
	if obj=="n1068":
		mas_per_pc = 1000/71
		delta_pa=25
		##
		## PA of disk in NGC 1068: -45 deg = 135 deg (Lopez Gonzaga+ 2014)
		pa_init=110
		pa_max=160
		if np.isclose(lam,1e-5):
			## run 1068 model with Circinus data
			#oifits = "../MIDI_data/Circinus_clean.oifits"
			oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
			phot = phot_1068_10mu
	elif obj=="circinus":
		##
		## PA of disk in Circinus: 44 deg (Tristram+ 2014)
		pa_init=20
		pa_max=70
		mas_per_pc = 1000/20
		delta_pa=15
		if np.isclose(lam,1e-5):
			oifits = "../MIDI_data/Circinus_clean.oifits"
	else:
		raise ValueError("Object {0} not known".format(obj))
	
	pxscale_pc = 0.04 * np.sqrt(L_scale)
	pxscale_mas = pxscale_pc * mas_per_pc

	for scale in scales:
		this_pxscale_mas=pxscale_mas*scale
		lam=1.2e-5
		i=img2vis(m, this_pxscale_mas, lam, oifits=oifits, phot=phot, delta_pa=delta_pa)
#		if os.path.isfile(i.f_plot):
#			print(i.f_plot + " exists. Continuing.")
#			continue

		if oifits:
			### FIXING PA (of disk) to 44 deg for Circinus
			#i.optimize_pa(fixed_pa=44)
			i.optimize_pa(step=2,pa_init=pa_init,pa_max=pa_max)
	#	i.fixed_pa_best=True
		i.make_plot()
		##
		## rename file for particular scale
#		pdb.set_trace()
		if scale != 1.0:
			this_f_plot=i.f_plot.split(".png")[0]+"_"+"{0:.2f}".format(scale)+".png"
			os.rename(i.f_plot,this_f_plot)
