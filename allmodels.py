from img2vis import img2vis
import glob
import numpy as np
import pdb

##
## all 10 mu models
models=glob.glob("../models/Bernd_2016-05-17/*bild_10_*.fits")
##
## only 1068, 10 micron models with 1068 luminosity
#models=glob.glob("../models/Bernd_2016-05-17/n1068_bild_10_*L1.00000.fits")
##
## only 1068, 10 micron models
#models=glob.glob("../models/Bernd_2016-05-17/n1068_bild_10_*.fits")
##
## selected Circinus models
#models=glob.glob("../models/Bernd_2016-05-17/circinus_bild_10_*L0.111111.fits")


j=0

phot_1068_10mu = 10.0

for m in models:
	L_scale = np.float(m.split(".fits")[0].split("L")[1])
	obj = m.split("/")[-1].split("_")[0]
	lam = np.round(np.float(m.split("/")[-1].split("_")[2])) * 1e-6
	oifits=False
	phot=False
	
	if obj=="n1068":
		mas_per_pc = 1000/71
		delta_pa=25
		if np.isclose(lam,1e-5):
			oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
			phot = phot_1068_10mu
	elif obj=="circinus":
		mas_per_pc = 1000/20
		delta_pa=15
		if np.isclose(lam,1e-5):
			oifits = "../MIDI_data/Circinus_clean.oifits"
	else:
		raise ValueError("Object {0} not known".format(obj))
	
#	pdb.set_trace()
	pxscale_pc = 0.04 * np.sqrt(L_scale)
	pxscale_mas = pxscale_pc * mas_per_pc

	i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot, delta_pa=delta_pa)
	if oifits:
		i.optimize_pa()
	i.make_plot()
	j+=1