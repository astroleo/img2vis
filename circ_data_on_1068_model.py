from img2vis import img2vis
import glob
import numpy as np
import pdb

##
## all 10 mu models of 1068
models=glob.glob("../models/Bernd_2016-05-17/circ_data_on_1068_model/*.fits")

for m in models:
	L_scale = np.float(m.split(".fits")[0].split("L")[1])
	obj = m.split("/")[-1].split("_")[0]
	lam = np.round(np.float(m.split("/")[-1].split("_")[2])) * 1e-6
	oifits=False
	phot=False
	
	if obj=="n1068":
		mas_per_pc = 1000/71
		delta_pa=15
		oifits = "../MIDI_data/Circinus_clean.oifits"
	
#	pdb.set_trace()
	pxscale_pc = 0.04 * np.sqrt(L_scale)
	pxscale_mas = pxscale_pc * mas_per_pc

	i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot, delta_pa=delta_pa)
	if oifits:
		i.optimize_pa()
	i.make_plot()
