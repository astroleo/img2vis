from img2vis import img2vis
import glob
import numpy as np
import pdb

models=glob.glob("../models/Bernd_2016-05-17/*.fits")
j=0

phot_1068_10mu = 10.0

for m in models:
	L_scale = np.float(m.split(".fits")[0].split("L")[1])
	obj = m.split("/")[-1].split("_")[0]
	lam = np.round(np.float(m.split("/")[-1].split("_")[2])) * 1e-6
	oifits=False
	phot=False
	
	if obj=="circinus":
		continue

	if obj=="n1068":
		mas_per_pc = 1000/71
		if np.isclose(lam,1e-5):
			#oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
			#phot = phot_1068_10mu
			##
			## try 1068 model with Circinus visibilities
			oifits = "../MIDI_data/Circinus_clean.oifits"
	elif obj=="circinus":
		mas_per_pc = 1000/20
		if np.isclose(lam,1e-5):
			oifits = "../MIDI_data/Circinus_clean.oifits"
	else:
		raise ValueError("Object {0} not known".format(obj))
	
#	pdb.set_trace()
	pxscale_pc = 0.04 * np.sqrt(L_scale)
	pxscale_mas = pxscale_pc * mas_per_pc

	i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot)
	if oifits:
		i.optimize_pa()
	i.make_plot()
	j+=1