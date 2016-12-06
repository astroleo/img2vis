from img2vis import img2vis
import numpy as np

#m="../models/Bernd_2016-05-17/n1068_bild_10_45L1.00000.fits"
m="../models/Bernd_2016-05-17/circinus_bild_10_35L0.333333.fits"

## compute px scale
L_scale = np.float(m.split(".fits")[0].split("L")[1])
pxscale_pc = 0.04 * np.sqrt(L_scale)
mas_per_pc=1000/20
pxscale_mas = pxscale_pc * mas_per_pc

lam=1e-5
#oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
#phot=10.0
#i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot)

oifits = "../MIDI_data/Circinus_clean.oifits"
i=img2vis(m, pxscale_mas, lam, oifits=oifits)
i.optimize_pa(fixed_pa=44)
i.make_plot()