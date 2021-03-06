from img2vis import img2vis

##
## pixel scales (mas per px)
pxscale_1068 = 0.04 * 1000/71 ### 0.08 pc per px, 1000 mas = 71 pc in NGC 1068
pxscale_circ = 0.01333 * 1000/20 ### 0.08 pc per px, 1000 mas = 20 pc in Circinus
##
## total flux references:
## NGC 1068: 10 Jy at 10.0 micron (Raban+ 2009)
phot_1068_10mu = 10.0
##
## other source data
pa_circ_midi_disk = 44 ## east of north
##
## data references
oifits1068="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
oifitscirc="../MIDI_data/Circinus_clean.oifits"

pas=[]
chi2=[]

for pa in np.arange(18)*10:
	i=img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc, pa=pa)
	chi2.append(i.vis_chi2())
	pas.append(pa)

pa_min_chi2=40
i=img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc, pa=pa_min_chi2)
i.make_plot()

img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc)
img2vis("../models/Bernd_2016-05-13/n1068_bild_10_30L1.00000.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
img2vis("../models/Bernd_2016-05-13/n1068_bild_10_35L1.00000.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)

# test
# i=img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc, pa=20, binary=True)