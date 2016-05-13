from img2vis import img2vis

##
## pixel scales (mas per px)
pxscale_1068 = 0.08 * 1000/71 ### 0.08 pc per px, 1000 mas = 71 pc in NGC 1068
pxscale_1068_hr = 0.04 * 1000/71 ### 0.08 pc per px, 1000 mas = 71 pc in NGC 1068
pxscale_circ = 0.08 * 1000/20 ### 0.08 pc per px, 1000 mas = 20 pc in Circinus
pxscale_circ_hr = 0.04 * 1000/20 ### 0.04 pc per px, 1000 mas = 20 pc in Circinus / high-res images
##
## total flux references:
## NGC 1068: 10 Jy at 10.0 micron (Raban+ 2009)
phot_1068_10mu = 10.0
##
## data references
oifits1068="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
oifitscirc="../MIDI_data/Circinus_clean.oifits"

#fft("../models/Bernd_2016-03-14/bild_circinus_1_65_70.fits", pxscale_circ, 1e-5, oifits=oifitscirc)
#fft("../models/Bernd_2016-03-14/bild_circinus_1_65_70_hr.fits", pxscale_circ_hr, 1e-5, oifits=oifitscirc)

#fft("../models/Bernd_2016-03-14/bild_n1068_1_30_70_055_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_30_70_055.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_07_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_07.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_11_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_11.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_055_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-03-14/bild_n1068_1_60_70_055.fits", pxscale_1068, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)


#fft("../models/Schartmann_2009/data/maug_000_00.fits", 1.14, 1.2e-5)
#fft("../models/Schartmann_2009/data/maug_a00_30.fits", 1.14, 1.2e-5)
#fft("../models/Schartmann_2009/data/maug_a00_60.fits", 1.14, 1.2e-5)
#fft("../models/Schartmann_2009/data/maug_a00_90.fits", 1.14, 1.2e-5)
#fft("../models/Schartmann_2009/data/maug_b00_08.fits", 1.14, 1.2e-5)

#models=glob.glob("../models/Schartmann_2008/data/*.fits")
#for m in models:
#	hdr = fits.getheader(m)
#	pixsize=1000*hdr["PIXSIZE"]
#	lam=hdr["LAMBDA1"]
#	fft(m,pixsize,lam)


#fft("../models/Bernd_2016-04-05/bild_circinus2_60_70_hr.fits", pxscale_circ_hr, 1e-5, oifits=oifitscirc)
#fft("../models/Bernd_2016-04-05/bild_circinus2_65_70_hr.fits", pxscale_circ_hr, 1e-5, oifits=oifitscirc)
#fft("../models/Bernd_2016-04-05/bild_n1068_2_60_70_055_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)
#fft("../models/Bernd_2016-04-05/bild_n1068_2_65_70_055_hr.fits", pxscale_1068_hr, 1e-5, oifits=oifits1068, phot=phot_1068_10mu)

##
###### =============== NIKUTTA MODELS ===============
##
#10 pixels = 2 R_d
#1 px = 0.2 R_d; R_d = 5 px
## assume we have Circinus: R_d = 0.03 pc -> pxscale = 0.006 pc * 1000 mas/20 pc = 0.3 mas
pxscale_nikutta = 0.3
fft("../models/Nikutta/IMG-AA00-TORUSG-sig30-i66.422-Y010-N05.0-q0.0-tv020.0.fits", pxscale_nikutta, 1.25e-5, oifits=oifitscirc)

