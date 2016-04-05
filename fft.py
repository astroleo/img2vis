import numpy as np
from astropy.modeling import models
from matplotlib import pyplot as plt
from matplotlib import colors as c
import matplotlib.cm as cmx

from azimuthalAverage import azimuthalAverage
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline

import pdb

def read_midi_oifits(f,lam,dlam):
	hdu=fits.open(f)
	w=hdu[3].data
	ww=w["EFF_WAVE"]
	ix=(ww>lam-dlam)&(ww<lam+dlam)
	
	v=hdu[4].data
	vv=v["VISAMP"]
	vis = np.average(vv[:,ix],axis=1)
	u=v["UCOORD"]
	v=v["VCOORD"]
	bl=np.sqrt(u**2+v**2)
	pa=np.rad2deg(np.arctan(u/v))
	return(bl,pa,u,v,vis)

##
## function fft
##
## PURPOSE
##    take a model image as input, calculate visibilities and overlay observed data
##
## PARAMETERS
##    f_model   path to FITS file of input model image
##    pxscale   pixel scale in milli-arcseconds (mas)
##    lam       wavelength in meters
##
## OPTIONAL PARAMETERS
##    oifits    path to OIFITS file containing visibilities for this object
##
def fft(f_model,pxscale,lam,oifits=False):
	##
	## print some info
	print("Pixel scale: ", pxscale, " mas per pixel px")
	print("Wavelength: ", lam, " m")
	##
	## set parameters
	dlam=0.2e-6 ## half-width of wavelength box to extract visibilities from data

	hdu=fits.open(f_model)
	img=hdu[0].data
	gridsize=img.shape[0]
	##
	## add point source to image (for testing purposes)
#	binsep_mas = 10
#	binsep_px = binsep_mas / pxscale
#	print("Binary separation: ", binsep_mas, " mas")
#	img[250,250] += 100000
#	img[250,250+binsep_px] += 100000
	##
	## =============== compute FFT frequencies and scale ===============
	##
	fft_freq = np.fft.fftfreq(gridsize,pxscale)
	fft_img = np.abs(np.fft.rfft2(img))
	##
	## determine norm for visibilities, roll axes so that values start at 0 freq.
	roll=np.round(fft_img.shape[0]/2)
	r=roll.astype("int")
	vis_norm=fft_img[0,0]
	vis=np.roll(fft_img,r,0)/vis_norm
	freq=np.roll(fft_freq,r,0)
	##
	## pxscale -> fftscale
	fftscale = np.diff(freq)[0] ## cycles / mas per pixel in FFT image
	mas2rad=np.deg2rad(1/3600000) ## mas per rad
	fftscale = fftscale/mas2rad * lam ## meters baseline at given lam per px in FFT image
	print("Pixel scale in FFT image is: ", fftscale, " m (Baseline) per pixel")
	##
	## read observed data
	if oifits:
		bl,pa,u,v,vis_obs=read_midi_oifits(oifits,lam,dlam)
	##
	## =============== make nice plot ===============
	##
	plt.subplot(221)
	## cut out central region
	c_mas = 100 ## half-size of cut-out box (in mas)
	c_px = c_mas/pxscale
	p1 = np.shape(img)[0]/2 - c_px
	p2 = np.shape(img)[0]/2 + c_px
	img_cut = img[p1:p2,p1:p2]
	plt.imshow(img_cut/np.max(img_cut),origin="lower")
	plt.colorbar(label="Normalized intensity")
	plt.title("Image plane")
	##
	## set number of axis labels
	nax=5
	xt = 2 * c_px * np.arange(nax+1)/nax
	xt_label = pxscale * xt
	plt.xticks(xt,xt_label - c_mas)
	plt.yticks(xt,xt_label - c_mas)
	plt.xlabel("x [mas]")
	plt.ylabel("y [mas]")


	plt.subplot(222)
	plt.imshow(vis,origin="lower")
	plt.colorbar(label="Visibility amplitude")
	plt.title("Fourier plane")
	nax=5
	xt = vis.shape[1] * np.arange(nax+1)/nax
	yt = vis.shape[0] * np.arange(2*nax+1)/(2*nax)
	plt.xticks(xt,-(fftscale*xt).astype(int))
	plt.yticks(yt,(fftscale*(yt-roll)).astype(int))
	plt.xlabel("u [m]")
	plt.ylabel("v [m]")
	max_bl = 130
	xylim=np.round(max_bl/fftscale)
	plt.xlim([0,xylim])
	plt.ylim([roll-xylim,roll+xylim])


	plt.subplot(223)
	r,V = azimuthalAverage(vis,center=[0,roll],returnradii=True)
	plt.plot(fftscale * r,V)
	plt.ylim([0,1])
	plt.xlim([0,130])
	plt.xlabel("Projected baseline [m]")
	plt.ylabel("Visibility amplitude")
	plt.title("Azimuthally averaged visibility amplitude")
	if oifits:
		plt.plot(bl,vis_obs,'ks',label="MIDI data")
		plt.legend(numpoints=1)


	plt.subplot(224)
#	plt.plot(fftscale * np.arange(vis.shape[1]),vis[roll,:])
#	plt.ylim([0,1])
#	plt.xlim([0,130])
#	plt.xlabel("Projected baseline [m]")
#	plt.ylabel("Visibility amplitude")
#	plt.title("Meridional cut through (u,v) plane")
	cNorm=c.Normalize(vmin=0,vmax=1.0)
	scalarMap=cmx.ScalarMappable(norm=cNorm,cmap="rainbow")
#	pdb.set_trace()
	for iu,iv,ivis in zip(u,v,vis_obs):
		plt.plot(-iu,iv,color=scalarMap.to_rgba(ivis),mew=0,ms=4,marker="o")
		plt.plot(iu,-iv,color=scalarMap.to_rgba(ivis),mew=0,ms=4,marker="o")
	plt.xlim([-130,130])
	plt.ylim([-130,130])
	xyticks=np.array([-100,-50,0,50,100])
	plt.xticks(xyticks,-xyticks)
	plt.yticks(xyticks,xyticks)
	plt.xlabel("u [m]")
	plt.ylabel("v [m]")
	plt.colorbar()
	plt.title("Observed (u,v) plane")
	


	plt.suptitle(f_model.split(".fits")[0] + str(" (lam={0} m)".format(lam)))

	plt.tight_layout()

	plt.savefig(f_model.split(".fits")[0]+".png")
	plt.clf()



# pixel scale:
pxscale_1068 = 0.08 * 1000/71 ### 0.08 pc per px, 1000 mas = 71 pc in NGC 1068
pxscale_circ = 0.08 * 1000/20 ### 0.08 pc per px, 1000 mas = 20 pc in Circinus
pxscale_circ_hr = 0.04 * 1000/20 ### 0.04 pc per px, 1000 mas = 20 pc in Circinus / high-res images


fft("bild_circinus_1_65_70.fits", pxscale_circ, 1e-5, oifits="../img2vis/MIDI_data/Circinus_clean.oifits")
fft("bild_circinus_1_65_70_hr.fits", pxscale_circ_hr, 1e-5, oifits="../img2vis/MIDI_data/Circinus_clean.oifits")

fft("circinus_bild_10_fo.fits", pxscale_circ, 1e-5, oifits="../img2vis/MIDI_data/Circinus_clean.oifits")
fft("circinus_bild_10.fits", pxscale_circ, 1e-5, oifits="../img2vis/MIDI_data/Circinus_clean.oifits")


###fft("bild_1068.fits",0.04 * 1000/71,1e-5,"../img2vis/MIDI_data/NGC1068.oifits") ## note: not yet proper OIFITS!

#fft("bild_circinus_1_65_70_hr.fits","circinus")
#fft("bild_circinus_1_65_70.fits","circinus")

#fft("bild_n1068_1_30_70_055_hr.fits","ngc1068")
#fft("bild_n1068_1_30_70_055.fits","ngc1068")
#fft("bild_n1068_1_60_70_07_hr.fits","ngc1068")
#fft("bild_n1068_1_60_70_07.fits","ngc1068")
#fft("bild_n1068_1_60_70_11_hr.fits","ngc1068")
#fft("bild_n1068_1_60_70_11.fits","ngc1068")
#fft("bild_n1068_1_60_70_055_hr.fits","ngc1068")
#fft("bild_n1068_1_60_70_055.fits","ngc1068")