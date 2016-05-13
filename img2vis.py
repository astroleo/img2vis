import numpy as np
from astropy.modeling import models
from matplotlib import pyplot as plt
from matplotlib import colors as c
import matplotlib.cm as cmx
import glob

from azimuthalAverage import azimuthalAverage
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline

import pdb

def read_midi_oifits(f,lam,dlam,phot=False):
	hdu=fits.open(f)
	w=hdu[3].data
	ww=w["EFF_WAVE"]
	ix=(ww>lam-dlam)&(ww<lam+dlam)
	
	v=hdu[4].data
	if phot:
		vv=v["CFLUX"]
		vv/=phot
	else:
		vv=v["VISAMP"]
	vis = np.average(vv[:,ix],axis=1)
	u=v["UCOORD"]
	v=v["VCOORD"]
	bl=np.sqrt(u**2+v**2)
	pa=np.rad2deg(np.arctan(u/v))
	return(bl,pa,u,v,vis)

def vis_chi2(fftscale, fft_img, uv_obs, vis_obs):
	pass

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
##    phot      if OIFITS file has a CFLUX field (assuming: [Jy]), compute visibility with this total flux [Jy]
##    nikutta   Robert's models are stored in a particular FITS extension and need to be padded, set to True when visualizing his models
##
def img2vis(f_model,pxscale,lam,oifits=False,phot=False,nikutta=False):
	##
	## print some info
	print("Pixel scale: ", pxscale, " mas per pixel px")
	print("Wavelength: ", lam, " m")
	##
	## set parameters
	dlam=0.2e-6 ## half-width of wavelength box to extract visibilities from data

	hdu=fits.open(f_model)
	img=hdu[0].data

	##	
	if nikutta:
		img = img[6,:,:]
		##
		## need to pad image with 0s to get enough higher Fourier frequencies
		# images are 101x101, let's pad 100 px each side, i.e. final img is 301x301
		img_pad = np.zeros([301,301])
		img_pad[100:201,100:201] = img
		img=img_pad

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
		if phot:
			bl,pa,u,v,vis_obs=read_midi_oifits(oifits,lam,dlam,phot=phot)
		else:
			bl,pa,u,v,vis_obs=read_midi_oifits(oifits,lam,dlam,phot=False)

	##
	## =============== make nice plot ===============
	##
	plt.subplot(221)
	## cut out central region
	c_mas = 100 ## half-size of cut-out box (in mas)
	if nikutta:
		c_mas = 45
	c_px = c_mas/pxscale
	p1 = np.shape(img)[0]/2 - c_px
	p2 = np.shape(img)[0]/2 + c_px
	img_cut = img[p1:p2,p1:p2]
	if nikutta:
		img_cut = img
	norm = np.median(img_cut) + 10*np.std(img_cut)
#	norm = np.max(img_cut) ## leads to very shallow images if central point source is not removed
	plt.imshow(img_cut/norm,origin="lower",vmin=0,vmax=1)
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

	max_bl=130
	numpoints=3
	
	xt = np.arange(numpoints)/(numpoints-1) * max_bl
	yt = (-1 + np.arange(2*numpoints-1)/(numpoints-1)) * max_bl
	plt.xticks(xt/fftscale, (-xt).astype(int))
	plt.yticks(roll+yt/fftscale, yt.astype(int))
#	pdb.set_trace()
	
	
#	nax=20
#	xt = vis.shape[1] * np.arange(nax+1)/nax
#	yt = vis.shape[0] * np.arange(2*nax+1)/(2*nax)
#	plt.xticks(xt,-(fftscale*xt).astype(int))
#	plt.yticks(yt,(fftscale*(yt-roll)).astype(int))
	plt.xlabel("u [m]")
	plt.ylabel("v [m]")
	xylim=np.round(max_bl/fftscale)
	plt.xlim([0,xylim])
	plt.ylim([roll-xylim,roll+xylim])


	plt.subplot(223)
	if oifits:
		plt.plot(bl,vis_obs,'ks',label="MIDI data")
		plt.legend(numpoints=1)
	if nikutta:
		r,V = azimuthalAverage(vis,center=[0,roll],returnradii=True,binsize=2)
	else:
		r,V = azimuthalAverage(vis,center=[0,roll],returnradii=True)
#	pdb.set_trace()
	plt.plot(fftscale * r,V, label="model")
	plt.ylim([0,1])
	plt.xlim([0,130])
	plt.xlabel("Projected baseline [m]")
	plt.ylabel("Visibility amplitude")
	plt.title("Azimuthally averaged visamp")

	if oifits:
		plt.subplot(224)
	#	plt.plot(fftscale * np.arange(vis.shape[1]),vis[roll,:])
	#	plt.ylim([0,1])
	#	plt.xlim([0,130])
	#	plt.xlabel("Projected baseline [m]")
	#	plt.ylabel("Visibility amplitude")
	#	plt.title("Meridional cut through (u,v) plane")
		cNorm=c.Normalize(vmin=0,vmax=1.0)
		scalarMap=cmx.ScalarMappable(norm=cNorm,cmap="rainbow")
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
		plt.colorbar(label="Visibility amplitude")
		plt.title("Observed (u,v) plane")
	



	plt.tight_layout()
	plt.suptitle(f_model.split(".fits")[0] + str(" (lam={0} m)".format(lam)),fontsize=6)

	plt.savefig(f_model.split(".fits")[0]+".png")
	plt.clf()