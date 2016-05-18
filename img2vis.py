import numpy as np
from astropy.modeling import models
from matplotlib import pyplot as plt
from matplotlib import colors as c
import matplotlib.cm as cmx
import glob

from azimuthalAverage import azimuthalAverage
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.interpolation import rotate

import pdb


##
## FUNCTION read_midi_oifits
##
## PURPOSE
##    do what it says
##
def read_midi_oifits(f,lam,dlam,phot=False):
	hdu=fits.open(f)
	w=hdu[3].data
	ww=w["EFF_WAVE"]
	ix=(ww>lam-dlam)&(ww<lam+dlam)

	v=hdu[4].data

	if phot:
		raise ValueError("Need to check computation of errors in photometric mode...")
	
		vv=v["CFLUX"]
		vv/=phot
	else:
		vv = v["VISAMP"]
		vv_noise = v["VISAMPERR"]

	vis = np.average(vv[:,ix],axis=1)
	## average noise and divide by sqrt(n) for sample average
	vis_noise = np.average(vv_noise[:,ix],axis=1)/np.sqrt(np.sum(ix))
	
	u=v["UCOORD"]
	v=v["VCOORD"]
	bl=np.sqrt(u**2+v**2)
	pa=np.rad2deg(np.arctan(u/v))
	return(bl,pa,u,v,vis,vis_noise)

##
## FUNCTION modelvis
##
## return visibility at given u,v position
##
## NOTE: we are not using the class variables here but the possibly modified local variables
##       vis, fftscale (modified by vis_chi2)
##
def modelvis(u,v,vis,fftscale,roll):
	## round to nearest pixel position in image
	x=np.round(u/fftscale)
	y=roll+np.round(v/fftscale)
	if u<0:
		x=-x
		y=-y
	return(vis[y,x]) ## Python arrays are y,x not x,y...


class img2vis():
	"""
	A class to convert model surface brightness distributions to visibilities, make nice plots and adjust PA and scale so that the model matches with observed data
	
	NOTES
		PA is defined east of west, i.e. counter-clockwise on sky
			here we treat the image as an image on sky, i.e. pixel coordinates = - RA coordinates
			(the RA axis increases to the left, the pixel coordinates increase to the right)
			Since the same is true for both image and (u,v) plane, we just relabel to RA / (u,v) coordinates
			at the end and keep the image and the fourier transform in pixel space.

	
	PARAMETERS
	   f_model   path to FITS file of input model image
	   pxscale   pixel scale in milli-arcseconds (mas)
	   lam       wavelength in meters
	
	OPTIONAL PARAMETERS
	   oifits    path to OIFITS file containing visibilities for this object
	   phot      if OIFITS file has a CFLUX field (assuming: [Jy]), compute visibility with this total flux [Jy]
	   nikutta   Robert's models are stored in a particular FITS extension and need to be padded, 
						set to True when 	visualizing his models
		pa		Position Angle (east of west / counter-clockwise on sky) that the input image will be rotated before transforming it
		binary	if set to True, will add a binary to the image to test if the transform / plotting etc. works
	"""
	
	
	def __init__(self,f_model,pxscale,lam,oifits=False,phot=False,nikutta=False,pa=0,binary=False):
		self.f_model=f_model
		self.pxscale=pxscale
		self.lam=lam
		self.oifits=oifits
		self.nikutta=nikutta
		##
		## print some info
		print("Pixel scale: ", pxscale, " mas per pixel px")
		print("Wavelength: ", lam, " m")
		##
		## set parameters
		dlam=0.2e-6 ## half-width of wavelength box to extract visibilities from data

		hdu=fits.open(f_model)
		## rotate by PA (note: axis should be inverted, i.e. RA should be
		##    increasing to the left. But currently let's work in pixel space
		##    and relabel the x-axis later.

		##	
		if self.nikutta:
			if pa != 0:
				raise ValueError("Not sure if padding for Nikutta models work with rotation")
			self.img = self.img[6,:,:]
			##
			## need to pad image with 0s to get enough higher Fourier frequencies
			# images are 101x101, let's pad 100 px each side, i.e. final img is 301x301
			img_pad = np.zeros([301,301])
			img_pad[100:201,100:201] = self.img
			self.img=img_pad
		
		self.img=hdu[0].data

		if binary:
			##
			## add point source to image (for testing purposes)
			binsep_mas = 10
			binsep_px = binsep_mas / pxscale
			print("Binary separation: ", binsep_mas, " mas")
			self.img[250,250] += 100000
			self.img[250,250+binsep_px] += 100000

		## "rotate" rotates clockwise (careful: matplotlib.imshow uses origin="upper" per default)
		##  PA is counter-clockwise on sky (see note above for coordinate systems)
		self.img=rotate(self.img, -pa, reshape=True)

		gridsize=self.img.shape[0]

		##
		## =============== compute FFT frequencies and scale ===============
		##
		fft_freq = np.fft.fftfreq(gridsize,pxscale)
		fft_img = np.abs(np.fft.rfft2(self.img))
		##
		## determine norm for visibilities, roll axes so that values start at 0 freq.
		self.roll=np.round(fft_img.shape[0]/2)
		r=self.roll.astype("int")
		vis_norm=fft_img[0,0]
		self.vis=np.roll(fft_img,r,0)/vis_norm
		freq=np.roll(fft_freq,r,0)
		##
		## pxscale -> fftscale
		fftscale = np.diff(freq)[0] ## cycles / mas per pixel in FFT image
		mas2rad=np.deg2rad(1/3600000) ## mas per rad
		self.fftscale = fftscale/mas2rad * lam ## meters baseline at given lam per px in FFT image
		print("Pixel scale in FFT image is: ", fftscale, " m (Baseline) per pixel")
		##
		## read observed data
		if oifits:
			if phot:
				self.bl,self.pa,self.u,self.v,self.vis_obs,self.vis_obs_noise=read_midi_oifits(oifits,lam,dlam,phot=phot)
			else:
				self.bl,self.pa,self.u,self.v,self.vis_obs,self.vis_obs_noise=read_midi_oifits(oifits,lam,dlam,phot=False)

	##
	## METHOD vis_chi2
	##
	## rotate and scale model FFT image to get best match with observed visibilities
	##
	## pa: position angle (deg) by which **image plane** is rotated
	## scale: scale factor by which **image plane** is shrinked / enlarged
	##
#	def vis_chi2(self,pa,scale):
	def vis_chi2(self):
#		fft_pa_rad = np.deg2rad(pa+90)
#		fft_scale = 1/scale
#		## rotate and rescale FFT image according to pa, scale
#		fftscale=self.fftscale*fft_scale
#		vis = self.vis
		
		chi2=0
		for u,v,vis_obs,vis_obs_noise in zip(self.u,self.v,self.vis_obs,self.vis_obs_noise):
			chi2 += (vis_obs - modelvis(u,v,self.vis,self.fftscale,self.roll))**2/vis_obs_noise**2
		return chi2



	##
	## METHOD make_plot
	##
	## plot and save model, FFT, azimuthal average and observed data
	##
	def make_plot(self):
		##
		## =============== make nice plot ===============
		##
		plt.subplot(221)
		## cut out central region
		c_mas = 100 ## half-size of cut-out box (in mas)
		if self.nikutta:
			c_mas = 45
		c_px = c_mas/self.pxscale
		p1 = np.shape(self.img)[0]/2 - c_px
		p2 = np.shape(self.img)[0]/2 + c_px
		img_cut = self.img[p1:p2,p1:p2]
		if self.nikutta:
			img_cut = self.img
		norm = np.median(img_cut) + 10*np.std(img_cut)
	#	norm = np.max(img_cut) ## leads to very shallow images if central point source is not removed
		plt.imshow(img_cut/norm,origin="lower",vmin=0,vmax=1)
		plt.colorbar(label="Normalized intensity")
		plt.title("Image plane")
		##
		## set number of axis labels
		nax=5
		xt = 2 * c_px * np.arange(nax+1)/nax
		xt_label = self.pxscale * xt
		plt.xticks(xt,xt_label - c_mas)
		plt.yticks(xt,xt_label - c_mas)
		plt.xlabel("x [mas]")
		plt.ylabel("y [mas]")


		plt.subplot(222)
		plt.imshow(self.vis,origin="lower")
		plt.colorbar(label="Visibility amplitude")
		plt.title("Fourier plane")

		max_bl=130
		numpoints=3
	
		xt = np.arange(numpoints)/(numpoints-1) * max_bl
		yt = (-1 + np.arange(2*numpoints-1)/(numpoints-1)) * max_bl
		plt.xticks(xt/self.fftscale, (-xt).astype(int))
		plt.yticks(self.roll+yt/self.fftscale, yt.astype(int))
	#	pdb.set_trace()
	
	
	#	nax=20
	#	xt = vis.shape[1] * np.arange(nax+1)/nax
	#	yt = vis.shape[0] * np.arange(2*nax+1)/(2*nax)
	#	plt.xticks(xt,-(fftscale*xt).astype(int))
	#	plt.yticks(yt,(fftscale*(yt-self.roll)).astype(int))
		plt.xlabel("u [m]")
		plt.ylabel("v [m]")
		xylim=np.round(max_bl/self.fftscale)
		plt.xlim([0,xylim])
		plt.ylim([self.roll-xylim,self.roll+xylim])


		plt.subplot(223)
		if self.oifits:
			plt.plot(self.bl,self.vis_obs,'ks',label="MIDI data")
			plt.legend(numpoints=1)
		if self.nikutta:
			r,V = azimuthalAverage(self.vis,center=[0,self.roll],returnradii=True,binsize=2)
		else:
			r,V = azimuthalAverage(self.vis,center=[0,self.roll],returnradii=True)
	#	pdb.set_trace()
		plt.plot(self.fftscale * r,V, label="model")
		plt.ylim([0,1])
		plt.xlim([0,130])
		plt.xlabel("Projected baseline [m]")
		plt.ylabel("Visibility amplitude")
		plt.title("Azimuthally averaged visamp")

		if self.oifits:
			plt.subplot(224)
		#	plt.plot(fftscale * np.arange(vis.shape[1]),vis[roll,:])
		#	plt.ylim([0,1])
		#	plt.xlim([0,130])
		#	plt.xlabel("Projected baseline [m]")
		#	plt.ylabel("Visibility amplitude")
		#	plt.title("Meridional cut through (u,v) plane")
			cNorm=c.Normalize(vmin=0,vmax=1.0)
			scalarMap=cmx.ScalarMappable(norm=cNorm,cmap="rainbow")
			for iu,iv,ivis in zip(self.u,self.v,self.vis_obs):
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
		plt.suptitle(self.f_model.split(".fits")[0] + str(" (lam={0} m)".format(self.lam)),fontsize=6)

		plt.savefig(self.f_model.split(".fits")[0]+".png")
		plt.clf()