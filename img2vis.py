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
from scipy.ndimage import map_coordinates

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
		vv=v["CFLUX"]
		vv/=phot
		vv_noise=v["CFLUXERR"]
		vv_noise/=phot
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
## NOTE about coordinate systems: we are only plotting and using the "right", 
##    i.e. negative "u" side of the (u,v) plane, but pixel coordinates are positive
##    from the origin, i.e. we need to invert the u coordinate here.
##
def modelvis(u,v,vis,fftscale,roll):
	## round to nearest pixel position in image
	u=-u
	x=np.round(u/fftscale)
	y=roll+np.round(v/fftscale)
	if u<0:
		x=-x
		y=-y
	return(vis[y,x]) ## Python arrays are y,x not x,y...


class img2vis():
	"""
	A class to convert model surface brightness distributions to visibilities, 
	make nice plots and adjust PA and scale so that the model matches with observed data
	
	NOTES
		PA is defined east of north, i.e. counter-clockwise on sky
		here we treat the image as an image on sky, i.e. pixel coordinates = - RA coordinates
		(the RA axis increases to the left, the pixel coordinates increase to the right)
		Since the same is true for both image and (u,v) plane, we just relabel to RA / (u,v) coordinates
		at the end and keep the image and the fourier transform in pixel space.
	
	PARAMETERS
		f_model		path to FITS file of input model image
		pxscale		pixel scale in milli-arcseconds (mas)
		lam			wavelength in meters
	
	OPTIONAL PARAMETERS
		oifits		path to OIFITS file containing visibilities for this object
		phot		if OIFITS file has a CFLUX field (assuming: [Jy]), compute visibility with this total flux [Jy]
		nikutta		Robert's models are stored in a particular FITS extension and need to be padded, 
						set to True when visualizing his models
		pa			Position Angle (east of north / counter-clockwise on sky) that the input image will be rotated before transforming it
		binary		if set to True, will add a binary to the image to test if the transform / plotting etc. works
		delta_pa    half-range of PA to use for PA selection of observed visibilities,
						i.e. data with PA +/- self.delta_pa will be chosen for data-model
						comparisons

	"""
	
	
	def __init__(self, f_model, pxscale, lam, oifits=False, phot=False, nikutta=False, pa=0, binary=False, delta_pa=15):
		self.f_model=f_model
		self.pxscale=pxscale
		self.lam=lam
		self.oifits=oifits
		self.nikutta=nikutta
		self.pa_best=False
		self.phot=phot
		self.delta_pa=delta_pa		
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
			binsep_mas=10
			binsep_px=binsep_mas / pxscale
			print("Binary separation: ", binsep_mas, " mas")
			self.img[250,250]+=100000
			self.img[250,250+binsep_px]+=100000

		##
		## perform the actual FFT
		self.rotate_and_fft(pa,verbose=True)
		
		##
		## read observed data
		if self.oifits:
			self.bl,self.pa,self.u,self.v,self.vis_obs,self.vis_obs_noise=read_midi_oifits(oifits,lam,dlam,phot=phot)
		
	def rotate_and_fft(self,pa,verbose=False):
		## "rotate" rotates clockwise (careful: matplotlib.imshow uses origin="upper" per default)
		##  PA is counter-clockwise on sky (see note above for coordinate systems)
		self.img=rotate(self.img, -pa, reshape=True)

		gridsize=self.img.shape[0]

		##
		## =============== compute FFT frequencies and scale ===============
		##
		fft_freq=np.fft.fftfreq(gridsize,self.pxscale)
		fft_img=np.abs(np.fft.rfft2(self.img))
		##
		## determine norm for visibilities, roll axes so that values start at 0 freq.
		self.roll=np.round(fft_img.shape[0]/2)
		r=self.roll.astype("int")
		vis_norm=fft_img[0,0]
		self.vis=np.roll(fft_img,r,0)/vis_norm
		freq=np.roll(fft_freq,r,0)
		##
		## pxscale -> fftscale
		fftscale=np.diff(freq)[0] ## cycles / mas per pixel in FFT image
		mas2rad=np.deg2rad(1/3600000) ## mas per rad
		self.fftscale = fftscale/mas2rad * self.lam ## meters baseline per px in FFT image at given wavelength
		if verbose:
			print("Pixel scale in FFT image is: ", self.fftscale, " m (Baseline) per pixel")


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
			if vis_obs_noise == 0:
				raise ValueError("vis_obs_noise must not be 0")
			chi2 += (vis_obs - modelvis(u,v,self.vis,self.fftscale,self.roll))**2/vis_obs_noise**2
		return chi2

	def optimize_pa(self):
	
		self.pas=[]
		self.chi2=[]

		for pa in np.arange(18)*10:
			##
			## this could be made more efficient by only calling the sub-routine rotate_and_fft
			## but then need to watch out for differential vs. absolute rotations
			i=img2vis(self.f_model, self.pxscale, self.lam, oifits=self.oifits, pa=pa, phot=self.phot)
			self.chi2.append(i.vis_chi2())
			self.pas.append(pa)
		
		## chose global chi2 minimum here for the moment, and plot PA vs. chi2 
		##    so that we see if global minimum is bad.
		self.pa_best = self.pas[np.argmin(self.chi2)]
		self.rotate_and_fft(self.pa_best)

	##
	## METHOD make_plot
	##
	## plot and save model, FFT, azimuthal average and observed data
	##
	def make_plot(self):
		##
		## =============== model image ===============
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

		##
		## =============== model visibilities on (u,v) plane ===============
		##
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
		
		##
		## choose two axes to show radial profiles, one along pa_best and one perpendicular to that
		## following http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
		r_m = 100 ## length of line in meters
		r_px = r_m/self.fftscale ## length of line in px coordinates
		num = r_m ## number of points to choose for interpolation
		if not self.pa_best:
			self.pa_best=0
		
		## pa1: perpendicular to pa_best
		## pa2: parallel to pa_best
		pa1 = 90 - self.pa_best
		pa2 = 180 - self.pa_best
		if self.pa_best > 90:
			pa1 = 270 - self.pa_best
			pa2 = 180 - self.pa_best

		x0, y0 = 0, self.roll
		x1, y1 = np.sin(np.pi/180 * (pa1)) * r_px, np.cos(np.pi/180 * (pa1)) * r_px + self.roll
		x,y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)
		visi1 = map_coordinates(self.vis, np.vstack((y,x))) ## interpolated model visibilities
		plt.plot([x0,x1],[y0,y1],'ro-')
		
		x1, y1 = np.sin(np.pi/180 * (pa2)) * r_px, np.cos(np.pi/180 * (pa2)) * r_px + self.roll
		x,y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)
		visi2 = map_coordinates(self.vis, np.vstack((y,x))) ## interpolated model visibilities
		plt.plot([x0,x1],[y0,y1],'go-')
		
#		pdb.set_trace()


		##
		## =============== chi^2 values as a function of PA ===============
		##
		plt.subplot(223)
		if self.oifits:	
			plt.plot(self.pas,self.chi2)
			plt.plot([self.pa_best,self.pa_best],plt.ylim(),'r')
			plt.xlabel("Position Angle (East of North)")
			plt.ylabel("chi square")
			plt.title("Optimal rotation of model")
		else:
			plt.plot(0,0)
			plt.title("No data available")

		##
		## =============== radial cuts along specific PA ===============
		##
		plt.subplot(224)
		plt.plot(np.arange(num), visi1, 'r',label="Model at PA = {0}".format(90+self.pa_best))
		plt.plot(np.arange(num), visi2, 'g',label="Model at PA = {0}".format(self.pa_best))
		plt.xlabel("Projected Baseline length [m]")
		plt.ylabel("Visibility")
		
		if self.oifits:
			##
			## overplot data with similar PA range
			##
			## first: move PAs into regime 0-180
			pa = self.pa
			pa[pa < 0] += 180
			pa[pa > 180] -= 180
			## and create complimentary pa's in range 180-360:
			pa_180 = pa+180
			##
			## then: move pa_best (by construction within [0,180]) in regime 0-180
			if self.pa_best < 0:
				pa_best += 180
			else:
				pa_best = self.pa_best
			
			##
			## pa_best and pa are now in range 0-180, but pa_best+90+self.delta_pa
			##    could be > 180, so compare against both pa and pa_180 and then 
			##    choose the combination of both
			ix1 = (pa > (pa_best+90 - self.delta_pa)) & (pa < (pa_best+90 + self.delta_pa))
			ix2 = (pa_180 > (pa_best+90 - self.delta_pa)) & (pa_180 < (pa_best+90 + self.delta_pa))
			ix = np.any([ix1,ix2],axis=0)			
			plt.plot(self.bl[ix],self.vis_obs[ix],'rx',label="MIDI, PA = {0} +/- {1}".format(self.pa_best+90,self.delta_pa))

			ix1 = (pa > (pa_best - self.delta_pa)) & (pa < (pa_best + self.delta_pa))
			ix2 = (pa_180 > (pa_best - self.delta_pa)) & (pa_180 < (pa_best + self.delta_pa))
			ix = np.any([ix1,ix2],axis=0)				
			plt.plot(self.bl[ix],self.vis_obs[ix],'gx',label="MIDI, PA = {0} +/- {1}".format(self.pa_best,self.delta_pa))

		plt.legend(numpoints=1,fontsize=8)
		
		
		
### below: old sub-plots showing azimuthally averaged visibilities (plot 3) and observed visibilities on (u,v) plane (plot 4)
# 		plt.subplot(223)
# 		if self.oifits:
# 			plt.plot(self.bl,self.vis_obs,'ks',label="MIDI data")
# 			plt.legend(numpoints=1)
# 		if self.nikutta:
# 			r,V = azimuthalAverage(self.vis,center=[0,self.roll],returnradii=True,binsize=2)
# 		else:
# 			r,V = azimuthalAverage(self.vis,center=[0,self.roll],returnradii=True)
# 	#	pdb.set_trace()
# 		plt.plot(self.fftscale * r,V, label="model")
# 		plt.ylim([0,1])
# 		plt.xlim([0,130])
# 		plt.xlabel("Projected baseline [m]")
# 		plt.ylabel("Visibility amplitude")
# 		plt.title("Azimuthally averaged visamp")
# 
# 		if self.oifits:
# 			plt.subplot(224)
# 		#	plt.plot(fftscale * np.arange(vis.shape[1]),vis[roll,:])
# 		#	plt.ylim([0,1])
# 		#	plt.xlim([0,130])
# 		#	plt.xlabel("Projected baseline [m]")
# 		#	plt.ylabel("Visibility amplitude")
# 		#	plt.title("Meridional cut through (u,v) plane")
# 			cNorm=c.Normalize(vmin=0,vmax=1.0)
# 			scalarMap=cmx.ScalarMappable(norm=cNorm,cmap="rainbow")
# 			for iu,iv,ivis in zip(self.u,self.v,self.vis_obs):
# 				plt.plot(-iu,iv,color=scalarMap.to_rgba(ivis),mew=0,ms=4,marker="o")
# 				plt.plot(iu,-iv,color=scalarMap.to_rgba(ivis),mew=0,ms=4,marker="o")
# 			plt.xlim([-130,130])
# 			plt.ylim([-130,130])
# 			xyticks=np.array([-100,-50,0,50,100])
# 			plt.xticks(xyticks,-xyticks)
# 			plt.yticks(xyticks,xyticks)
# 			plt.xlabel("u [m]")
# 			plt.ylabel("v [m]")
# 			plt.colorbar(label="Visibility amplitude")
# 			plt.title("Observed (u,v) plane")

		plt.tight_layout()
		plt.suptitle(self.f_model.split(".fits")[0] + str(" (lam={0} m)".format(self.lam)),fontsize=6)

		plt.savefig(self.f_model.split(".fits")[0]+".png")
		plt.clf()