from img2vis import img2vis

m="../models/Bernd_2016-05-17/n1068_bild_10_45L1.00000.fits"
pxscale_mas=0.04*1000/71
lam=1e-5
oifits="../MIDI_data/NGC1068_lopez-gonzaga2014.oifits"
phot=10.0
i=img2vis(m, pxscale_mas, lam, oifits=oifits, phot=phot)
i.optimize_pa()
i.make_plot()