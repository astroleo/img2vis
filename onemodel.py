from img2vis import img2vis
pxscale_circ = 0.01333 * 1000/20 ### 0.08 pc per px, 1000 mas = 20 pc in Circinus
oifitscirc="../MIDI_data/Circinus_clean.oifits"
i=img2vis("../models/Bernd_2016-05-13/circinus_bild_10_25L0.111111.fits", pxscale_circ, 1e-5, oifits=oifitscirc)
i.optimize_pa()
i.make_plot()