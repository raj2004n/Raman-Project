from raman_helper import *
from matplotlib.widgets import Slider, TextBox

#path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()
#raman_data = Raman_Data(path, 20, 20)

path = Path("~/Code/Data_SH/dataTwo").expanduser()
raman_data = Raman_Data(path, 30, 35)

area_by_region, shift_by_region, spectra_by_region, original_spectra_by_region = raman_data.get_area_regions(pipeline=2, regions=20)

# find peak of intensity for plotting
peak = 0
depth = np.inf
for p in range(1, 401):
    for r in range(0, 10):
        peak = max(np.max(spectra_by_region[p, r]), peak)
        depth = min(np.min(spectra_by_region[p, r]), depth)

# pad peak and depth for better y limit
peak *= 1.1
depth *= 0.9

def update(val):
    # index of spectra region
    index = int(raman_slider.val) 
    try:
        pixel = int(text_box.text)
    except ValueError:
        return # ignore
    
    # update image
    image.set_array(area_by_region[:, :, index])

    # update spectra region and pixel
    pixel_spectra[0].set_data(shift_by_region[index][1], spectra_by_region[pixel, index])
    #pixel_spectra[0].set_data(shift_by_region[index][1], original_spectra_by_region[pixel, index])

    # show region range instead of index
    raman_slider.valtext.set_text(shift_by_region[index][0])
    ax_spectra.relim()
    ax_spectra.autoscale_view(tight=True, scaley=False, scalex=True)
    ax_spectra.set_ylim(depth, peak)
    fig.canvas.draw_idle()

indices = np.arange(area_by_region.shape[-1])

fig, (ax_raman, ax_spectra) = plt.subplots(2, 1, figsize=(8, 20), squeeze=True, gridspec_kw={'height_ratios': [5, 2]})

# set up axes
ax_raman.set_title("Raman Image")
ax_raman.set_axis_off()

ax_spectra.set_title("Intensity Spectra")
ax_spectra.set_xlabel(f"Raman Shift cm$^{{-1}}$")
ax_spectra.set_ylabel(f"Intensity")
ax_spectra.set_ylim(depth, peak)

# axes for slider and textbox
ax_slider = fig.add_axes([0.15, 0.02, 0.5, 0.04])
ax_box = fig.add_axes([0.9, 0.02, 0.07, 0.04])

# raman image and pixel spectra
image = ax_raman.imshow(area_by_region[:, :, 0]) # show all x and y grid spots, and the chosen slice corresponding to index
pixel_spectra = ax_spectra.plot(shift_by_region[0][1], spectra_by_region[1, 0])
#pixel_spectra = ax_spectra.plot(shift_by_region[0][1], original_spectra_by_region[1, 0])

# text box to enter pixel
text_box = TextBox(ax_box, "Pixel:", textalignment="center")

# slider to choose region
raman_slider = Slider(
    ax=ax_slider,
    label="Raman Shift",
    valmin=0,
    valmax=indices[-1],
    valinit=0,
    valstep=indices
)

# intial values for slider text and text box
raman_slider.valtext.set_text(shift_by_region[0][0]) # initial raman shift value
text_box.set_val('1')

text_box.on_submit(update)
raman_slider.on_changed(update)

plt.show()