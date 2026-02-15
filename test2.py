from raman_helper import *
from matplotlib.widgets import Button, Slider

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()
raman_data = Raman_Data(path, 20, 20)

area_by_region, shift_by_region = raman_data.get_area_regions(1, 10)

print(shift_by_region.shape)
indices = np.arange(area_by_region.shape[-1])

fig, ax = plt.subplots(2)

image = ax[0].imshow(area_by_region[:, :, 0]) # show all x and y grid spots, and the chosen slice corresponding to index
#pixel_spectra = ax[1].plot(shift_by_region[0][1], area_by_region[5, 5, 0])

fig.subplots_adjust(left=0.25, bottom=0.25)

ax_raman = fig.add_axes([0.25, 0.1, 0.5, 0.04])

raman_slider = Slider(
    ax=ax_raman,
    label="Raman Shift",
    valmin=0,
    valmax=indices[-1],
    valinit=0,
    valstep=indices
)
raman_slider.valtext.set_text(shift_by_region[0]) # initial raman shift value

def update(val):
    image.set_array(area_by_region[:, :, raman_slider.val])
    #pixel_spectra.set_data(shift_by_region[5, 5, raman_slider.val], area_by_region[5, 5, raman_slider.val])
    raman_slider.valtext.set_text(shift_by_region[val, 0])
    fig.canvas.draw_idle()

raman_slider.on_changed(update)
plt.show()
