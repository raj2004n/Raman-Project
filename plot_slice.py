from raman_helper import *
from matplotlib.widgets import Button, Slider

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()
raman_data = Raman_Data(path, 20, 20)

#path = Path("~/Code/Data_SH/dataTwo").expanduser()
#raman_data = Raman_Data(path, 30, 35)

raman_slice = raman_data.get_slices(2)

spectral_data = raman_slice.spectral_data
spectral_axis = raman_slice.spectral_axis

chosen_band = spectral_axis[0]

indices = np.arange(len(spectral_axis))

fig, ax_raman = plt.subplots(figsize=(8,8))

# set up axes
ax_raman.set_title("Raman Image")
ax_raman.set_axis_off()

ax_im = rp.plot.image(raman_slice.band(spectral_axis[0]), ax=ax_raman)

im = ax_raman.get_images()[0]

ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])

raman_slider = Slider(
    ax=ax_slider,
    label="Raman Shift ",
    valmin=0,
    valmax=len(indices) - 1,
    valinit=0,
    valstep=1
)

def update(val):
    idx = int(raman_slider.val)
    im.set_array(spectral_data[:, :, idx])
    ax_raman.set_title(f"Shift: {spectral_axis[idx]:.2f} cm$^{{-1}}$")
    fig.canvas.draw_idle()

raman_slider.on_changed(update)
plt.show()