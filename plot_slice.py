from raman_helper import *
from matplotlib.widgets import Button, Slider

# their plotting only plots a band, so I make my own code to plot every spectra

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()

raman_data = Raman_Data(path, 20, 20)

#TODO: make the whole thing presentable
#TODO: improve the plot raman
raman_slice = raman_data.get_slice_pipeline(2)

spectral_data = raman_slice.spectral_data
spectral_axis = raman_slice.spectral_axis

chosen_band = spectral_axis[0]

# TODO: he would want to see the denoised graphs as well. plot an example of one

# Slider using rp lib, and combining with matplotlib
indices = np.arange(len(spectral_axis))

fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.25)

ax_im = rp.plot.image(raman_slice.band(spectral_axis[0]), ax=ax)

im = ax.get_images()[0]

ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])

raman_slider = Slider(
    ax=ax_slider,
    label="Raman Shift ",
    valmin=0,
    valmax=len(indices) - 1,
    valinit=0,
    valstep=1
)

# The function to be called anytime a slider's value changes
def update(val):
    idx = int(raman_slider.val)
    im.set_array(spectral_data[:, :, idx])
    ax.set_title(f"Shift: {spectral_axis[idx]:.2f} cm$^{{-1}}$")
    fig.canvas.draw_idle()

# register the update function with each slider
raman_slider.on_changed(update)
plt.show()


# Slider using matplot lib
"""
indices = np.arange(len(spectral_axis))
index = 0

fig, ax = plt.subplots(figsize=(8,8))
        
im = ax.imshow(spectral_data[:, :, index]) # show all x and y grid spots, and the chosen slice corresponding to index
ax.set_axis_off()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a .. oriented slider to control the amplitude
ax_raman = fig.add_axes([0.25, 0.1, 0.65, 0.03])

raman_slider = Slider(
    ax=ax_raman,
    label="Raman Shift",
    valmin=0,
    valmax=len(indices),
    valinit=0,
    valstep=indices
)
raman_slider.valtext.set_text(spectral_axis[0]) # initial raman shift value

# The function to be called anytime a slider's value changes
def update(val):
    im.set_array(spectral_data[:, :, raman_slider.val])
    raman_slider.valtext.set_text(spectral_axis[val])
    fig.canvas.draw_idle()

# register the update function with each slider
raman_slider.on_changed(update)
plt.show()
"""