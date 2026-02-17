from raman_helper import *
from matplotlib.widgets import Slider, TextBox

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()
raman_data = Raman_Data(path, 20, 20)

area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map = raman_data.get_area_range(1, 20)

fig, (ax_raman, ax_spectra) = plt.subplots(2, 1, figsize=(8, 20), squeeze=True, gridspec_kw={'height_ratios': [5, 2]})

# set up axes
ax_raman.set_title("Raman Image")
ax_raman.set_axis_off()

ax_spectra.set_title("Intensity Spectra")
ax_spectra.set_xlabel(f"Raman Shift cm$^{{-1}}$")
ax_spectra.set_ylabel(f"Intensity")

# axes for slider and textbox
ax_slider = fig.add_axes([0.15, 0.02, 0.5, 0.04])
ax_box = fig.add_axes([0.9, 0.02, 0.07, 0.04])

# raman image and pixel spectra
rp.plot.image(area_by_region[:, :, 0], ax=ax_raman)
image = ax_raman.get_images()[0]
pixel_spectra = ax_spectra.plot(raman_shift, spectra_by_pixel[1])

v_min, v_max = np.min(area_by_region), np.max(area_by_region)
image.set_clim(v_min, v_max)
raman_shift_arr = np.array(raman_shift)

cbar = ax_raman.images[0].colorbar
cbar.set_ticks(np.linspace(v_min, v_max, 5))

def update(val):
    # index of spectra region
    index = int(raman_slider.val) 
    try:
        pixel = int(text_box.text)
    except (ValueError, KeyError):
        return 
    
    # update image
    image.set_data(area_by_region[:, :, index])

    # update spectra region and pixel
    new_y = spectra_by_pixel[pixel]
    pixel_spectra[0].set_ydata(new_y)
    
    x_start = raman_shift_arr[index]
    x_end = raman_shift_arr[index + idx_step - 1]
    
    # update the position of the vertical lines
    lower_limit_line.set_xdata([x_start, x_start])
    upper_limit_line.set_xdata([x_end, x_end])

    raman_slider.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

    ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)

    fig.canvas.draw_idle()

indices = np.arange(area_by_region.shape[-1])

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

# vertical lines on the spectra graph to indicate range being viewed on image
lower_limit_line = ax_spectra.axvline(raman_shift[0], color='red', linestyle='--', alpha=0.7)
upper_limit_line = ax_spectra.axvline(raman_shift[idx_step], color='red', linestyle='--', alpha=0.7)

# intial values for slider text and text box
raman_slider.valtext.set_text(raman_shift[0]) # initial raman shift value
text_box.set_val('1')

text_box.on_submit(update)
raman_slider.on_changed(update)

hover_text = ax_raman.text(0.01, 0.99, '', transform=ax_raman.transAxes,
    va='top', ha='left', color='white', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

def on_hover(event):
    if event.inaxes == ax_raman:
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        col = np.clip(col, 0, raman_data.y - 1)
        row = np.clip(row, 0, raman_data.x - 1)
        pixel = pixel_map[row, col]
        hover_text.set_text(f"Pixel {pixel}")
        fig.canvas.draw_idle()
    else:
        hover_text.set_text('')
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)

def on_click(event):
    if event.inaxes == ax_raman and event.dblclick:
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        col = np.clip(col, 0, raman_data.y - 1)
        row = np.clip(row, 0, raman_data.x - 1)
        pixel = pixel_map[row, col]

        # update textbox
        text_box.set_val(str(pixel))

        # update spectra
        index = int(raman_slider.val)
        new_y = spectra_by_pixel[pixel]
        pixel_spectra[0].set_ydata(new_y)
        ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)

        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()