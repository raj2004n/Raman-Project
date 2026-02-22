import argparse
from pathlib import Path

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from raman_helper import Raman_Data


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Raman area-range viewer."
    )
    parser.add_argument("path", type=str, help="Path to the data directory.")
    parser.add_argument("x", type=int, help="Number of grid points in x.")
    parser.add_argument("y", type=int, help="Number of grid points in y.")
    parser.add_argument("--pipeline", type=int, default=1, choices=[1, 2, 3],
                        help="Preprocessing pipeline (default: 1).")
    parser.add_argument("--shift-range", type=float, default=20,
                        help="Rolling window width in cm⁻¹ (default: 20).")
    return parser.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path, x, y, pipeline, shift_range):
    raman_data = Raman_Data(Path(path).expanduser(), x, y)
    area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map = (
        raman_data.get_area_range(pipeline, shift_range)
    )
    return raman_data, area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map


# ── Plot setup ────────────────────────────────────────────────────────────────

def build_figure(area_by_region, spectra_by_pixel, raman_shift, idx_step):
    fig, (ax_raman, ax_spectra) = plt.subplots(
        2, 1, figsize=(8, 20), squeeze=True,
        gridspec_kw={"height_ratios": [5, 2]}
    )

    # raman image
    ax_raman.set_title("Raman Image")
    ax_raman.set_axis_off()
    rp.plot.image(area_by_region[:, :, 0], ax=ax_raman)
    image = ax_raman.get_images()[0]

    v_min, v_max = np.min(area_by_region), np.max(area_by_region)
    image.set_clim(v_min, v_max)

    cbar = ax_raman.images[0].colorbar
    cbar.set_ticks(np.linspace(v_min, v_max, 5))

    # spectra plot
    ax_spectra.set_title("Intensity Spectra")
    ax_spectra.set_xlabel(r"Raman Shift cm$^{-1}$")
    ax_spectra.set_ylabel("Intensity")
    (pixel_spectra_line,) = ax_spectra.plot(raman_shift, spectra_by_pixel[1])

    raman_shift_arr = np.array(raman_shift)
    lower_limit_line = ax_spectra.axvline(
        raman_shift_arr[0], color="red", linestyle="--", alpha=0.7
    )
    upper_limit_line = ax_spectra.axvline(
        raman_shift_arr[idx_step], color="red", linestyle="--", alpha=0.7
    )

    # hover label
    hover_text = ax_raman.text(
        0.01, 0.99, "", transform=ax_raman.transAxes,
        va="top", ha="left", color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    # slider and textbox axes
    ax_slider = fig.add_axes([0.15, 0.02, 0.5, 0.04])
    ax_box    = fig.add_axes([0.9,  0.02, 0.07, 0.04])

    indices = np.arange(area_by_region.shape[-1])
    raman_slider = Slider(
        ax=ax_slider, label="Raman Shift",
        valmin=0, valmax=indices[-1],
        valinit=0, valstep=indices,
    )
    raman_slider.valtext.set_text(str(raman_shift[0]))

    text_box = TextBox(ax_box, "Pixel:", textalignment="center")
    text_box.set_val("1")

    return (
        fig, ax_raman, ax_spectra,
        image, pixel_spectra_line,
        lower_limit_line, upper_limit_line,
        hover_text, raman_slider, text_box,
        raman_shift_arr,
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_update(fig, ax_spectra, image, pixel_spectra_line,
                lower_limit_line, upper_limit_line,
                raman_slider, text_box,
                area_by_region, spectra_by_pixel,
                raman_shift_arr, idx_step):

    def update(_val):
        index = int(raman_slider.val)
        try:
            pixel = int(text_box.text)
        except (ValueError, KeyError):
            return

        image.set_data(area_by_region[:, :, index])

        new_y = spectra_by_pixel[pixel]
        pixel_spectra_line.set_ydata(new_y)

        x_start = raman_shift_arr[index]
        x_end   = raman_shift_arr[index + idx_step - 1]
        lower_limit_line.set_xdata([x_start, x_start])
        upper_limit_line.set_xdata([x_end,   x_end])
        raman_slider.valtext.set_text(f"{x_start:.0f}-{x_end:.0f}")

        ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)
        fig.canvas.draw_idle()

    return update


def make_on_hover(fig, ax_raman, hover_text, pixel_map, raman_data):

    def on_hover(event):
        if event.inaxes == ax_raman:
            col = np.clip(int(event.xdata + 0.5), 0, raman_data.y - 1)
            row = np.clip(int(event.ydata + 0.5), 0, raman_data.x - 1)
            hover_text.set_text(f"Pixel {pixel_map[row, col]}")
        else:
            hover_text.set_text("")
        fig.canvas.draw_idle()

    return on_hover


def make_on_click(fig, ax_raman, ax_spectra,
                  pixel_spectra_line, text_box,
                  raman_slider, spectra_by_pixel,
                  pixel_map, raman_data):

    def on_click(event):
        if event.inaxes != ax_raman or not event.dblclick:
            return

        col   = np.clip(int(event.xdata + 0.5), 0, raman_data.y - 1)
        row   = np.clip(int(event.ydata + 0.5), 0, raman_data.x - 1)
        pixel = pixel_map[row, col]

        text_box.set_val(str(pixel)) # also fires on_submit → update()

        new_y = spectra_by_pixel[pixel]
        pixel_spectra_line.set_ydata(new_y)
        ax_spectra.set_ylim(np.min(new_y) * 0.9, np.max(new_y) * 1.1)
        fig.canvas.draw_idle()

    return on_click


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    raman_data, area_by_region, spectra_by_pixel, raman_shift, idx_step, pixel_map = (
        load_data(args.path, args.x, args.y, args.pipeline, args.shift_range)
    )

    (
        fig, ax_raman, ax_spectra,
        image, pixel_spectra_line,
        lower_limit_line, upper_limit_line,
        hover_text, raman_slider, text_box,
        raman_shift_arr,
    ) = build_figure(area_by_region, spectra_by_pixel, raman_shift, idx_step)

    update = make_update(
        fig, ax_spectra, image, pixel_spectra_line,
        lower_limit_line, upper_limit_line,
        raman_slider, text_box,
        area_by_region, spectra_by_pixel,
        raman_shift_arr, idx_step,
    )
    on_hover = make_on_hover(fig, ax_raman, hover_text, pixel_map, raman_data)
    on_click = make_on_click(
        fig, ax_raman, ax_spectra,
        pixel_spectra_line, text_box,
        raman_slider, spectra_by_pixel,
        pixel_map, raman_data,
    )

    text_box.on_submit(update)
    raman_slider.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


if __name__ == "__main__":
    main()