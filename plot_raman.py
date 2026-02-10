from raman_helper import *
from matplotlib.widgets import Button, Slider
import argparse
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=""
        )
    
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the folder (include folder name)."
        )
    
    parser.add_argument(
        "-x",
        "--x",
        type=int,
        required=True,
        help=""
        )
    
    parser.add_argument(
        "-y",
        "--y",
        type=int,
        required=True,
        help=""
        )
    
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=['sliding_spectra', 'whole_spectra'],
        required=True,
        help=""
        )
    
    parser.add_argument(
        "-s",
        "--slices",
        type=int,
        required=False,
        help=""
        )

    return parser.parse_args()

def plot(path, x, y, mode, slices):

    raman_visual = Raman_Data(path, x, y)
    # intensity slice includes entire spectra range for each pixel
    # integral slice includes area under entire specra in chunks for each pixel
    # whole_integrals is the total area under the curve
    # raman shifts is the corresponding x axis to both
    intensiy_slice, integral_slice, whole_integrals, raman_shifts = raman_visual.get_integrals(slices)

    if mode == 'whole_spectra':
        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.imshow(whole_integrals)
        fig.colorbar(im, ax=ax)   
        plt.show()
    elif mode == 'sliding_spectra':
        """
        # Define initial parameters
        indices = np.arange(len(raman_shifts) // slices)
        index = 0

        raman_slice = []
        i = 0
        j = 0
        while j < (len(raman_shifts) // slices):
            raman_slice.append(f"{raman_shifts[i]}-{raman_shifts[i + slices-1]}")
            i += slices
            j += 1

        fig, ax = plt.subplots(figsize=(8,8))
        
        im = ax.imshow(integral_slice[:, :, index]) # show all x and y grid spots, and the chosen slice corresponding to index
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
        raman_slider.valtext.set_text(raman_slice[0]) # initial raman shift value

        # The function to be called anytime a slider's value changes
        def update(val):
            im.set_array(intensiy_slice[:, :, raman_slider.val])
            raman_slider.valtext.set_text(raman_slice[val])
            fig.canvas.draw_idle()

        # register the update function with each slider
        raman_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            raman_slider.reset()

        button.on_clicked(reset)

        plt.show()
        
        """
        indices = np.arange(len(raman_shifts) // slices)
        index = 0

        raman_slice = []
        i = 0
        j = 0
        while j < (len(raman_shifts) // slices):
            raman_slice.append(f"{raman_shifts[i]}-{raman_shifts[i + slices-1]}")
            i += slices
            j += 1

        fig, ax = plt.subplots(figsize=(8,8))
        l, = ax.plot(integral_slice[0,0], raman_slice)
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

        # The function to be called anytime a slider's value changes
        def update(val):
            l.set_ydata(intensiy_slice[0, 0, raman_slider.val])
            raman_slider.valtext.set_text(raman_slice[val])
            fig.canvas.draw_idle()

        # register the update function with each slider
        raman_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            raman_slider.reset()

        button.on_clicked(reset)

        plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    plot(
        args.path,
        args.x,
        args.y,
        args.mode,
        args.slices
        )