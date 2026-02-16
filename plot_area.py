from raman_helper import *

path = Path("~/Code/Data_SH/FullCavity_20x20_2umsteps").expanduser()
raman_data = Raman_Data(path, 20, 20)

integrals, raman_shifts = raman_data.get_area(2)
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(integrals)
fig.colorbar(im, ax=ax)   
plt.show()