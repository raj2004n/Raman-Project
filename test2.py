from raman_helper import *
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

path = Path("~/Code/Data_SH/data").expanduser()
raman_data = Raman_Data(path, 5, 25)
integrals, raman_shifts = raman_data.get_area(1)

plt.imshow(integrals)
plt.show()