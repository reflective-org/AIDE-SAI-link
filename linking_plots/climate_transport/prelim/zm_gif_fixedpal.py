import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from plotlib import latp, fig_to_img, date_of_day
d = np.load('data/aoa_run.npz'); lat, p = d['lat'], d['plev'] / 100
zm = d['inj_zm'] * 1e9; cells = d['inj_cells']
hi = float(10 ** np.ceil(np.log10(np.percentile(zm[-1], 99.99)))); lo = hi * 1e-4
fr = []
for i in range(0, zm.shape[0], 2):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    latp(ax, lat, p, np.clip(zm[i], lo, None), 'Blues', None, None, 'mass mixing ratio [ppbm = 1e-9 kg/kg]', fig, norm=LogNorm(lo, hi))
    ax.plot(cells[:, 1], cells[:, 0], 'x', color='#cb181d', ms=7, mew=1.6)
    ax.set_title(f'Injection tracer, zonal-mean mixing ratio [ppbm], {date_of_day(i+1)} (day {i+1})', fontsize=9)
    fig.tight_layout(); fr.append(fig_to_img(fig, 75)); plt.close(fig)
# one shared palette for every frame, so a colour means the same value throughout
w, h = fr[0].size; strip = Image.new('RGB', (w, h * 6))
for k, j in enumerate(np.linspace(0, len(fr) - 1, 6).astype(int)): strip.paste(fr[j], (0, h * k))
pal = strip.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in fr]
q[0].save('inj_zonal_mean_fixedpal.gif', save_all=True, append_images=q[1:], duration=130, loop=0)
print(lo, hi, len(q))
