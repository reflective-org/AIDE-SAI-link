import numpy as np, time
from common import *
s = 100
z = np.load(f'{RUNS}/r2_tomas_fast/frames3d/step_{s:04d}.npz')
num = z['num']; dref = z['deff_wet_nm']; dpb = z['dp_bin_nm']
mmid = (dpb*1e-9)**3*np.pi/6*RHO_AER
c = np.load(CLIM)
meta_ = meta(); klevs = meta_['klevs']
t = time.time(); RH = c['RELHUM'][:, klevs]; print('rh read', time.time()-t, RH.shape)
m0, m1, w = month_weights(H0 + 6*s, c['month_centre_doy'])
rh = np.clip(((1-w)*RH[m0] + w*RH[m1])/100., 0, 1)
T = np.load(f'{MET}/T.npy', mmap_mode='r')[s]
wt = h2so4_equilibrium_wt(T, rh*100)
d3 = wet_d3(mmid[:, None, None, None], wt[None])
d2 = d3**(2/3)
de = (num*d3).sum(0)/np.maximum((num*d2).sum(0), 1e-300)*1e9
r = de/dref
print('ratio pct', np.nanpercentile(r, [0.1, 1, 50, 99, 99.9]))
# tropopause
Tf = np.load(f'{MET}/T_full.npy', mmap_mode='r')[s+1]; ps = np.load(f'{MET}/ps.npy', mmap_mode='r')[s+1]
kr = meta_['krad']
pf = c['hyam'][kr][:, None, None]*c['P0'] + c['hybm'][kr][:, None, None]*ps[None]
t=time.time(); pt = wmo_tropopause(Tf.reshape(len(kr), -1), pf.reshape(len(kr), -1)).reshape(192, 288); print('trop', time.time()-t)
print('nan frac', np.isnan(pt).mean())
lat = z['lat']
for a,b in [(-10,10),(30,40),(50,60),(70,90),(-90,-70)]:
    sel=(lat>=a)&(lat<b); print(a,b, np.nanmedian(pt[sel])/100)
plev = z['plev_pa']
strat = plev[:, None, None] < np.where(np.isnan(pt), np.inf, pt)[None]
print('excluded frac', 1-strat.mean())
