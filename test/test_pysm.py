import numpy as np
import healpy as hp
from lognormal_mocks import *
import matplotlib.pyplot as plt

# Your masks
mask_SO_file = '/home/spk24f/masks/LAT_hits_based_n2048_3deg_apo.fits'      # SO hit-count based mask
mask_galactic_file = '/home/spk24f/masks/mask_binary_gal_cut_n2048_3deg_apo.fits'  # galactic plane mask (e.g., |b| > 20°)
map_d10_93 = '/home/spk24f/gitreps/lenspyx/examples/partsky_lensing_anoise_fg_noNILC/map_hp_93_d10.fits'
nside = 2048
freq = 93
pysm_str = ['d10']
lmax = 3000

#GET MASKS
mask_SO = hp.read_map(mask_SO_file)
mask_galactic = hp.read_map(mask_galactic_file)
mask_total = mask_SO * mask_galactic

npix = mask_total.size
pmap = 4*np.pi / npix
w2 = np.sum((mask_total**2)*pmap) /np.pi / 4.
w4 = np.sum((mask_total**4)*pmap) /np.pi / 4.
w3 = np.sum((mask_total**3)*pmap) /np.pi / 4.

dust_pysm = hp.read_map(map_d10_93)

# Measure power spectrum ONLY in your unmasked region
cl_dust = hp.anafast(dust_pysm, lmax=lmax) 

# Measure mean in unmasked region
mean_dust_masked = np.mean(dust_pysm) #/ np.sum(mask_total)
rho_dust = np.array([mean_dust_masked])


# --- Option B: Use Planck measurements for specific sky fractions ---
# Planck papers quote dust amplitudes for different fsky values
# e.g., at 150 GHz for fsky ~ 0.4:
# A_dust ~ X uK^2 at ell=80 with alpha ~ 2.42

Cl = np.array([[cl_dust]])
# --- Option B: Use Planck measurements for specific sky fractions ---
# Planck papers quote dust amplitudes for different fsky values
# e.g., at 150 GHz for fsky ~ 0.4:
# A_dust ~ X uK^2 at ell=80 with alpha ~ 2.42

mean_g, cl_g = lognormal_mocks_stats(rho_dust, Cl,15000) #actually instead of nside should be accuracy parameter

print("mean_g",mean_g)
print("mean_d",mean_dust_masked)

mgauss = hp.synfast(cl_g[0,0],nside)
m = np.exp(mean_g[0] + mgauss)  # this is the lognormal map

cl_lognormal = hp.anafast(m,lmax=lmax)

ell = np.arange(len(cl_dust)) 

plt.loglog(ell,cl_dust,label='pysm cl')
plt.loglog(ell,cl_lognormal,label='lognormal cl')
plt.legend()
plt.xlabel("l")
plt.ylabel("cl")
plt.savefig("compare_pysm_lognormal")
