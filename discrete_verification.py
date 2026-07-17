import numpy as np

# model information
wp = -0.001
rp = 4
area = (2*rp)**2
vol_flow = wp*area

# domain information
lx = 128, 128
dx = 2.0, 2.0
x = np.arange(-lx[0]/2+dx[0]/2, lx[0]/2, dx[0])
y = np.arange(-lx[1]/2+dx[1]/2, lx[1]/2, dx[1])

X, Y = np.meshgrid(x, y)
n_gauss = [len(x), len(y)]

# create gaussian 
def gaussian(a, x, y, sigma=20):
    return a*np.exp(-((x**2 + y**2) / (2 * sigma**2)))
gaus = gaussian(1.0, X, Y, sigma=rp)

vol_flow_gaus = np.sum(gaus)*dx[0]*dx[1]

area_scaling = vol_flow/vol_flow_gaus

print(f"Volumetric flow rate of the square with velocity {wp}: {vol_flow} m^3/s")
print(f"Volumetric flow rate of the gaussian with a standard deviation of {rp} and velocity 1.0: {vol_flow_gaus} m^3/s")
print(f"velocity scaling for the volumetric flow rate of the Gaussian with a standard deviation of {rp}: {area_scaling}\n\tvelocity scaling = volumetric flow rate of the square/volumetric flow rate of the gaussian")
