import os
import numpy as np
import h5py

folder = '/Users/annapauls/Documents/Github repositories/3d_langmuir_gpu/localoutputs/sponge testing/gaussian/width = 10.0, rate = 0.0002777777777777778'
file_name = 'fields.jld2'
nx = np.array([48, 48, 48])
lx = np.array([320.0, 320.0, 96.0])
dx = lx / nx

xf = np.arange(0.0, lx[0], dx[0])
yf = np.arange(0.0, lx[1], dx[1])
zf = np.arange(0.0, lx[2], dx[2])
x = xf - 0.5*dx[0]
y = yf - 0.5*dx[1]
z = zf - 0.5*dx[2]

file = h5py.File(os.path.join(folder, file_name), 'a')
file.create_group('grid')
file.create_dataset('grid/Nx', data=nx[0])
file.create_dataset('grid/Ny', data=nx[1])
file.create_dataset('grid/Nz', data=nx[2])

file.create_dataset('grid/Lx', data=lx[0])
file.create_dataset('grid/Ly', data=lx[1])
file.create_dataset('grid/Lz', data=lx[2])

file.create_dataset('grid/Hx', data=0)
file.create_dataset('grid/Hy', data=0)
file.create_dataset('grid/Hz', data=0)

file.create_dataset('grid/Δxᶜᵃᵃ', data=dx[0])
file.create_dataset('grid/Δyᵃᶜᵃ', data=dx[1])
file.create_dataset('grid/z/Δᵃᵃᶜ', data=dx[2])

file.create_dataset('grid/xᶜᵃᵃ', data=x)
file.create_dataset('grid/yᵃᶜᵃ', data=y)
file.create_dataset('grid/z/cᵃᵃᶜ', data=z)
file.create_dataset('grid/z/cᵃᵃᶠ', data=zf)
file.close()
