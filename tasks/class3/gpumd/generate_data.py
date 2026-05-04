#!/usr/bin/env python3
"""Generate model.xyz structure files for the GPUMD benchmark."""
import numpy as np, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def write_xyz(filename, species, positions, L):
    N = len(species)
    lat = f"{L:.8f} 0.00000000 0.00000000 0.00000000 {L:.8f} 0.00000000 0.00000000 0.00000000 {L:.8f}"
    with open(filename, 'w') as f:
        f.write(f"{N}\nLattice=\"{lat}\" Properties=species:S:1:pos:R:3\n")
        for i in range(N):
            f.write(f"{species[i]} {positions[i,0]:.8f} {positions[i,1]:.8f} {positions[i,2]:.8f}\n")
    print(f"  {os.path.basename(filename)}: {N} atoms")

def gen_fcc(nx, a):
    basis = np.array([[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    return np.array([(np.array([ix,iy,iz])+b)*a for ix in range(nx) for iy in range(nx) for iz in range(nx) for b in basis])

def gen_diamond(nx, a):
    basis = np.array([[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5],[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75]])
    return np.array([(np.array([ix,iy,iz])+b)*a for ix in range(nx) for iy in range(nx) for iz in range(nx) for b in basis])

def gen_nacl(nx, a):
    bna = np.array([[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]])
    bcl = np.array([[.5,0,0],[0,.5,0],[0,0,.5],[.5,.5,.5]])
    pos, sp = [], []
    for ix in range(nx):
        for iy in range(nx):
            for iz in range(nx):
                for b in bna: pos.append((np.array([ix,iy,iz])+b)*a); sp.append("Na")
                for b in bcl: pos.append((np.array([ix,iy,iz])+b)*a); sp.append("Cl")
    return np.array(pos), sp

configs = [
    ("Ar_256000.xyz",  "Ar",  40, 5.26, gen_fcc),
    ("Ar_500000.xyz",  "Ar",  50, 5.26, gen_fcc),
    ("Si_27000.xyz",   "Si",  15, 5.43, gen_diamond),
    ("Si_64000.xyz",   "Si",  20, 5.43, gen_diamond),
]

print("Generating structure files...")
for fname, elem, nx, a, gen in configs:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        print(f"  {fname}: exists, skipping")
        continue
    pos = gen(nx, a)
    write_xyz(path, [elem]*len(pos), pos, nx*a)

# NaCl (special: two species)
for nx, fname in [(10, "NaCl_8000.xyz"), (16, "NaCl_32768.xyz")]:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        print(f"  {fname}: exists, skipping")
        continue
    pos, sp = gen_nacl(nx, 5.64)
    write_xyz(path, sp, pos, nx*5.64)

print("Done.")
