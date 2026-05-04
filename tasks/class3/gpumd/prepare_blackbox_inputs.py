#!/usr/bin/env python3
"""Prepare benchmark-owned standardized inputs for the gpumd black-box task."""

from __future__ import annotations

import ctypes
import json
import math
import os
import struct
from array import array
from dataclasses import dataclass


TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
OUT_DIR = os.path.join(TASK_DIR, ".blackbox_inputs_v1")
WORKLOAD_DIR = os.path.join(OUT_DIR, "workloads")

MAGIC = b"KHGPMD1\x00"
VERSION = 1
RAND_MAX = 2147483647
HEADER_FORMAT = "<8sIIII7d16d"
SYSTEM_KIND = {"lj": 1, "tersoff": 2, "coulomb": 3}

libc = ctypes.CDLL(None)
libc.srand.argtypes = [ctypes.c_uint]
libc.rand.restype = ctypes.c_int


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    filename: str
    ref_filename: str
    system: str
    steps: int
    dt: float
    mass: float
    energy_drift_tol: float
    force_rel_tol: float
    params: tuple[float, ...]


WORKLOADS = [
    WorkloadSpec(
        "LJ_Ar_256K",
        "Ar_256000.xyz",
        "lj_Ar_256000.bin",
        "lj",
        100,
        0.001,
        39.948,
        0.01,
        1e-4,
        (0.0104, 3.4, 8.5),
    ),
    WorkloadSpec(
        "LJ_Ar_500K",
        "Ar_500000.xyz",
        "lj_Ar_500000.bin",
        "lj",
        50,
        0.001,
        39.948,
        0.01,
        1e-4,
        (0.0104, 3.4, 8.5),
    ),
    WorkloadSpec(
        "Tersoff_Si_27K",
        "Si_27000.xyz",
        "tersoff_Si_27000.bin",
        "tersoff",
        30,
        0.0005,
        28.0855,
        0.01,
        1e-4,
        (
            1830.8,
            471.18,
            2.4799,
            1.7322,
            1.1e-6,
            0.78734,
            1.0039e5,
            16.217,
            -0.59825,
            2.7,
            3.0,
            3.0,
            0.0,
            1.0,
        ),
    ),
    WorkloadSpec(
        "Tersoff_Si_64K",
        "Si_64000.xyz",
        "tersoff_Si_64000.bin",
        "tersoff",
        20,
        0.0005,
        28.0855,
        0.01,
        1e-4,
        (
            1830.8,
            471.18,
            2.4799,
            1.7322,
            1.1e-6,
            0.78734,
            1.0039e5,
            16.217,
            -0.59825,
            2.7,
            3.0,
            3.0,
            0.0,
            1.0,
        ),
    ),
    WorkloadSpec(
        "Coulomb_NaCl_8K",
        "NaCl_8000.xyz",
        "coulomb_NaCl_8000.bin",
        "coulomb",
        30,
        0.001,
        29.22,
        0.01,
        1e-2,
        (0.3, 7.0),
    ),
    WorkloadSpec(
        "Coulomb_NaCl_32K",
        "NaCl_32768.xyz",
        "coulomb_NaCl_32768.bin",
        "coulomb",
        20,
        0.001,
        29.22,
        0.01,
        1e-2,
        (0.3, 7.0),
    ),
]


def _c_rand_float() -> float:
    return libc.rand() / RAND_MAX


def _parse_xyz(path: str):
    with open(path) as f:
        n = int(f.readline().strip())
        header = f.readline().strip()
        lattice_key = 'Lattice="'
        start = header.find(lattice_key)
        if start < 0:
            raise ValueError(f"missing Lattice in {path}")
        start += len(lattice_key)
        end = header.find('"', start)
        lat = [float(x) for x in header[start:end].split()]
        box = (lat[0], lat[4], lat[8])

        x = array("d")
        y = array("d")
        z = array("d")
        atom_type = array("i")
        charge = array("f")
        species = []

        type_map = {"Ar": 0, "Si": 0, "Na": 0, "Cl": 1}
        charge_map = {"Na": 1.0, "Cl": -1.0}

        for _ in range(n):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"invalid atom line in {path}")
            sp = parts[0]
            species.append(sp)
            x.append(float(parts[1]))
            y.append(float(parts[2]))
            z.append(float(parts[3]))
            atom_type.append(type_map.get(sp, 0))
            charge.append(charge_map.get(sp, 0.0))

    return n, box, x, y, z, atom_type, charge, species


def _apply_deterministic_perturbation(n: int, box: tuple[float, float, float], x, y, z) -> None:
    min_l = min(box)
    perturb = 0.01 * min_l / (n ** (1.0 / 3.0))
    libc.srand(54321 + n)
    lx, ly, lz = box
    for i in range(n):
        x[i] += perturb * (_c_rand_float() - 0.5)
        y[i] += perturb * (_c_rand_float() - 0.5)
        z[i] += perturb * (_c_rand_float() - 0.5)

        if x[i] < 0.0:
            x[i] += lx
        elif x[i] >= lx:
            x[i] -= lx
        if y[i] < 0.0:
            y[i] += ly
        elif y[i] >= ly:
            y[i] -= ly
        if z[i] < 0.0:
            z[i] += lz
        elif z[i] >= lz:
            z[i] -= lz


def _init_velocities(n: int, temperature: float, mass: float):
    vx = array("d", [0.0]) * n
    vy = array("d", [0.0]) * n
    vz = array("d", [0.0]) * n
    libc.srand(42)
    scale = math.sqrt(temperature / mass)
    sum_vx = 0.0
    sum_vy = 0.0
    sum_vz = 0.0

    for i in range(n):
        u1 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        u2 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        u3 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        u4 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        u5 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        u6 = (libc.rand() + 1.0) / (RAND_MAX + 2.0)
        vx_i = scale * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        vy_i = scale * math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)
        vz_i = scale * math.sqrt(-2.0 * math.log(u5)) * math.cos(2.0 * math.pi * u6)
        vx[i] = vx_i
        vy[i] = vy_i
        vz[i] = vz_i
        sum_vx += vx_i
        sum_vy += vy_i
        sum_vz += vz_i

    sum_vx /= n
    sum_vy /= n
    sum_vz /= n
    for i in range(n):
        vx[i] -= sum_vx
        vy[i] -= sum_vy
        vz[i] -= sum_vz
    return vx, vy, vz


def _temperature_for(spec: WorkloadSpec) -> float:
    if spec.system == "lj":
        return 0.5
    if spec.system == "tersoff":
        return 0.1
    return 0.05


def _pack_params(spec: WorkloadSpec) -> list[float]:
    params = list(spec.params)
    if len(params) > 16:
        raise ValueError(f"too many params for {spec.name}")
    params.extend([0.0] * (16 - len(params)))
    return params


def _write_workload(spec: WorkloadSpec) -> dict:
    src = os.path.join(DATA_DIR, spec.filename)
    n, box, x, y, z, atom_type, charge, _species = _parse_xyz(src)
    _apply_deterministic_perturbation(n, box, x, y, z)
    vx, vy, vz = _init_velocities(n, _temperature_for(spec), spec.mass)

    out_path = os.path.join(WORKLOAD_DIR, f"{spec.name}.bin")
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        SYSTEM_KIND[spec.system],
        n,
        spec.steps,
        spec.dt,
        spec.mass,
        spec.energy_drift_tol,
        spec.force_rel_tol,
        box[0],
        box[1],
        box[2],
        *_pack_params(spec),
    )

    with open(out_path, "wb") as f:
        f.write(header)
        x.tofile(f)
        y.tofile(f)
        z.tofile(f)
        vx.tofile(f)
        vy.tofile(f)
        vz.tofile(f)
        atom_type.tofile(f)
        charge.tofile(f)

    return {
        "name": spec.name,
        "system": spec.system,
        "file": os.path.basename(out_path),
        "ref_file": spec.ref_filename,
        "atom_count": n,
        "steps": spec.steps,
        "energy_drift_tol": spec.energy_drift_tol,
        "force_rel_tol": spec.force_rel_tol,
    }


def prepare() -> str:
    os.makedirs(WORKLOAD_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "workloads.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            workloads = manifest.get("workloads", [])
            if (
                manifest.get("version") == VERSION
                and manifest.get("format") == "kh_gpumd_workload_v1"
                and workloads
                and all("ref_file" in item for item in workloads)
                and all(
                    os.path.isfile(os.path.join(WORKLOAD_DIR, item.get("file", "")))
                    for item in workloads
                )
            ):
                return OUT_DIR
        except Exception:
            pass

    manifest = {
        "version": VERSION,
        "format": "kh_gpumd_workload_v1",
        "workloads": [],
    }
    for spec in WORKLOADS:
        manifest["workloads"].append(_write_workload(spec))

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return OUT_DIR


if __name__ == "__main__":
    print(prepare())
