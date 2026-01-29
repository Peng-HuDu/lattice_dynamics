#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function: collect static energies of phonon-modulated structures.

Author: Peng-Hu Du, penghdu@foxmail.com, ref: [https://zhuanlan.zhihu.com/p/703248206]
Date: 2026.01.09
"""


import argparse
import os
import re
import subprocess
from typing import Optional
import numpy as np


def get_num_atoms_vasp(outcar_path):
    """
    从VASP OUTCAR文件中获取原子数
    
    Args:
        outcar_path (str): OUTCAR文件所在路径
        
    Returns:
        natoms (int): 原子数
    """
    try:
        res = subprocess.run(
            r'grep -n -s "position of ions in cartesian coordinates" ' + outcar_path,
            shell=True, capture_output=True, text=True)
        if not res.stdout:
            return None
        begin_id = int(res.stdout.split(":")[0])
        natoms = 0
        with open(outcar_path) as f:
            for i, line in enumerate(f, 1):
                if i > begin_id:
                    coords = re.findall(r"[-+]?\d+\.\d+", line)
                    if len(coords) == 3:
                        natoms += 1
                    else:
                        break
        return natoms or None
    except Exception:
        return None
    
def get_energy_vasp(outcar_path):
    """
    从VASP OUTCAR文件中获取电子自洽最后一步的能量
    
    Args:
        outcar_path (str): OUTCAR文件所在路径
        
    Returns:
        energy (float): 总能量
    """
    try:
        res = subprocess.run(
            r'grep -s "free  energy   TOTEN" ' + outcar_path + " | tail -1 | awk '{print $5}'",
            shell=True, capture_output=True, text=True)
        if res.stdout.strip():
            return float(res.stdout.strip())
        else:
            return None
    except Exception:
        return None  


def get_num_atoms_qe(scf_path):
    """
    从QE scf.out文件中获取原子数
    
    Args:
        scf_path (str): scf.out文件所在路径
        
    Returns:
        natoms (int): 原子数
    """
    try:
        with open(scf_path) as f:
            for line in f:
                if "number of atoms/cell" in line:
                    # 例如：  number of atoms/cell =  20
                    m = re.search(r"=\s*([0-9]+)", line)
                    if m:
                        return int(m.group(1))
        return None
    except Exception:
        return None
    
def get_energy_qe(scf_path):
    """
    从QE scf.out文件中获取电子自洽最后一步的能量
    
    Args:
        scf_path (str): scf.out文件所在路径
        
    Returns:
        energy (float): 总能量
    """
    try:
        # grep, 取最后一行
        res = subprocess.run(
            r'grep -s "!    total energy" ' + scf_path + " | tail -1 | awk '{print $5}'",
            shell=True, capture_output=True, text=True,
        )
        en_ry = res.stdout.strip()
        return float(en_ry) * 13.6056980659 if en_ry else None  # Ry→eV
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect static energies from MPOSCAR_* SCF directories.")
    parser.add_argument('--amplitudes', nargs=3, type=int, required=True,
                        help='start stop num')
    parser.add_argument("--code", type=str, choices=['vasp', 'qe', 'auto'], default='auto',
                        help='which code to parse (default auto)')
    args = parser.parse_args()
    
    print("\nCollecting static energies from MPOSCAR_* SCF directories...")
    amps = np.linspace(start=args.amplitudes[0], 
                       stop=args.amplitudes[1],
                       num=args.amplitudes[2]) 
    print(f"{'Amplitude':>10}  {'Natoms':>6}  {'Total energy (eV/atom)':>18}  {'Source':>8}")
    
    for amp in amps:
        d = f'MPOSCAR_{amp:0.2f}'
        outcar  = os.path.join(d, 'OUTCAR')
        scfout  = os.path.join(d, 'scf.out')
        
        energy, natoms, src = 1e14, 1, 'None'  # 默认失败
        
        if args.code in ('vasp', 'auto') and os.path.isfile(outcar):
            e = get_energy_vasp(outcar)
            n = get_num_atoms_vasp(outcar)
            if e is not None:
                energy, src = e, 'vasp'
            if n is not None:
                natoms = n

        if args.code in ('qe', 'auto') and src == 'None' and os.path.isfile(scfout):
            e = get_energy_qe(scfout)
            n = get_num_atoms_qe(scfout)
            if e is not None:
                energy, src = e, 'qe'
            if n is not None:
                natoms = n
        
        print(f"{amp:10.2f}  {natoms:6d}  {energy/natoms:18.8f}  {src:>8}")  
