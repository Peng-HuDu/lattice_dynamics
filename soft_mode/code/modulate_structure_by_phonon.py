#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function: modulate one phonon mode (soft mode, imaginary mode)

Author: Peng-Hu Du, penghdu@foxmail.com, ref: [https://zhuanlan.zhihu.com/p/703248206]
Date: 2026.01.09
"""


from __future__ import annotations

from pathlib import Path
from typing import Sequence

import os
import argparse
import numpy as np

import phonopy
from phonopy.interface.vasp import read_vasp, write_vasp
from phonopy.interface.qe import PH_Q2R, read_pwscf
from phonopy.interface.calculator import get_default_cell_filename


def load_phonopy_from_vasp(poscar_in='POSCAR',
                          force_sets='FORCE_SETS',
                          supercell_matrix=None):
    """
    从VASP输出文件构建Phonopy对象
    
    Args:
        poscar_in (str): 原胞文件
        force_sets (str): 位移-力集合
        supercell_matrix (list[int] or None): 超胞矩阵
        
    Returns:
        phonopy: Phonopy对象
    """
    # 读取原胞
    cell = read_vasp(poscar_in)
    
    # 处理力常数
    force_file = Path(force_sets)
    
    # 构建Phonopy对象
    ph = phonopy.load(supercell_matrix=supercell_matrix,
                           unitcell=cell,
                           calculator='vasp',
                           force_sets_filename=force_file)
    
    # 保存yaml备份
    ph.save(settings={'force_constants': True})
    print("[INFO] Phonopy object built from VASP files.")
    
    return ph


def load_phonopy_from_qe(scf_in='scf.in',
                        fc_raw='system.fc',
                        supercell_matrix=None):
    """
    从QE输出文件构建Phonopy对象
    
    Args:
        scf_in (str): QE `pw.x`的输入文件*+（包含晶胞信息）
        fc_raw (str): `matdyn.x` 或 `q2r.x` 生成的力常数文件
        supercell_matrix (list[int] or None): 超胞矩阵
        
    Returns:
        phonopy: Phonopy对象
    """
    # 读取原胞
    cell, _ = read_pwscf(scf_in)
    
    # 将.fc转换为Phonopy可读的force_constants.hdf5
    q2r = PH_Q2R(fc_raw)
    q2r.run(cell)
    q2r.write_force_constants()
    ph = phonopy.load(supercell_matrix=supercell_matrix,
                          unitcell=cell,
                          calculator='qe',
                          force_constants_filename='force_constants.hdf5')
    
    # 保存yaml备份
    ph.save(settings={'force_constants': True})
    print("[INFO] Phonopy object built from QE files.")
    
    return ph
    
    
def load_phonopy_from_yaml(yaml_file='band.yaml'):
    """
    直接从Phonopy输出的YAML文件构建Phonopy对象
    Args:
        yaml_file (str): Phonoy输出的YAML文件
        
    Returns:
        phonopy: Phonopy对象
    """
    ph = phonopy.load(yaml_file)
    print(f"[INFO] Loaded Phonopy object from {yaml_file}.")
    
    return ph


def run_modulation(ph, qpoint, branch_idx, amplitude_list, dim, natoms_in_primitive=None, code='yaml'):
    """
    计算并写出调制后的POSCAR & YAML
    
    Args:
        ph (Phonopy): 已具备力常数的Phonopy对象
        qpoint (list[float]): q点分数坐标
        branch_idx (int): 声子支指数 (0表示第一支)
        amplitude_list (list[float]): 调制幅度列表
        dim (list[int]): 超胞维度
        natoms_in_primitive (int or None): 原胞原子数; 缺省时取 ph.unitcell 大小
        code (str): 输入文件来源
    """
    natoms = natoms_in_primitive or len(ph.unitcell)
    
    # Phonopy的band index从0开始
    phonon_modes = [[list(qpoint), branch_idx, amp * np.sqrt(natoms), 0.0] for amp in amplitude_list]
    
    ph.run_modulations(dim, phonon_modes)
    ph.write_yaml_modulations()
    
    disps = []
    for amp, u in zip(amplitude_list, ph._modulation._u):
        cell = ph._modulation._get_cell_with_modulation(u)
        write_vasp(filename=f"MPOSCAR_{(amp):2.2f}", cell=cell)
        disps.append(u)
    
    total_disp = np.sum(disps, axis=0)
    cell_total = ph._modulation._get_cell_with_modulation(total_disp)
    write_vasp(filename=f"MPOSCAR-total", cell=cell_total)
        
    zero_disp = np.zeros(total_disp.shape, dtype=complex)
    cell_zero = ph._modulation._get_cell_with_modulation(zero_disp)
    write_vasp(filename=f"MPOSCAR-orig", cell=cell_zero)
        
    if code == 'qe':
        os.system('sed -i "s/   1.0/    0.529177/g" MPOSCAR*')
        print(f"[INFO] Modulation finished → {Path.cwd()}")


if __name__ == "__main__":
    # 输入参数解析
    parser = argparse.ArgumentParser(description='Modulate structure by phonon mode.')
    parser.add_argument('--code', type=str, default='yaml', choices=['vasp', 'qe', 'yaml'],
                        help="inputs from which code (default: 'yaml')")
    parser.add_argument('--input_structure_file', type=str, default='POSCAR',
                        help="input structure file (default: 'POSCAR')")
    parser.add_argument('--input_force_sets_file', type=str, default='FORCE_SETS',
                        help="input force sets file (default: 'FORCE_SETS')")
    parser.add_argument('--input_force_constants_file', type=str, default='system.fc',
                        help="input force constants file (default: 'system.fc')")
    parser.add_argument('--input_supercell_matrix', nargs=3, type=int, default=[2, 2, 2],
                        help="supercell matrix used in inputs (default: [2, 2, 2])")
    parser.add_argument('--input_yaml_file', type=str, default='band.yaml',
                        help="input yaml file (default: 'bamd.yaml')")
    parser.add_argument('--qpoint', nargs=3, type=float, required=True,
                        help='q-point fractional coordinates')
    parser.add_argument('--branch_idx', type=int, default=0, required=True,
                        help='phonon branch index (0-based)')
    parser.add_argument('--amplitudes', nargs=3, type=int, required=True,
                        help='modulate amplitude (unitless)')    
    parser.add_argument('--dim', nargs=3, type=int, required=True,
                        help='supercell dimensions for modulated structures')    
    args = parser.parse_args()
    
    if args.code == 'vasp':
        ph = load_phonopy_from_vasp(poscar_in=args.input_structure_file,
                                         force_sets=args.input_force_sets_file,
                                         supercell_matrix=args.input_supercell_matrix)
    elif args.code == 'qe':
        ph = load_phonopy_from_qe(scf_in=args.input_structure_file,
                                       fc_raw=args.input_force_constants_file,
                                       supercell_matrix=args.input_supercell_matrix)
    elif args.code == 'yaml':
        ph = load_phonopy_from_yaml(yaml_file=args.input_yaml_file)
    else:
        raise ValueError("Code must be 'vasp', 'qe', or 'yaml'.")
        
    run_modulation(ph=ph,
                   qpoint=args.qpoint,
                   branch_idx=args.branch_idx,
                   amplitude_list=np.linspace(start=args.amplitudes[0],
                                              stop=args.amplitudes[1],
                                              num=args.amplitudes[2]),
                   dim=args.dim,
                   code=args.code)
