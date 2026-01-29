#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function：读取Phonopy输出的声子频率和特征矢量，为纳米结构创建波包的位移和速度

Author: Peng-Hu Du, penghdu@foxmail.com
Date: 2026.01.04
"""


import numpy as np
from ase import Atoms
from ase.io import read, write
import yaml  #用于读取 Phonopy 的输出文件 *.yaml


def find_qpoint_index_in_yaml(qpoint, yaml_data, tolerance=1e-5):
    """
    在 Phonopy yaml 格式数据中查找最接近给定 qpoint 的索引
    
    Args:
        qpoint (list): 目标 q 点 [qx, qy, qz]
        yaml_data (dict): 解析后的 yaml 数据
        tolerance (float): 匹配 q 点的容差
        
    Returns:
        int: 匹配的 q 点在 yaml 数据中的索引
    """
    print("\nFinding target qpoint in yaml data...")
    target_qpoint = np.array(qpoint)
    for i, q_info in enumerate(yaml_data['phonon']):
        q_position = np.array(q_info['q-position'])
        if np.allclose(q_position, target_qpoint, atol=tolerance):
            print(f"  - Found qpoint {qpoint} in yaml data (index: {i})")
            return i
    raise ValueError(f"  - Not found qpoint {qpoint} in yaml data with tolerance {tolerance}")


def read_lattice_phonon_frequency_eigenvector_velocity_from_yaml(yaml_file, qpoint, branch_index):
    """
    从 Phonopy 的 *.yaml 输出文件 (如 band.yaml) 中读取晶格、指定 q 点和分支的频率和特征矢量、群速度
    
    Args:
        yaml_file (str): Phonopy yaml 输出文件
        qpoint (list): 目标 q 点 [qx, qy, qz]
        branch_index (int): 目标声子分支索引 (0-based)
    
    Returns:
        tuple: (reciprocal_lattice_matrix, primitive_lattice_matrix, primitive_lattice_points, frequency_THz (float), eigenvector_complex (np.array), group_velocity (np.array))
    """
    print(f"\nReading lattice, phonon frequency and eigenvector at qpoint {qpoint} in branch {branch_index} from {yaml_file}...")
    
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # 读取倒格子基矢
    reciprocal_lattice_matrix = np.array(yaml_data['reciprocal_lattice'])  #shape: (3, 3)
    print(f"  - Read reciprocal lattice matrix (rows are a*, b*, c* in 2pi/A): \n{reciprocal_lattice_matrix}")
    
    # 读取原胞基矢
    primitive_lattice_matrix = np.array(yaml_data['lattice'])  #shape: (3, 3)
    print(f"  - Read primitive lattice matrix (rows are a, b, c in A): \n{primitive_lattice_matrix}")

    # 读取原胞原子
    primitive_lattice_points = yaml_data['points']  #list of dicts
    print(f"  - Read primitive cell with {len(primitive_lattice_points)} atoms")


    # 查找 q 点
    qpoint_idx = find_qpoint_index_in_yaml(qpoint, yaml_data)
    
    # 获取 q 点的数据
    qpoint_data = yaml_data['phonon'][qpoint_idx]
    
    # 获取频率 (单位 THz)
    freqs_THz = [band['frequency'] for band in qpoint_data['band']]  #list of floats
    if branch_index >= len(freqs_THz):
        raise ValueError(f"Branch index {branch_index} exceeds number of branches ({len(freqs_THz)}) at qpoint {qpoint}")
    
    selected_freq_THz = freqs_THz[branch_index]
    print(f"  - Selected branch index: {branch_index}, Frequency (THz): {selected_freq_THz}")
    
    # 获取特征矢量 (单位 amu^0.5)
    evecs_complex_raw = qpoint_data['band'][branch_index]['eigenvector']  #shape: (natoms, 3, 2), format: [[[real_x1, imag_x1], [real_y1, imag_y1], [real_z1, imag_z2]...]...]

    evecs_complex = [] 
    for atom_data in evecs_complex_raw:  #iterate over atoms
        for component_data in atom_data:  #iterate over x, y, z components
            real_part = component_data[0]
            imag_part = component_data[1]
            evecs_complex.append(complex(real_part, imag_part))
    evecs_complex = np.array(evecs_complex, dtype=complex)  #shape: (3*natoms,)  

    natoms = evecs_complex.shape[0] // 3
    if natoms == 0:
        raise ValueError(f"Calculated number of atom is 0. Check the eigenvector data for qpoint {qpoint}, branch {branch_index}")

    selected_evecs_complex = evecs_complex.reshape(natoms, 3)  #为了方便后续计算，重塑为(natoms, 3)
    print(f"  - Eigenvector shape (after reshape): {selected_evecs_complex.shape}")

    # 获取群速度
    try:
        group_velocity_A_THz = qpoint_data['band'][branch_index]['group_velocity']  # list [vx, vy, vz] in A*THz
        group_velocity_A_THz = np.array(group_velocity_A_THz)  # np.array, shape: (3,)
        print(f"  - Read group velocity (A THz): {group_velocity_A_THz}")
    except (KeyError, IndexError) as e:
        print(f"  - Error reading group velocity: {e}")
        raise ValueError(f"Group velocity not found in yaml file for qpoint {qpoint}, branch {branch_index}.")
    
    return reciprocal_lattice_matrix, primitive_lattice_matrix, primitive_lattice_points, selected_freq_THz, selected_evecs_complex, group_velocity_A_THz


def create_phonon_wavepacket_from_phonopy(structure_file, phonopy_yaml_file, qpoint, branch_index, 
                             amplitude=0.1, width=5.0, center=None):
    """
    从 Phonopy yaml 输出文件读取声子信息，用于创建声子波包，生成GPUMD的输入结构文件
    
    Args:
        struct_file (str): 包含原子坐标的 *.xyz 结构文件
        phonopy_yaml_file (str): Phonopy 输出的 *.yaml 文件 (band.yaml)
        qpoint (list): 用于创建波包的 qpoint [qx, qy, qz] 
        branch_index (int): 选择的声子支索引 (0-based)
        amplitude (float): 控制波包振幅
        width (float): 控制波包宽度 (高斯函数的标准差)
        center (list or None): 波包中心坐标[x, y, z]；如果为None，则使用结构的几何中心;
        output_file (str): 输出的*.extxyz 结构文件
        
    Returns:
        tuple: (initial_displacements (np.array), initial_velocities (np.array), masses (np.array))
    """
    print(f"\nReading structure file {structure_file}...")
    atoms_supercell = read(structure_file, format='extxyz')
    natoms_supercell = len(atoms_supercell)
    masses_supercell = atoms_supercell.get_masses()
    positions_supercell = atoms_supercell.get_positions()  #shape: (natoms_supercell, 3)
    
    if center is None:
        center = np.mean(positions_supercell, axis=0)
    else:
        center = np.array(center)  # 转换为 numpy 数组
    print(f"  - Read structure, number of atoms: {natoms_supercell}, wavepacket center: {center}")

    
    # 从 Phonopy yaml 文件读取晶格信息、声子频率和特征矢量
    reciprocal_lattice_matrix, primitive_lattice_matrix, primitive_lattice_points, freq_THz, evecs_complex, group_velocity_A_THz = read_lattice_phonon_frequency_eigenvector_velocity_from_yaml(phonopy_yaml_file, qpoint, branch_index)
    freq = freq_THz * 1e12 * 2 * np.pi  # 转换为角频率 (rad/s)
    group_velocity = group_velocity_A_THz * 1e12  # unit: A/s

    
    print("\nMapping supercell atoms to primitive cell atoms...")
    primitive_lattice_inv = np.linalg.inv(primitive_lattice_matrix)  #计算原胞基矢的逆矩阵 
    positions_frac = (positions_supercell @ primitive_lattice_inv)  #计算超胞原子相对于原胞基矢的分数坐标
    

    # 存储映射信息的字典, 键是超胞原子索引 i, 值包括 k (原胞内原子索引), l (原胞索引), R_l (原胞位置) 的字典
    mapping_info = []
    for i in range(natoms_supercell):
        position_frac_i = positions_frac[i]

        l = np.floor(position_frac_i).astype(int)  #shape: (3,)
        position_frac_i_in_primitive = position_frac_i - l  #shape: (3,)
        R_l = l @ primitive_lattice_matrix  #shape: (3, )

        # 寻找最接近的原胞原子 k
        min_dist_sq = float('inf')
        closest_k = -1
        for k, point in enumerate(primitive_lattice_points):
            position_frac_k = np.array(point['coordinates'])  #shape: (3,)
            for offset in np.array([[dx, dy, dz] for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]):                
                position_frac_shifted = position_frac_k + offset
                dist_sq = np.sum((position_frac_i_in_primitive - position_frac_shifted)**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_k = k
        if closest_k == -1:
            raise RuntimeError(f"Could not find corresponding primitive cell atom for supercell atom {i}")

        # 存储映射信息到字典
        mapping_info.append({'k': closest_k, 'l': l, 'R_l': R_l})
    print(f"  - Mapping completed for {natoms_supercell} supercell atoms")

    
    print("\nCreating phonon wavepacket...")
    print("  - Calculating initial displacements and velocities...")
    disps_complex = np.zeros((natoms_supercell, 3), dtype=complex)
    vels_complex = np.zeros((natoms_supercell, 3), dtype=complex)
    
    qpoint_cartesian = 2 * np.pi * np.dot(np.array(qpoint), reciprocal_lattice_matrix)  #shape: (3,)
    print(f"  - Calculated qpoint cartesian (rad/A): {qpoint_cartesian}")
    
    for i in range(natoms_supercell):
        phase_factor = np.exp(1j * np.dot(qpoint_cartesian, mapping_info[i]['R_l'] - center))
        gaussian_factor = np.exp(-np.linalg.norm(mapping_info[i]['R_l'] - center)**2 / (2 * width**2))
        disps_complex[i, :] = amplitude * (evecs_complex[mapping_info[i]['k']] / np.sqrt(masses_supercell[i])) * gaussian_factor * phase_factor  # unit: A
        vels_complex[i, :] = disps_complex[i, :] * (np.dot(group_velocity, mapping_info[i]['R_l'] - center) / (width**2) - 1j * freq)  # unit: A/s

    disps_real =  disps_complex.real
    vels_real = vels_complex.real
    print(f"  - Initial displacement shape: {disps_real.shape}")
    print(f"  - Initial velocity shape: {vels_real.shape}")
    
    return atoms_supercell, disps_real, vels_real, mapping_info


def write_gpumd_config_with_properties(atoms_supercell, disps_real, vels_real, mapping_info, nep_model_file='nep.txt', output_file='model_with_wavepacket.xyz'):
    """
    接入 create_phonon_wavepacket_from_phonopy 函数的输出, 输出 GPUMD 构型文件
    
    Args:
        atoms_supercell (ase.Atoms): 原始超胞结构
        disps_real (np.array): 实空间的原子位移, unit: A
        vels_real (np.array): 实空间的原子速度, unit: A/s
        nep_model_file (str): NEP 模型文件 (default: 'nep.txt')
        output_file (str): 输出文件名 (default: 'model_with_wavepacket.xyz')
    """
    print(f"\nWriting GPUMD configuration file {output_file}...")
    
    # 创建用于写入的 Atoms 对象
    atoms_with_wavepacket = atoms_supercell.copy()

    positions_new = atoms_supercell.get_positions() + disps_real  # unit: A
    atoms_with_wavepacket.set_positions(positions_new) 
    
    #atoms_with_wavepacket.set_cell(np.array([[54.12400+0.1, 0, 0], [0, 62.90900+0.1, 0], [0, 0, 15]]))
    #atoms_with_wavepacket.set_pbc([False, False, False])

    
    # 添加 mapping 信息到 arrays
    ## 创建用于存储 l 和 k 信息的数组
    #natoms = len(atoms_with_wavepacket)
    #l_info = np.array([mapping_info[i]['l'] for i in range(natoms)], dtype=int)
    #k_info = np.array([mapping_info[i]['k'] for i in range(natoms)], dtype=int)
    
    ## 将信息添加到 arrays    
    #atoms_with_wavepacket.arrays['mapping_l'] = l_info
    #atoms_with_wavepacket.arrays['mapping_k'] = k_info  
 

    # 加载 NEP 计算器
    try:
        from pynep.calculate import NEP
        calc = NEP(nep_model_file)
        atoms_with_wavepacket.calc = calc
        print("  - NEP calculator loaded and set successfully")
    except ImportError as e:
        print(f"  - Error importing pynep: {e}")
        return
    except Exception as e:
        print(f"  - Error loading NEP model or setting calculator: {e}")
        return    

    # 计算势能
    try:
        potential_energy = atoms_with_wavepacket.get_potential_energy()  # unit: eV
        print(f"  - Calculated potential energy (eV): {potential_energy:.8f}")
    except Exception as e:
        print(f"  - Error calculating potential energy: {e}")
        return   # 如果力计算异常, 可能整个物理量计算都不可靠
    
    # 计算原子势能
    #try:
    #    potential_energies_atom = atoms_with_wavepacket.get_potential_energies()  # unit: eV
    #    print(f"  - Calculated atomic potential energies (eV) for {len(potential_energies_atom)} atoms")
    #except Exception as e:
    #    print(f"  - Error calculating atomic potential energies: {e}")
    #    potential_energies_atom = np.full(len(atoms_with_wavepacket), np.nan, dtype=np.float64)

    # 计算动能
    #try:
    #    atoms_with_wavepacket.set_velocities(vels_real)  # unit: A/fs
    #    kinetic_energy = atoms_with_wavepacket.get_kinetic_energy()  # unit: eV
    #    print(f"  - Calculated kinetic energy (eV): {kinetic_energy:.8f}")
    #except Exception as e:
    #    print(f"  - Error calculating kinetic energy: {e}")
    #    kinetic_energy = 0.0  # 如果出错, 设为0
        
    # 计算总能量
    #total_energy = potential_energy + kinetic_energy
    #total_energy = atoms_with_wavepacket.get_total_energy()
    #print(f"  - Calculated total energy (PE + KE) (eV): {total_energy:.8f}")
    
    # 计算应力
    #try:
    #    stress_full = atoms_with_wavepacket.get_stress(voigt=False)  # unit: eV/A^3, shape: (3, 3)
    #    stress_flat = stress_full.flatten()  # unit: eV/A^3, shape: (9,)
    #    print(f"  - Calculated stress (GPUMD format, eV/A^3): {stress_flat}")
    #except Exception as e:
    #    print(f"  - Error calculating stress: {e}")
    #    stress_flat = np.zeros(9)  # 如果出错, 设为0向量

    # 计算维里
    #try:
    #    volume = atoms_with_wavepacket.get_volume()  # unit: A^3
    #    virial_full = stress_full * volume  # unit: eV, shape: (3, 3)
    #    virial_flat = virial_full.flatten()  # unit: eV, shape: (9,)
    #    print(f"  - Calculated virial (GPUMD format, eV): {virial_flat}")
    #except Exception as e:
    #    print(f"  - Error calculating virial: {e}")
    #    virial_flat = np.zeros(9)  # 如果出错, 设为0向量
        
    # 计算原子受力
    #try:
    #    forces = atoms_with_wavepacket.get_forces()  # unit: eV/A
    #    print(f"  - Calculated forces (eV/A) for {len(forces)} atoms")
    #except Exception as e:
    #    print(f"  - Error calculating forces: {e}")
    #    return  # 如果力计算异常, 可能整个物理量计算都不可靠


    # 添加信息和数组
    atoms_with_wavepacket.calc = None 
    atoms_with_wavepacket.info['Time'] = 0.0  # unit: fs
    atoms_with_wavepacket.info['energy'] = potential_energy  # unit: eV
    #atoms_with_wavepacket.info['virial'] = virial_flat  # unit: eV
    #atoms_with_wavepacket.info['stress'] = stress_flat  # unit: eV/A^3
    atoms_with_wavepacket.arrays['vel'] = vels_real / 1e15  # unit: A/fs
    #atoms_with_wavepacket.arrays['forces'] = forces  # unit: eV/A
    #atoms_with_wavepacket.arrays['energy_atom'] = potential_energies_atom  # unit: eV
    
    # 写入文件
    try:
        write(output_file, atoms_with_wavepacket, format='extxyz')
        print(f"  - GPUMD configuration file saved as {output_file} (extxyz format, includes velocities, forces, energy)")
    except Exception as e:
        print(f"  - Error writing to file {output_file}: {e}")
      

if __name__ == "__main__":
    # 配置参数
    struct_file = './model_without_wavepacket.xyz'
    phonopy_yaml_file = 'band.yaml'
    qpoint = [0.25, 0, 0]
    branch_index = 0
    amplitude = 0.001  # 控制波包振幅, unit: A
    width = 5  # 控制波包宽度, unit: A
    center = None  # 初始波包中心, unit: A
    nep_model_file = './nep.txt'
    output_file = './model_with_wavepacket.xyz'  # 输出的 GPUMD 构型文件
    
    try:
        atoms_supercell, disps_real, vels_real, mapping_info = create_phonon_wavepacket_from_phonopy(struct_file, phonopy_yaml_file, qpoint, branch_index, amplitude, width, center)
        write_gpumd_config_with_properties(atoms_supercell, disps_real, vels_real, mapping_info, nep_model_file, output_file)
        print("\nPhonon wavepacket initialization completed.")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except KeyError as e:
        print(f"Yaml file format error or missing key: {e}")
        print("Please check if the yaml file was correctly generated by Phonopy.")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
