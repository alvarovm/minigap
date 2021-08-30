#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from IPython.display import HTML

class Structure3DPlot:
    def __init__(self, structure):
        self.structure = structure.copy()
        self.SetVisualParameters()

        
    def SetVisualParameters(self, **kwargs):
        # bl is short for bond length
        # bw is short for bond width
        
        self.title = kwargs.get("title", "")        
        
        self.figsize = kwargs.get('figsize', (6.5, 6))
        self.elevation = kwargs.get('elevation', 70)
        self.azimuth = kwargs.get('azimuth', 0)
        self.sidelength = kwargs.get('sidelength', 6)
        self.no_grid = kwargs.get('no_grid', True)
        self.no_axis = kwargs.get('no_axis', True)
        
        self.element_colors = kwargs.get('element_colors', {"C":"g", "O":"r", "H":"k", "N":"b"})
        self.element_sizes = kwargs.get('element_sizes', {"C":0.77, "O":0.74, "H":0.46, "N":0.74})
        
        self.atom_size_scale_factor = kwargs.get('atom_size_scale_factor', 120/self.sidelength)
        
        self.bond_color = kwargs.get('bond_color', "gray")
        
        self.variable_bw = kwargs.get('variable_bw', True)
        self.relaxed_bl = kwargs.get('relaxed_bl', 1.45)
        self.max_bl = kwargs.get('max_bl', 1.8)
        self.relaxed_bw = kwargs.get('relaxed_bw', 24/self.sidelength)
        self.bond_grow_exponent = kwargs.get('bond_grow_exponent', 1.5)
        self.bond_shrink_exponent = kwargs.get('bond_shrink_exponent', 4)

        self.SetAtomList()
        self.SetBondList()
           
    def SetAtomList(self):
        # center around origin
        self.positions = self.structure.positions - np.mean(self.structure.positions, axis=0)
        self.n_atoms = len(self.positions)
        self.elements = self.structure.get_chemical_symbols()
        self.atom_sizes = []
        self.atom_colors = []
        for i in range(self.n_atoms):
            self.atom_sizes.append(self.atom_size_scale_factor * self.element_sizes[self.elements[i]])
            self.atom_colors.append(self.element_colors[self.elements[i]])
        return self.positions
                 
    def SetBondList(self):
        self.bonds = []
        self.bws = []
        for i in range(self.n_atoms):
            for j in range(i):
                bl_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                if bl_ij <= self.max_bl:
                    self.bonds.append(np.array([*self.positions[[i,j]].transpose()]))
                    self.bws.append(self.BondWidth(bl_ij))
        self.n_bonds = len(self.bonds)
        self.bonds = np.array(self.bonds)
        return self.bonds
    
                            
    def BondWidth(self, bl):
        if bl <= self.max_bl:
            if self.variable_bw:
                if bl > self.relaxed_bl:
                    bw_exponent = self.bond_shrink_exponent
                else:
                    bw_exponent = self.bond_grow_exponent
                bw = self.relaxed_bw*np.exp(-bw_exponent*(bl - self.relaxed_bl))
                return bw
            else:
                return self.relaxed_bw
        else:
            return 0.0
        
    def Plot(self):
        self.fig = plt.figure(figsize = self.figsize)#figsize)
        self.ax = plt.subplot(111, projection='3d')
        self.boundary = self.sidelength/2
        self.ax.set_xlim(-self.boundary, self.boundary)
        self.ax.set_ylim(-self.boundary, self.boundary)
        self.ax.set_zlim(-self.boundary, self.boundary)
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        if self.no_grid:
            self.ax.grid(False)
        if self.no_axis:
            self.ax.axis('off')
            
        for i in range(self.n_atoms):
            self.ax.plot3D(*self.positions[i][:, None], 'o', c=self.atom_colors[i], ms=self.atom_sizes[i], zorder=1)
        
        for i in range(self.n_bonds):
            self.ax.plot3D(*self.bonds[i], '-', c = self.bond_color, lw = self.bws[i], zorder=0)
        
        self.ax.set_title(self.title)
        
        return self.ax
        
        
class Structure3DAnimation:
    def __init__(self, structure_list):
        self.structure_list = structure_list.copy()
        self.SetVisualParameters()
        self.n_structs = len(self.structure_list)
        # get guess for number of atoms and bonds
        self.n_atoms_likely_max = len(self.structure_list[0].positions)
        self.n_bonds_likely_max = self.n_atoms_likely_max*2 + 1
        
        
    def SetVisualParameters(self, **kwargs):
        # bl is short for bond length
        # bw is short for bond width
        
        self.verbose = kwargs.get('verbose', True)
        self.print_interval = kwargs.get("print_interval", 10)
        self.title = kwargs.get("title", "")
        
        self.frames = kwargs.get('frames', 100)
        self.frame_rate = kwargs.get('frame_rate', 10)
        
        self.rotate = kwargs.get('rotate', False)
        self.rotation_rates = kwargs.get('rotation_rates', {'elev':.1, 'azim':.05})
        
        self.figsize = kwargs.get('figsize', (6.5, 6))
        self.elevation = kwargs.get('elevation', 70)
        self.azimuth = kwargs.get('azimuth', 0)
        self.sidelength = kwargs.get('sidelength', 6)
        self.no_grid = kwargs.get('no_grid', True)
        self.no_axis = kwargs.get('no_axis', True)
        
        self.element_colors = kwargs.get('element_colors', {"C":"g", "O":"r", "H":"k", "N":"b"})
        self.element_sizes = kwargs.get('element_sizes', {"C":0.77, "O":0.74, "H":0.46, "N":0.74})
        self.atom_size_scale_factor = kwargs.get('atom_size_scale_factor', 120/self.sidelength)
        
        self.adjust_COM = kwargs.get("adjust_COM", True)
        
        self.bond_color = kwargs.get('bond_color', "gray")
        
        self.variable_bw = kwargs.get('variable_bw', True)
        self.relaxed_bl = kwargs.get('relaxed_bl', 1.45)
        self.max_bl = kwargs.get('max_bl', 1.8)
        self.relaxed_bw = kwargs.get('relaxed_bw', 24/self.sidelength)
        self.bond_grow_exponent = kwargs.get('bond_grow_exponent', 1.5)
        self.bond_shrink_exponent = kwargs.get('bond_shrink_exponent', 4)
        
           
    def SetAtomList(self):
        # center around origin
        if self.adjust_COM in {"dynamic", True}:
            self.positions = self.structure.positions - np.mean(self.structure.positions, axis=0)
        elif self.adjust_COM == "initial":
            self.positions = self.structure.positions - np.mean(self.structure_list[0].positions, axis=0)
        elif self.adjust_COM in {"off", False}:
            self.positions = self.structure.positions
        
        self.n_atoms = len(self.positions)
        self.elements = self.structure.get_chemical_symbols()
        self.atom_sizes = []
        self.atom_colors = []
        for i in range(self.n_atoms):
            self.atom_sizes.append(self.atom_size_scale_factor * self.element_sizes[self.elements[i]])
            self.atom_colors.append(self.element_colors[self.elements[i]])
        return self.positions
                 
    def SetBondList(self):
        self.bonds = []
        self.bws = []
        for i in range(self.n_atoms):
            for j in range(i):
                bl_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                if bl_ij <= self.max_bl:
                    self.bonds.append(np.array([*self.positions[[i,j]].transpose()]))
                    self.bws.append(self.BondWidth(bl_ij))
        self.n_bonds = len(self.bonds)
        if self.n_bonds > self.n_bonds_likely_max:
            print("Warning: More bonds detected than expected")
        self.bonds = np.array(self.bonds)
        return self.bonds
    
                            
    def BondWidth(self, bl):
        if bl <= self.max_bl:
            if self.variable_bw:
                if bl > self.relaxed_bl:
                    bw_exponent = self.bond_shrink_exponent
                else:
                    bw_exponent = self.bond_grow_exponent
                bw = self.relaxed_bw*np.exp(-bw_exponent*(bl - self.relaxed_bl))
                return bw
            else:
                return self.relaxed_bw
        else:
            return 0.0
        
    def Plot(self):
        self.fig = plt.figure(figsize = self.figsize)#figsize)
        self.ax = plt.subplot(111, projection='3d')
        self.boundary = self.sidelength/2
        self.ax.set_xlim(-self.boundary, self.boundary)
        self.ax.set_ylim(-self.boundary, self.boundary)
        self.ax.set_zlim(-self.boundary, self.boundary)
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        if self.no_grid:
            self.ax.grid(False)
        if self.no_axis:
            self.ax.axis('off')
            
        if type(self.frames) == int:
            self.frame_list = np.linspace(0, self.n_structs, self.frames, endpoint=False, dtype=int)
        else:
            self.frame_list = self.frames
            
        self.lines_atoms = []
        for i in range(self.n_atoms_likely_max):
            line_atoms, = self.ax.plot3D([], [], 'o')
            self.lines_atoms.append(line_atoms)
        self.lines_bonds = []
        for i in range(self.n_bonds_likely_max):
            line_bonds, = self.ax.plot3D([], [], '-', c=self.bond_color)
            self.lines_bonds.append(line_bonds)
    
        self.interval = 1000/self.frame_rate
        self.frame_counter = 0
        self.animation = animation.FuncAnimation(self.fig, self.DrawFrame, frames=self.frame_list, interval=self.interval, blit=True)
        
        plt.close(self.fig)        
        return HTML(self.animation.to_html5_video())

    
    def DrawFrame(self, j):
        
        if self.verbose and not np.mod(self.frame_counter, self.print_interval):
            print("Generating animation frame {} ({:.0%} complete)".format(self.frame_counter, self.frame_counter/len(self.frame_list)))
        self.frame_counter += 1
        
        self.structure = self.structure_list[j].copy()
        self.SetAtomList()
        self.SetBondList()        
        
        for i in range(self.n_atoms):
            self.lines_atoms[i].set_data(self.positions[i][0], self.positions[i][1])
            self.lines_atoms[i].set_3d_properties(self.positions[i][2])
            self.lines_atoms[i].set_color(self.atom_colors[i])
            self.lines_atoms[i].set_markersize(self.atom_sizes[i])
            self.lines_atoms[i].set_zorder(1)
        for i in range(self.n_bonds_likely_max):
            if i < self.n_bonds:
                self.lines_bonds[i].set_data(self.bonds[i][0], self.bonds[i][1])
                self.lines_bonds[i].set_3d_properties(self.bonds[i][2])
                self.lines_bonds[i].set_zorder(0)
                self.lines_bonds[i].set_linewidth(self.bws[i])
            else:
                self.lines_bonds[i].set_linewidth(0)
                
        if self.rotate:
            self.ax.view_init(elev=self.elevation + self.rotation_rates["elev"]*j , azim=self.azimuth + self.rotation_rates["azim"]*j)
        
        self.title_j = self.title + "\nStructure #{}".format(j)
        self.ax.set_title(self.title_j)
        return self.lines_atoms