#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:11:59 2020

@author: armin
"""

from fenics import *
import numpy as np
import os
from dolfin_utils.meshconvert import meshconvert
#### Helper scripts for mesh import ####

def load_mesh(meshfile):
    if not os.path.exists(meshfile+".h5"):
        meshconvert.convert2xml(meshfile+".msh",meshfile+".xml")
        
        mesh = Mesh(meshfile +".xml")
        boundaries = MeshFunction("size_t", mesh, meshfile + "_facet_region.xml")
        domains    = MeshFunction("size_t", mesh, meshfile + "_physical_region.xml")
    
        hdf = HDF5File(mesh.mpi_comm(), meshfile + ".h5", "w")
        hdf.write(mesh, "/mesh")
        hdf.write(domains, "/domains")
        hdf.write(boundaries, "/boundaries")
        
        os.remove(meshfile + ".xml")
        os.remove(meshfile + "_facet_region.xml")
        os.remove(meshfile + "_physical_region.xml")      
    
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), meshfile+ ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    mesh.init()
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(domains, "/domains")

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    hdf.read(boundaries, "/boundaries")
    return mesh, domains, boundaries
