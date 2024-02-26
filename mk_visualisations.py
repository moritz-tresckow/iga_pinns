import dolfinx
import meshio
from mpi4py import MPI
import pyvista
from pyvista import examples

import ufl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fenicsx_scripts import *

path_coors = './fem_ref/coordinates.csv'
coordinates = np.loadtxt(path_coors, delimiter = ',')


meshfile = './fem_ref/fenicsx_mesh/quad_doms/quad_doms_wider_yoke'

msh, ct, msh_l, ct_l= load_mesh(meshfile)
V = dolfinx.fem.FunctionSpace(msh, ('Lagrange',1))
output_file = './potential_nn.pdf'
uh = calc_eq(meshfile, [1,2000], 1000)
ref_data = uh.x.array
ref_values = np.loadtxt('./ref_data.csv', delimiter = ',')



def plot_on_msh(meshfile, output_file, data):
    vmin = 0
    msh, ct, msh_l, ct_l= load_mesh(meshfile)
    V = dolfinx.fem.FunctionSpace(msh, ('Lagrange',1))
    u_pl = dolfinx.fem.Function(V)
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    sargs = dict(width = 0.5, vertical=False, position_x = 0.25, position_y = 0.075, n_labels = 4,label_font_size = 30, fmt ='%.2f')
    plotter = pyvista.Plotter(notebook = False, off_screen = True, shape = (1,1))
    plotter.add_mesh(grid, 
                     scalars = data,
                     cmap = 'inferno',
                     lighting = 'True',
                     show_edges = False,
                     show_scalar_bar = True,
                     scalar_bar_args=sargs)

    #plotter.view_xy()
    plotter.set_background('white')
    plotter.camera.zoom('tight')
    vmin = np.amin(data)
    vmax = np.amax(data)
    plotter.save_graphic(output_file)
    plotter.close()

plot_on_msh(meshfile, './reference.svg', ref_values)


def eval_on_coordinates_old(uh, domain, boundary, coordinates):
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, coordinates) 
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, coordinates)

        num_entities_local = domain.topology.index_map(2).size_local + domain.topology.index_map(2).num_ghosts
        entities = np.arange(num_entities_local, dtype = np.int32)
        mid_tree = dolfinx.geometry.create_midpoint_tree(domain, 2, entities)
        cells = []
        for i, point in enumerate(coordinates):
            try:
                if len(colliding_cells.links(i))>0:
                    cells.append(colliding_cells.links(i)[0])
                else:
                    cells.append(cell_candidates.links(i)[0])
            except:
                print(i, 'Substituting value with clossest colliding entity')
                ent = dolfinx.geometry.compute_closest_entity(bb_tree, mid_tree, domain, point)
                cells.append(ent[0])
        
        coordinates = np.concatenate((coordinates, np.zeros((coordinates.shape[0], 1))), axis = 1)
        sol = uh.eval(coordinates, cells)
        return sol


eval_on_coordinates_old(uh, msh, ct, coordinates)
