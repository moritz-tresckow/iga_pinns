import dolfinx
import dolfinx.fem.petsc
import meshio
from mpi4py import MPI
import pyvista
import ufl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import sys
np.set_printoptions(threshold = sys.maxsize)



def load_mesh(path):
    def write_msh(path, msh, cell_type, prune_z = True):
        path = path + '_' + cell_type + '.xdmf'
        points = msh.points[:,:2]  if prune_z else msh.points
        meshio.write(path , meshio.Mesh(points = points, cells = {cell_type: msh.get_cells_type(cell_type)}, cell_data = {'name_to_read': [msh.get_cell_data('gmsh:physical',cell_type)]}))
	
    def read_domain_msh(path):
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path, 'r') as xdmf:
	         msh = xdmf.read_mesh(name='Grid')
	         ct = xdmf.read_meshtags(msh, name = 'Grid')
        return msh, ct 
    

    def read_boundary_msh(path, msh):
    	msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim -1)
    	with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path, 'r') as xdmf:
    		msh_l = xdmf.read_mesh(name='Grid')
    		ct_l = xdmf.read_meshtags(msh, name = 'Grid')
    	return msh_l, ct_l 
    
    
    msh = meshio.read(path + '.msh')
    write_msh(path, msh, 'triangle')
    write_msh(path, msh, 'line')
    msh, ct = read_domain_msh(path + '_triangle.xdmf')
    msh_l, ct_l = read_boundary_msh(path + '_line.xdmf', msh)

    return msh, ct, msh_l, ct_l

def mk_material(msh, ct, vertices,  material_markers, material_vals):	
        Q = dolfinx.fem.FunctionSpace(msh, ('DG', 0))
        nu = dolfinx.fem.Function(Q)
    
        def set_material(material_markers, material_vals):
    	    for i,j in zip(material_markers, material_vals):
    		    mat = ct.find(i)
    		    nu.x.array[mat] = np.full_like(mat, j, dtype = ScalarType)
    	    return nu 
        nu = set_material(material_markers, material_vals)
        return nu 

def mk_source(msh, ct, vertices,  source_markers, source_vals):	
        Q = dolfinx.fem.FunctionSpace(msh, ('DG', 0))
        j_source = dolfinx.fem.Function(Q)
    
        def set_source(source_markers, source_vals):
    	    for i,j in zip(source_markers, source_vals):
    		    j_s = ct.find(i)
    		    j_source.x.array[j_s] = np.full_like(j_s, j, dtype = ScalarType)
    	    return j_source 
        j_source = set_source(source_markers, source_vals)
        return j_source 

def curl2D(v):
    return ufl.as_vector((v.dx(1), v.dx(0)))

def calc_eq(meshfile, mu, js, coordinates):
    msh, ct, msh_l, ct_l= load_mesh(meshfile)
    V = dolfinx.fem.FunctionSpace(msh, ('Lagrange',1))
    vertices = V.tabulate_dof_coordinates() 
    #vertices = vertices[:,0:2]

    boundary_markers = [5] 
    dirichlet_vals = [0]	
    material_markers = [1,2,3] 
    mu0 = mu[0]
    mur = mu[1] 
    material_vals = [1/mu0, 1/(mu0*mur), 1/mu0]	

    def locate_dofs(idx):
        dirichlet_facets = ct_l.find(idx)
        dirichlet_dofs = dolfinx.fem.locate_dofs_topological(V, 1, dirichlet_facets)
        return dirichlet_dofs
    bcs = []
    d_dofs = []
    for i,j in zip(boundary_markers, dirichlet_vals):
        dirichlet_dofs = locate_dofs(i)
        bcs.append(dolfinx.fem.dirichletbc(ScalarType(j), dirichlet_dofs, V))
        d_dofs.append(dirichlet_dofs)

    nu = mk_material(msh, ct, vertices, material_markers, material_vals)

    dx = ufl.Measure('dx', domain = msh, subdomain_data = ct)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    uh = dolfinx.fem.Function(V)
    
    source_markers = [3]
    source_vals = [js]
    j_source = mk_source(msh, ct, vertices, source_markers, source_vals)	

    A = nu*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx 
    A = dolfinx.fem.form(A)
    lhs = dolfinx.fem.petsc.assemble_matrix(A, bcs)
    lhs.assemble()

    b = j_source*v*ufl.dx
    b = dolfinx.fem.form(b) 
    rhs = dolfinx.fem.petsc.create_vector(b)
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(lhs)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    with rhs.localForm() as loc_rhs:
        loc_rhs.set(0)
    dolfinx.fem.petsc.assemble_vector(rhs, b)
        
    dolfinx.fem.petsc.apply_lifting(rhs, [A], [bcs]) 
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode = PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(rhs, bcs)
    solver.solve(rhs, uh.vector)

    print('Minimum', np.amin(uh.x.array))
    print('Maximum', np.amax(uh.x.array))
    print('Calculated something!!')


    def eval_on_coordinates(uh, domain, boundary, coordinates):
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
    print('Evaluating on coordinates...')
    sol = eval_on_coordinates(uh, msh, ct, coordinates)
    print(sol.shape)
    #exit()
    #np.savetxt('./ref_data.csv', uh.x.array, delimiter = ',', comments = '')
    return sol







def eval_on_coordinates_old(uh, domain, boundary, coordinates):
        #bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)
        exit()
        bb_tree = dolfinx.geometry.BoundingBoxTree(domain)
        coordinates = np.concatenate((coordinates, np.zeros((coordinates.shape[0], 1))), axis = 1)
        exit()
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, coordinates[0:3,:]) 
        print(cell_candidates)
        exit()
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
        sol = uh.eval(coordinates, cells)
        return sol




def cal_functional(vals, msh):
    V = dolfinx.fem.FunctionSpace(msh, ('Lagrange',1))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = vals
    func =u*ufl.dx 
    func = dolfinx.fem.form(func)
    func_val = dolfinx.fem.assemble_scalar(func)
    return np.array(func_val)


def plot_on_coordinates(ref_values):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    vmin = np.amin(ref_values)
    vmax = np.amax(ref_values)

    plt.figure()
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap = 'viridis')
    m.set_array([])

    for i in [5,6,7]:
        step = 100**2
        local_coors = coordinates[i*step:(i+1)*step, :]
        local_vals = ref_values[i*step:(i+1)*step]
        local_x = local_coors[:,0]
        local_y = local_coors[:,1]
        xx = np.reshape(local_x, (100, 100))
        yy = np.reshape(local_y, (100, 100))
        uu = np.reshape(local_vals, (100, 100))
        plt.contourf(xx, yy, uu, norm = norm, levels = 100)
    plt.colorbar(m)
    plt.savefig('./new_pic.png')



#meshfile = './fem_ref/fenicsx_mesh/quad_new/quad_new'
#path_coors = './fem_ref/coordinates.csv'
#coordinates = np.loadtxt(path_coors, delimiter = ',')
#coordinates = 0
# ref_values = calc_eq(meshfile, [1,2000], 1000, coordinates)
#print(np.amax(ref_values), np.amin(ref_values))
#plot_on_coordinates(ref_values)
