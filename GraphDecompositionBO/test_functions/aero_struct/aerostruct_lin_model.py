""" Script evaluates the effect of different coupling parameters based on
    Gaussian approximations of the output uncertainty of the models. The code
    is adapted from OpenAeroStruct developed by Jasa et. al. (2018) and it 
    currently runs with OpenMDAO 1.7.3 and pyoptsparse 1.0. 

    Authors: Ricardo Baptista, Matthias Poloczek
    Date:    June 2018
"""

from __future__ import division
from time import time
import numpy as np
import pdb

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from openmdao.devtools.partition_tree_n2 import view_tree

from CombinatorialBO.test_functions.aero_struct.geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from CombinatorialBO.test_functions.aero_struct.transfer import TransferDisplacements, TransferLoads
from CombinatorialBO.test_functions.aero_struct.vlm import VLMStates, VLMFunctionals
from CombinatorialBO.test_functions.aero_struct.spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from CombinatorialBO.test_functions.aero_struct.materials import MaterialsTube
from CombinatorialBO.test_functions.aero_struct.functionals import FunctionalBreguetRange, FunctionalEquilibrium

from CombinatorialBO.test_functions.aero_struct.gs_newton import HybridGSNewton
from CombinatorialBO.test_functions.aero_struct.b_spline import get_bspline_mtx
from CombinatorialBO.test_functions.aero_struct.compute_derivs import comp_partial_derivatives

import sys
import pickle
from CombinatorialBO.test_functions.aero_struct.velcoupling import velcoupling
import copy
import os.path


def aerostruct_linear(rand_inputs, fixed_inputs):

    # Seperate fixed input parameters
    num_in   = fixed_inputs['num_in']
    num_out  = fixed_inputs['num_out']
    n_subdom = fixed_inputs['n_subdom']
    dec_vars = fixed_inputs['dec_vars']

    # Setup mesh
    mesh = gen_crm_mesh(num_in, num_out, num_x=2)
    num_x, num_y = mesh.shape[:2]
    num_twist = np.max([int((num_y - 1) / 5), 5])

    r = radii(mesh)
    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = np.atleast_2d(np.array([num_x, num_y]))
    fem_ind = [num_y]
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

    # Set the number of thickness control points and the initial thicknesses
    num_thickness = num_twist
    t = r / 10.

    # Define the aircraft properties
    span  = rand_inputs['span'].val
    alpha = rand_inputs['alpha'].val
    M     = rand_inputs['M'].val
    a     = rand_inputs['a'].val
    rho   = rand_inputs['rho'].val

    # Define the fixed aircraft properties
    W0  = 0.5 * 2.5e6
    CT  = 9.81 * 17.e-6
    R   = 14.3e6
    CL0 = 0.2
    CD0 = 0.015

    # Define the material properties
    from aluminum import E, G, stress, mrho

    # Load mean data
    if not np.all(dec_vars['def_mesh']) or not np.all(dec_vars['loads']):
        mean_data = pickle.load(open("orig_model_output.p", "rb"))
    else:
        N = 2*(num_in + num_out) - 3
        mean_data = {'loads':np.zeros((N,6)), 'def_mesh':np.zeros((2*N,3))}

    # Create the top-level system
    root = Group()

    # Define Jacobians for b-spline controls
    tot_n_fem = np.sum(fem_ind[:, 0])
    num_surf = fem_ind.shape[0]
    jac_twist = get_bspline_mtx(num_twist, num_y)
    jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

    # Define the independent variables
    indep_vars = [
        ('span', span),
        ('twist_cp', np.zeros(num_twist)),
        ('thickness_cp', np.ones(num_thickness)*np.max(t)),
        ('a',a),
        ('M',M),
        ('alpha', alpha),
        ('rho', rho),
        ('r', r),
        ('aero_ind', aero_ind)
    ]

    # Add material components to the top-level system
    root.add('indep_vars',
             IndepVarComp(indep_vars),
             promotes=['*'])
    root.add('twist_bsp',
             Bspline('twist_cp', 'twist', jac_twist),
             promotes=['*'])
    root.add('thickness_bsp',
             Bspline('thickness_cp', 'thickness', jac_thickness),
             promotes=['*'])
    root.add('tube',
             MaterialsTube(fem_ind),
             promotes=['*'])

    # Create a coupled group to contain the aero, sruct, and transfer components
    coupled = Group()
    coupled.add('mesh',
                GeometryMesh(mesh, aero_ind),
                promotes=['*'])
    coupled.add('def_mesh',
                TransferDisplacements(aero_ind, fem_ind),
                promotes=['*'])
    coupled.add('vlmstates',
                VLMStates(aero_ind, dec_vars, mean_data),
                promotes=['*'])
    coupled.add('velcoupling',
                velcoupling(),
                promotes=['*'])
    coupled.add('loads',
                TransferLoads(aero_ind, fem_ind, dec_vars, mean_data),
                promotes=['*'])
    coupled.add('spatialbeamstates',
                SpatialBeamStates(aero_ind, fem_ind, E, G, dec_vars, mean_data),
                promotes=['*'])

    # Set solver properties
    coupled.nl_solver = Newton()
    coupled.nl_solver.options['iprint'] = 1
    coupled.ln_solver = ScipyGMRES()
    coupled.ln_solver.options['iprint'] = 1
    coupled.ln_solver.preconditioner = LinearGaussSeidel()
    coupled.vlmstates.ln_solver = LinearGaussSeidel()
    coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

    coupled.nl_solver = NLGaussSeidel()   ### Uncomment this out to use NLGS
    coupled.nl_solver.options['iprint'] = 1
    coupled.nl_solver.options['atol'] = 1e-5
    coupled.nl_solver.options['rtol'] = 1e-12

    coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
    coupled.nl_solver.nlgs.options['iprint'] = 1
    coupled.nl_solver.nlgs.options['maxiter'] = 6
    coupled.nl_solver.nlgs.options['atol'] = 1e-8
    coupled.nl_solver.nlgs.options['rtol'] = 1e-12
    coupled.nl_solver.newton.options['atol'] = 1e-7
    coupled.nl_solver.newton.options['rtol'] = 1e-7
    coupled.nl_solver.newton.options['maxiter'] = 1
    coupled.nl_solver.newton.options['iprint'] = 1

    # Add the coupled group and functional groups to compute performance
    root.add('coupled',
             coupled,
             promotes=['*'])
    root.add('vlmfuncs',
             VLMFunctionals(aero_ind, CL0, CD0),
             promotes=['*'])
    root.add('spatialbeamfuncs',
             SpatialBeamFunctionals(aero_ind, fem_ind, E, G, stress, mrho),
             promotes=['*'])
    root.add('fuelburn',
             FunctionalBreguetRange(W0, CT, R, aero_ind),
             promotes=['*'])
    root.add('eq_con',
             FunctionalEquilibrium(W0, aero_ind),
             promotes=['*'])

    # Set the optimization problem settings
    prob = Problem()
    prob.root = root

    # Setup the problem and produce an N^2 diagram
    prob.setup()

    # Run the problem as selected in the command line argument
    prob.run_once()

    # Collect output vals and variable index
    output_vals = prob.root.unknowns.vec
    output_vars = model_in_out(prob)

    # Compute gradients of aerostruct model
    (dR_dy, dR_dx, _, _, _) = comp_partial_derivatives(prob, rand_inputs, jacobian_mode='fd')

    # Save ref_data for reference model
    if np.all(dec_vars['def_mesh']) and np.all(dec_vars['loads']):

        # Extract loads and mesh outputs
        ref_data = {}
        ref_data['loads'] = prob['loads']
        ref_data['def_mesh'] = prob['def_mesh']
        ref_data['output_vals'] = output_vals
        ref_data['output_vars'] = output_vars
        ref_data['dR_dy'] = dR_dy
        ref_data['dR_dx'] = dR_dx

        # Save in pickle file
        pickle.dump(ref_data, open("ref_model_output.p", "wb"))

    return ref_data


def model_output(ref_data, fixed_inputs):
    """ MODEL_OUTPUT: Assemble output and gradients of aerostruct model. """

    # Collect data from mean_data
    output_vars = ref_data['output_vars']
    output_vals = ref_data['output_vals']
    dR_dy = ref_data['dR_dy']
    dR_dx = ref_data['dR_dx']

    # Mask decoupled entries
    dR_dy_approx = mask_derivatives(dR_dy, output_vars, fixed_inputs)

    # Compute sensitivities
    dy_dx = -1.0*np.linalg.solve(dR_dy_approx, dR_dx)

    return (output_vals, dy_dx, output_vars)


def mask_derivatives(dR_dy, output_vars, fixed_inputs):
    """ MASK_DERIVATIVES: Function makes entries in dR_dy matrix
    based on indices in the decoupling_idx list. """

    # Seperate fixed input parameters
    dec_vars = fixed_inputs['dec_vars']
    dec_loads = dec_vars['loads']
    dec_mesh  = dec_vars['def_mesh']

    # Setup dR_dy_approx
    dR_dy_approx = copy.deepcopy(dR_dy)

    # Find zero entries in binary model vector
    dec_loads_idx = np.where(dec_loads.reshape(dec_loads.size) == 0)[0]
    dec_mesh_idx  = np.where(dec_mesh.reshape(dec_mesh.size) == 0)[0]

    # Find indices for loads and mesh
    output_str = list(output_vars.keys())
    out_loads_str_idx = [i for i, s in enumerate(output_str) if 'loads' in s]
    out_mesh_str_idx  = [i for i, s in enumerate(output_str) if 'def_mesh' in s]
    out_loads_idx = output_vars[output_str[out_loads_str_idx[0]]]
    out_mesh_idx  = output_vars[output_str[out_mesh_str_idx[0]]]

    # Define coupling (dependents of loads and def_mesh)
    loads_dep = ['coupled.def_mesh.def_mesh','coupled.spatialbeamstates.fem.disp_aug']
    mesh_dep  = ['coupled.loads.loads', 'coupled.vlmstates.circ.circulations', \
        'coupled.vlmstates.forces.sec_forces', 'coupled.vlmstates.wgeom.b_pts', \
        'coupled.vlmstates.wgeom.mid_b', 'coupled.vlmstates.wgeom.c_pts', \
        'coupled.vlmstates.wgeom.widths', 'coupled.vlmstates.wgeom.S_ref', \
        'coupled.vlmstates.wgeom.normals']

    # Set indices corresponding to decoupled variables in dR_dy_approx to 0
    for k in range(len(loads_dep)):

        # Find index for output variable
        loads_dep_str_idx = [i for i, s in enumerate(output_str) if loads_dep[k] in s]
        loads_dep_idx = output_vars[output_str[loads_dep_str_idx[0]]]

        # Decouple entries in dR_dy_approx
        dec_subarray_1 = np.array(range(out_loads_idx[0], out_loads_idx[1]))
        if len(dec_loads_idx) != 0:
            dec_subarray_1 = dec_subarray_1[dec_loads_idx]
            dec_subarray_2 = range(loads_dep_idx[0], loads_dep_idx[1])
            dR_dy_approx[np.ix_(dec_subarray_2,dec_subarray_1)] = 0

    # Set indices corresponding to decoupled variables in dR_dy_approx to 0
    for k in range(len(mesh_dep)):

        # Find index for output variable
        mesh_dep_str_idx = [i for i, s in enumerate(output_vars.keys()) if mesh_dep[k] in s]
        mesh_dep_idx = output_vars[output_str[mesh_dep_str_idx[0]]]

        # Decouple entries in dR_dy_approx
        dec_subarray_1 = np.array(range(out_mesh_idx[0], out_mesh_idx[1]))
        if len(dec_mesh_idx) != 0:
            dec_subarray_1 = dec_subarray_1[dec_mesh_idx]
            dec_subarray_2 = range(mesh_dep_idx[0], mesh_dep_idx[1])
            dR_dy_approx[np.ix_(dec_subarray_2,dec_subarray_1)] = 0

    return dR_dy_approx


def model_in_out(prob):
    
    # Extract names of promoted variables
    sys_prom_name = prob.root._sysdata.to_prom_name

    # Extract all unknowns and connections
    sys_connect   = prob.root.connections
    sys_unknowns  = prob.root.unknowns

    sys_unknowns_keys = sys_unknowns.keys()
    sys_unknowns_vals = sys_unknowns.values()
    
    # Setup dict to store output_vars
    output_vars = {}

    # Print output variable names
    for i in range(len(sys_unknowns_keys)):

        # Extract var and pathname
        var = sys_unknowns_keys[i]
        full_var = sys_unknowns_vals[i]['pathname']

        # If variable in sys_connect, find other name
        if var in sys_connect:
            var_src = sys_connect[var][0]
            var_pro = sys_prom_name[var_src]
        else:
            var_pro = var
            
        # Extract indices and assign in dictionary
        out_idx = sys_unknowns._dat[var_pro].slice
        output_vars[full_var] = out_idx

    return (output_vars)


def model_stats(output, dy_dx, rand_cov, qoi_idx):
    """ MODEL_STATS: Function computes mean and covariance of 
    the decoupled model based on the evaluated derivatives. """

    # Extract mean from output
    model_mean = output;

    # Assemble covariance matrix for original model
    model_cov = np.dot(np.dot(dy_dx,rand_cov),dy_dx.T)

    # Extract mean and covariance for specified qoi
    qoi_array  = np.array(qoi_idx)
    model_mean = model_mean[qoi_array]
    model_cov  = model_cov[qoi_array,:][:,qoi_array]

    return (model_mean, model_cov)


def linearize_model(ref_data, fixed_inputs, qoi, rand_inputs):

    # Collect output vals and derivatives of aerostruct model
    (out_vals, dy_dx, out_vars) = model_output(ref_data, fixed_inputs)

    # Find input covariance matrix
    cov_mat = input_cov(rand_inputs)

    # Find QoI
    qoi_idx = extract_qoi(out_vars, qoi)

    # Compute model stats
    mean, cov = model_stats(out_vals, dy_dx, cov_mat, qoi_idx)

    return mean, cov


def input_cov(rand_inputs):
    """ INPUT_COV: Function setups covariance matrix for the 
    input variables based on specified variance. """
    
    # Setup vector to store variance
    n_inputs = len(rand_inputs)
    diag_var = np.zeros(n_inputs)

    # Set variance for each input
    for input in rand_inputs:

        # Find index for input_name
        input_idx = list(rand_inputs.keys()).index(input)
        diag_var[input_idx] = rand_inputs[input].var

    cov_mat = np.diag(diag_var)

    return cov_mat


def extract_qoi(output_vars, qoi):

    # Get output variables names
    out_names = list(output_vars.keys())

    qoi_idx = []
    for i in range(len(qoi)):

        # Split QoI and check if parts contained in out_names strings
        qoi_split = qoi[i].split('.')
        out_names_idx = [i for i, s in enumerate(out_names) if all(part in s for part in qoi_split)]

        # Find indices in output_vars and append to qoi_idx
        qoi_var_list = output_vars[out_names[out_names_idx[0]]]
        qoi_idx.extend(range(qoi_var_list[0],qoi_var_list[1]))

    return qoi_idx


class gaussian_rv:

    def __init__(self, mean, variance):
        self.type = 'random'
        self.mean = mean
        self.var  = variance
        self.std  = np.sqrt(variance)
        self.val  = []

    def sample(self, n_samples):
        return np.random.normal(self.mean, self.std, n_samples)


def decoupled_vars(N, loads_mat, mesh_mat, dec_loads_vect, dec_mesh_vect):
    """ DECOUPLED_VARS: Function generates a matrix of the
    entries in the loads and mesh fields to be fixed to mean values. """

    # Check that loads_mat and mesh_mat match in size
    if (np.max(loads_mat)+1) != len(dec_loads_vect):
        raise ValueError('Specified model is of incorrect size (loads)')

    if (np.max(mesh_mat)+1) != len(dec_mesh_vect):
        raise ValueError('Specified model is of incorrect size (mesh)')

    # Generate binary matrices for loads and mesh
    dec_loads = np.ones((N,6))
    dec_mesh  = np.ones((2*N,3))

    # Loads: find zero entries from dec_loads_vect
    dec_vect_idx = np.where(dec_loads_vect == 0)[0]
    for k in range(len(dec_vect_idx)):

        # Find index in loads_mat and set entries to zero
        mat_elems = np.where(loads_mat == dec_vect_idx[k])
        dec_loads[mat_elems] = 0

    # Mesh: find zero entries from dec_mesh_vect
    dec_vect_idx = np.where(dec_mesh_vect == 0)[0]
    for k in range(len(dec_vect_idx)):

        # Find index in loads_mat and set entries to zero
        mat_elems = np.where(mesh_mat == dec_vect_idx[k])
        dec_mesh[mat_elems] = 0

    return (dec_loads, dec_mesh)


def dec_mat(N, n_subdom):
    """ Construct matrices with indexing for different domains ."""

    # Find number of points per side
    pts_per_side = (N-1)/2
    if np.mod(pts_per_side, n_subdom) != 0:
        raise ValueError('Number of Subdomains doesn''t divide mesh size!')

    # Find total number of subdomains & length
    tot_sub  = 2*n_subdom + 1
    subd_len = pts_per_side/n_subdom

    # Set the index of the middle subdomain
    mid_subdom = n_subdom

    # Generate matrix for loads
    loads_mat = np.array([], dtype=np.int64).reshape(0,6)
    for i in range(tot_sub):
        
        #start_idx = 6*i
        #end_idx   = 6*i + 5
        #row_idx  = np.linspace(start_idx, end_idx, 6)
        row_idx = i*np.ones(6)

        if i==mid_subdom:
            subd_idx = np.tile(row_idx, (1,1))
        else:
            subd_idx = np.tile(row_idx, (int(subd_len),1))
        loads_mat = np.vstack((loads_mat, subd_idx))

    # Generate matrix for mesh
    mesh_mat = np.array([], dtype=np.int64).reshape(0,3)
    for i in range(tot_sub):
        
        #start_idx = 3*i
        #end_idx   = 3*i + 2
        #row_idx  = np.linspace(start_idx, end_idx, 3)
        row_idx = i*np.ones(3)

        if i==mid_subdom:
            subd_idx = np.tile(row_idx, (1,1))
        else:
            subd_idx = np.tile(row_idx, (int(subd_len),1))
        mesh_mat = np.vstack((mesh_mat, subd_idx))
    #mesh_mat = np.vstack((mesh_mat, mesh_mat + end_idx + 1))
    mesh_mat = np.vstack((mesh_mat, mesh_mat + tot_sub))

    return (loads_mat, mesh_mat)


def setup_model(model_params, model_loads, model_mesh):
    """ Function computes mean and covariance for model. """

    # Unravel model params
    rand_inputs  = model_params['rand_inputs']

    # Set fixed mesh parameters
    fixed_inputs = model_params['fixed_inputs']
    num_in   = fixed_inputs['num_in']
    num_out  = fixed_inputs['num_out']
    n_subdom = fixed_inputs['n_subdom']
    
    # Setup decoupled variable arrays
    N = 2*(num_in + num_out) - 3
    (loads_mat, mesh_mat) = dec_mat(N, n_subdom)
    (dec_loads, dec_mesh) = decoupled_vars(N, loads_mat, mesh_mat, model_loads, model_mesh)
    dec_dict = {'loads':dec_loads, 'def_mesh':dec_mesh}

    # Set fixed inputs
    fixed_inputs['dec_vars'] = dec_dict

    return (rand_inputs, fixed_inputs)


def kldiv_mvn(mean_1, mean_2, cov_1, cov_2):
    """ KLDIV_MVN: Fucntion computes the KL divergence between 
    two multivariate Gaussian distributions specified by a 
    mean vector and covariance matrix. """

    # Determine the size of the vectors
    K = len(mean_1)

    # Compute cov_2\cov_1
    try:
        cov2_sl_cov1 = np.linalg.solve(cov_2, cov_1)
    except:
        return float('Inf')

    # Compute KL divergence
    d_KL = 0.5*(np.trace(cov2_sl_cov1) + \
        np.dot((mean_2 - mean_1).T,np.linalg.solve(cov_2,(mean_2 - mean_1))) \
        - K - np.log(np.linalg.det(cov2_sl_cov1)))

    # Check zero
    if (np.absolute(d_KL) <= 1e-12):
        d_KL = 0
   
    # If NaN, return Inf
    if (np.iscomplex(d_KL) or d_KL < 0 or np.isnan(d_KL)):
        d_KL = float('Inf')

    return d_KL


def kl_decoupled_models(dec_models):
    """
    :param dec_models: np.array shape (21,)
    :param model_params:
    :return:
    """
    # Set random variable parameters
    span = gaussian_rv(59.0, 1.0)
    M = gaussian_rv(0.84, 0.01)
    alpha = gaussian_rv(3.0, 0.1)
    rho = gaussian_rv(0.38, 0.01)
    a = gaussian_rv(295.4, 1.0)

    # Set fixed mesh parameters
    num_in = 2
    num_out = 3
    n_subdom = 3

    # Set QoI
    qoi = ['norm_fuel', 'CL', 'failure']

    # Set random inputs to mean
    span.val = span.mean
    M.val = M.mean
    alpha.val = alpha.mean
    rho.val = rho.mean
    a.val = a.mean

    # Setup random and fixed inputs
    rand_inputs = {'span': span, 'M': M, 'alpha': alpha, 'rho': rho, 'a': a}
    fixed_inputs = {'num_in': num_in, 'num_out': num_out, 'n_subdom': n_subdom, 'dec_vars': []}

    # Setup model_params
    model_params = {'rand_inputs': rand_inputs, 'fixed_inputs': fixed_inputs}

    # Copy model_params
    ref_model_params = copy.deepcopy(model_params)

    # Setup reference binary model
    ref_model_mesh  = np.ones(14)
    ref_model_loads = np.ones(7)

    # Setup rand_inputs, fixed_inputs
    (rand_inputs, ref_fixed_inputs) = setup_model(ref_model_params, ref_model_loads, ref_model_mesh)

    # Check if refence model exists before running it
    if os.path.isfile("ref_model_output.p"):
        ref_data = pickle.load(open("ref_model_output.p","r"))
    else:
        ref_data = aerostruct_linear(rand_inputs, ref_fixed_inputs)

    # Compute mean and covaraince of reference aerostruct model
    (ref_mean, ref_cov) = linearize_model(ref_data, ref_fixed_inputs, qoi, rand_inputs)

    # Load decoupled model from the argument
    model_loads = dec_models[:7]
    model_mesh  = dec_models[7:]

    # Setup decoupled model
    act_model_params = copy.deepcopy(model_params)
    (_, act_fixed_inputs) = setup_model(act_model_params, model_loads, model_mesh)

    # Compute mean and covaraince of decoupled aerostruct model
    (dec_mean, dec_cov) = linearize_model(ref_data, act_fixed_inputs, qoi, rand_inputs)

    # Evaluate KL divergence
    kl_value = kldiv_mvn(ref_mean, dec_mean, ref_cov, dec_cov)
    return kl_value


if __name__ == '__main__':
    couplings = np.random.randint(0, 2, (21,))
    print(couplings)
    print(kl_decoupled_models(couplings))

