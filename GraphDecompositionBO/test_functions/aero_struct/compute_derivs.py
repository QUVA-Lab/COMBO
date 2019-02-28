from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.component import Component

from openmdao.components.indep_var_comp import IndepVarComp

import numpy as np
import scipy.io as sio

def comp_partial_derivatives(prob, inputs, jacobian_mode='fd'):
    """ Compute partial derivatives for all components in the model """

    # Extract root
    root = prob.root

    # Linearize the model
    root._sys_linearize(root.params, root.unknowns, root.resids)

    # Derivatives should just be checked without parallel adjoint for now.
    voi = None

    # Extract names of promoted variables
    sys_prom_name = root._sysdata.to_prom_name

    # Extract all parameters, unknowns and connections
    sys_params   = root.params
    sys_unknowns = root.unknowns
    sys_connect  = root.connections

    # Get input keys
    input_str = inputs.keys()

    # Declare matrix for partials
    partials_y = -np.eye(sys_unknowns.vec.size)
    partials_x = np.zeros((sys_unknowns.vec.size, len(input_str)))

    # Check derivative calculations for all comps at every level of the system hierarchy.
    comps = root.components(recurse=True)

    for comp in comps:

        # IndepVarComps are just clutter.
        if isinstance(comp, IndepVarComp):
            continue

        # Setup dictionary to store derivatives
        jac_dict = OrderedDict()

        # Extract data from component
        params = comp.params
        unknowns = comp.unknowns
        resids = comp.resids
        dparams = comp.dpmat[voi]
        dunknowns = comp.dumat[voi]
        dresids = comp.drmat[voi]
        states = comp.states

        # Skip if all of our inputs are unconnected.
        if len(dparams) == 0:
            continue

        # Work with all params that are not pbo.
        param_list = [item for item in dparams if not dparams.metadata(item).get('pass_by_obj')]
        param_list.extend(states)
        unkn_list = [item for item in dunknowns if not dunknowns.metadata(item).get('pass_by_obj')]

        # --------------------------------------------------------------------------
        # COMPUTE DERIVATIVES FOR EACH COMPONENT
        # --------------------------------------------------------------------------

        # Finite difference computation
        if jacobian_mode == 'fd':

            # Finite Difference computation setup
            dresids.vec[:] = 0.0
            root.clear_dparams()
            dunknowns.vec[:] = 0.0

            # Run finite difference for component
            fd_func = comp.fd_jacobian
            jac_dict = fd_func(params, unknowns, resids, use_check=True)

        # Forward derivative computation
        if jacobian_mode == 'fwd':

            # Create all our keys and allocate Jacs
            for p_name in param_list:

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Check dimensions of user-supplied Jacobian
                for u_name in unkn_list:

                    u_size = np.size(dunknowns[u_name])
                    if comp._jacobian_cache:

                        # We can perform some additional helpful checks.
                        if (u_name, p_name) in comp._jacobian_cache:

                            user = comp._jacobian_cache[(u_name, p_name)].shape

                            # User may use floats for scalar jacobians
                            if len(user) < 2:
                                user = (user[0], 1)

                            if user[0] != u_size or user[1] != p_size:
                                msg = "derivative in component '{}' of '{}' wrt '{}' is the wrong size. " + \
                                      "It should be {}, but got {}"
                                msg = msg.format(cname, u_name, p_name, (u_size, p_size), user)
                                raise ValueError(msg)

                    jac_dict[(u_name, p_name)] = np.zeros((u_size, p_size))

            # Forward derivative computation
            for p_name in param_list:

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Send columns of identity
                for idx in range(p_size):
                    dresids.vec[:] = 0.0
                    root.clear_dparams()
                    dunknowns.vec[:] = 0.0

                    dinputs._dat[p_name].val[idx] = 1.0
                    dparams._apply_unit_derivatives()
                    dunknowns._scale_derivatives()
                    comp.apply_linear(params, unknowns, dparams, dunknowns, dresids, 'fwd')
                    dresids._scale_derivatives()

                    for u_name, u_val in dresids.vec_val_iter():
                        jac_dict[(u_name, p_name)][:, idx] = u_val

        # --------------------------------------------------------------------------
        # ASSIGN DERIVATIVES FOR EACH COMPONENT TO GLOBAL MATRIX
        # --------------------------------------------------------------------------

        # Find ouput stored in comp_ders
        comp_ders = jac_dict
        comp_ders_elems = jac_dict.keys()

        # Extract derivatives for each in_out_pair
        for in_out_pair in comp_ders_elems:

            # Extract input and output variables
            o_var, i_var = in_out_pair

            # Extract Jacobian values
            jacobian_vals = np.array(comp_ders[o_var, i_var])

            # Join variable names and find promoted name
            o_var_abs = '.'.join((comp.pathname, o_var))
            i_var_abs = '.'.join((comp.pathname, i_var))

            # Extract promoted variable name
            o_var_pro = sys_prom_name[o_var_abs]
            i_var_pro = sys_prom_name[i_var_abs]

            # States are fine ...
            #if i_var in states:
            #    pass

            # If variable in sys_connect, find other name (var source)
            if i_var_abs in sys_connect:
                i_var_src = sys_connect[i_var_abs][0]
                i_var_pro = sys_prom_name[i_var_src]

            # Extract indices for o_var_pro
            o_start, o_end = sys_unknowns._dat[o_var_pro].slice

            # Check if i_var is input_list
            if i_var not in input_str:

                # Extract index for i_var_pro
                i_start, i_end = sys_unknowns._dat[i_var_pro].slice

                # Assign entry to partials_y
                if jacobian_vals.shape != partials_y[o_start:o_end, i_start:i_end].shape and jacobian_vals.size > 1:
                    jacobian_vals = jacobian_vals[:,0]
                partials_y[o_start:o_end, i_start:i_end] = jacobian_vals

            else:

                # Find index in i_var_src
                in_idx = input_str.index(i_var)

                # Assign entry to partials_x
                if jacobian_vals.shape != partials_x[o_start:o_end, in_idx].shape and jacobian_vals.size > 1:
                    jacobian_vals = jacobian_vals[:,0]
                partials_x[o_start:o_end, in_idx] = jacobian_vals

    sys_unknowns_keys = sys_unknowns.keys()
    sys_unknowns_vals = sys_unknowns.values()

    #Connect dR_dx & dR_dy variables
    for i in range(len(sys_unknowns_keys)):

        # Extract variable name
        var = sys_unknowns_keys[i]

        # Check if variable is in input_str      
        if var in input_str:
            in_idx = input_str.index(var)
            o_start, o_end = sys_unknowns._dat[var].slice
            partials_x[o_start:o_end, in_idx] = 1.0

    # --------------------------------------------------------------------------
    # FIND INPUT AND OUTPUT VARIABLE NAMES
    # --------------------------------------------------------------------------

    # Declare dictionary to store entries of output and input variables
    output_vars = {}
    input_vars  = {}

    # Extract keys and values for unknowns and parameters
    sys_unknowns_keys = sys_unknowns.keys()
    sys_params_keys   = sys_params.keys()
    sys_unknowns_vals = sys_unknowns.values()
    sys_params_vals   = sys_params.values()

    # Print output variable names
    for i in range(len(sys_unknowns_keys)):

        # Extract var and pathname
        var = sys_unknowns_keys[i]
        full_var = sys_unknowns_vals[i]['pathname']

        # If variable in sys_connect, find other name
        if var in sys_connect:
            i_var_src = sys_connect[var][0]
            i_var_pro = sys_prom_name[i_var_src]
        else:
            i_var_pro = var
            
        # Extract indices and assign in dictionary
        out_idx = sys_unknowns._dat[i_var_pro].slice
        output_vars[full_var] = out_idx
        #output_vars[i_var_pro] = out_idx

    # Print input variable names
    for input in input_str:

        # Find variable indices and assign in dictionary
        in_idx = input_str.index(input)
        input_vars[input] = in_idx

    # Extract output variables from sys_unknowns as a vector
    output_val = sys_unknowns.vec

    return partials_y, partials_x, output_vars, input_vars, output_val
