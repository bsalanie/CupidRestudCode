#*******************************************************
#* Copyright (c) 2020 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

import numpy as np
from scipy.optimize import minimize, OptimizeResult, Bounds, LinearConstraint, \
                           NonlinearConstraint, BFGS, SR1
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator
from knitro.numpy import *

#----------------------------------------------------------------------------
#     KNITRO LOAD FUNCTION
#----------------------------------------------------------------------------


def _load_options(kc, options=None):
    '''Load Knitro solver options into KN_context `kc`.
    The options of the scipy functions are `disp` (to display the results)
    and `maxiter` (to limit the number of iteration).
    All the options of Knitro can be set.
    By default, `outlev` is set at 0 (no results are displayed).

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    options: dict, optional
        A dictionary of solver options.
        Generic options :
            `maxiter`: int
            Maximum number of iterations to perform.
            `disp`: bool
            Set to True to print convergence messages.

    '''

    #--- Skip option loading if undefined
    if options is None:
        return
        
    #---- Load the scipy options
    for option in options:
        option = option.lower()

        # Display level option
        # By default in scipy: no results displayed
        if option == 'disp':
            disp = options.get('disp', None)
            if disp:
                KN_set_int_param(kc, "outlev", 2)
            else:
                KN_set_int_param(kc, "outlev", 0)
                
        # Define a maximum number of iterations
        elif option == 'maxiter':
            maxiter = options.get('maxiter', None)
            KN_set_int_param(kc, "maxit", maxiter)
            
        # Pass a knitro option file
        elif option == 'file_options':
            file_ = options.get('file_options', None)
            if file_ is not None:
                KN_load_param_file(kc, file_)

        # Load other knitro options
        elif isinstance(options.get(option), int):
            KN_set_int_param(kc, option, options.get(option))
            
        elif isinstance(options.get(option), float):
            if option == 'tol':
                KN_set_double_param(kc, 'feastol', options.get(option))
            else:
                KN_set_double_param(kc, option, options.get(option))
                
        elif isinstance(options.get(option), str):
            KN_set_char_param(kc, option, options.get(option))
            
        else:
            print("Knitro-Python error: the option %s does not exist in Knitro" \
                  % option)


def _load_constraints(kc, n, constraints=None, HV_linear_operator=False):
    '''Load constraints into KN_context kc.

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    n: int
        Size of the problem (size of x)

    constraints:  {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Available constraints are:
            - class:'scipy.optimize.LinearConstraint'
            - class:'scipy.optimize.NonlinearConstraint'
        Structure containing all the :class:'scipy.optimize.LinearConstraint' or
        class:'scipy.optimize.NonlinearConstraint' related information.

    HV_linear_operator: Bool
        If set to true, it means that hessian of constraints and objective is
        computed with the hessian vector and a LinearOperator object
    '''
    #--- Skip constraint loading if undefined
    if constraints is None:
        return

    #--- Add the constraints to the problem.
    # Check that upper bound and lower bound have same length than variables.
    # if not, same for all.
    for cons in constraints:
        if isinstance(cons, NonlinearConstraint):
            if (np.size(cons.ub) == 1
                    and np.size(cons.fun(np.ones(n))) == 1):
                cons.ub = [cons.ub]
            if (np.size(cons.lb) == 1
                    and np.size(cons.fun(np.ones(n))) == 1):
                cons.lb = [cons.lb]
            else:
                if np.size(cons.ub) == 1:
                    cons.ub = [cons.ub] * n
                if np.size(cons.lb) == 1:
                    cons.lb = [cons.lb] * n

        if isinstance(cons, LinearConstraint):
            if np.size(cons.ub) == 1 and np.size(cons.A) == 1:
                cons.ub = [cons.ub]
            if np.size(cons.lb) == 1 and np.size(cons.A) == 1:
                cons.lb = [cons.lb]
            else:
                if np.size(cons.ub) == 1:
                    cons.ub = np.repeat(cons.ub, n)
                if np.size(cons.lb) == 1:
                    cons.lb = np.repeeat(cons.lb, n)

    #--- Number of constraints
    nb_cons = sum(
        [len(x.ub) if isinstance(x.ub, list) else 1 for x in constraints])
    KN_add_cons(kc, nb_cons)

    idx_cons = 0
    non_linear_constraints = {}
    linear_constraints = []
    for block_constraint in constraints:
        #---- Add constraint
        if isinstance(block_constraint, LinearConstraint):
            _load_linear_constraints(kc, n, block_constraint, idx_cons,
                                     linear_constraints)
        elif isinstance(block_constraint, NonlinearConstraint):
            non_linear_constraints[idx_cons] = {}
            _load_non_linear_constraints(kc, n, block_constraint, idx_cons,
                                         non_linear_constraints,
                                         HV_linear_operator)

        #--- Update the index of the constraint
        if (isinstance(block_constraint.lb, np.ndarray)
                or isinstance(block_constraint.lb, list)):
            idx_cons += len(block_constraint.lb)
        else:
            idx_cons += 1
    nb_linear_constraint = np.sum(linear_constraints)
    nb_non_linear_constraint = len(non_linear_constraints)

    return (nb_cons, nb_linear_constraint, nb_non_linear_constraint,
            non_linear_constraints)


def _load_linear_constraints(kc, n, block_constraint, idx_cons,
                             linear_constraints):
    '''Load linear constraints into KN_context kc.

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    n: int
        Size of the problem (size of x)

    block_constraint: Constraint object
        Structure containing one linear constraint related information
        class:'scipy.optimize.LinearConstraint'

    linear_constraints: list
        List to get the number of linear constraints

    idx_cons: int
        Index of the constraint

    '''
    #---- Force feasibility
    if block_constraint.keep_feasible:
        KN_set_int_param(kc, 'bar_feasible', 1)
        print(
            "'Keep feasible' parameter forces all the constraints (linear and \
             non linear) to remain feasible and not only one constraint. \
             'Keep feasible' parameter is only available for algorithm 1 and 2"
        )

    #---- Constraints bounds
    lower_bound = np.array(block_constraint.lb)
    upper_bound = np.array(block_constraint.ub)
    if isinstance(block_constraint.lb, (np.ndarray, list)):
        index_cons_kn = np.arange(idx_cons,
                                  (len(block_constraint.lb) + idx_cons))
    else:
        index_cons_kn = np.array(idx_cons, ndmin=1)

    KN_set_con_lobnds(kc, indexCons=index_cons_kn, cLoBnds=lower_bound)
    KN_set_con_upbnds(kc, indexCons=index_cons_kn, cUpBnds=upper_bound)

    #---- Constraints structure : Add coefficients for linear constraint.
    if n == 1:  # Problem size is 1
        index_cons_kn = np.array(idx_cons, ndmin=1)
        coefs_kn = np.array(block_constraint.A, ndmin=1)
        index_vars_kn = np.zeros(1)
        KN_add_con_linear_struct(
            kc,
            indexCons=index_cons_kn,
            indexVars=index_vars_kn,
            coefs=coefs_kn)
    elif np.array(block_constraint.A).ndim > 1:
        for cons in range(idx_cons, (len(block_constraint.lb) + idx_cons)):
            A_cons = np.array(block_constraint.A[cons])
            index_vars_kn = np.arange(A_cons.size)
            coefs_kn = A_cons
            linear_constraints += [1]
            KN_add_con_linear_struct(
                kc, indexCons=cons, indexVars=index_vars_kn, coefs=coefs_kn)
    # if only one constraint
    else:
        A_cons = np.array(block_constraint.A)
        index_vars_kn = np.arange(A_cons.size)
        index_cons_kn = np.arange(idx_cons,
                                  len(block_constraint.lb) + idx_cons)
        coefs_kn = A_cons
        KN_add_con_linear_struct(
            kc,
            indexCons=index_cons_kn,
            indexVars=index_vars_kn,
            coefs=coefs_kn)


def _load_non_linear_constraints(kc,
                                 n,
                                 block_constraint,
                                 idx_cons,
                                 non_linear_constraints,
                                 HV_linear_operator=False):
    '''Load nonlinear constraints into KN_context kc.

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    n: int
        Size of the problem (size of x)

    block_constraint: Constraint object
        Structure containing one nonlinear constraint related information.
        class:'scipy.optimize.NonlinearConstraint'

    idx_cons: int
        Index of the constraint

    non_linear_constraints: dict
        Dictionnary containing constraints per indice, with all the elements
        of the non linear constraint (jac, hess, sparsity, indices,...)

    HV_linear_operator: Bool
        If set to true, it means that hessian of constraints and objective is
        computed with the hessian vector and a LinearOperator object
    
    The scipy NonLinearConstraint structure supports options.
    Providing option finite_diff_jac_sparsity means that
    the user is giving the sparsity pattern for the jacobian and/or hessian.
        finite_diff_jac_sparsity =
        {'jacIndexCons': Store nnzj indexes (row) of each nonzero
        in the Jacobian of the constraints,
         'jacIndexVars': Store nnzj index (column) of each nonzero
         in the Jacobian of the constraints ,
         'hessIndexVars1': Store nnzh index of each nonzero
         in the Hessian of the Lagrangian,
         'hessIndexVars2': Store nnzh index of each nonzero
         in the Hessian of the Lagrangian}
    '''
    #---- Force feasibility
    if block_constraint.keep_feasible == True:
        KN_set_int_param(kc, 'bar_feasible', 1)
        print(
            "'Keep feasible' parameter force all the constraints (linear and \
                non linear) to remain feasible and not only keep feasible \
                one constraint")

    #---- Constraints bounds
    if n == 1:  #Problem size is 1
        idx_cons_kn = idx_cons
    else:
        index_cons_kn = np.arange(idx_cons,
                                  (len(block_constraint.lb) + idx_cons))
    lower_bound = np.array(block_constraint.lb, ndmin=1)
    upper_bound = np.array(block_constraint.ub, ndmin=1)

    KN_set_con_lobnds(kc, indexCons=index_cons_kn, cLoBnds=lower_bound)
    KN_set_con_upbnds(kc, indexCons=index_cons_kn, cUpBnds=upper_bound)

    non_linear_constraints[idx_cons]['indexCons'] = index_cons_kn
    non_linear_constraints[idx_cons]['fun'] = block_constraint.fun

    #---- Relative step size for the finite difference approximation
    if block_constraint.finite_diff_rel_step is not None:
        non_linear_constraints[idx_cons]['finite_diff_rel_step'] = \
                                        block_constraint.finite_diff_rel_step

    sparsity = False
    #---- Constraints jacobian
    if block_constraint.finite_diff_jac_sparsity is not None:
        sparsity_matrix = block_constraint.finite_diff_jac_sparsity
        
        #---- Sparsity pattern
        if isinstance(sparsity_matrix, dict):
            sparsity = True
            non_linear_constraints[idx_cons]['sparsity'] = True

            #---- Jacobian
            if 'jacIndexCons' in sparsity_matrix.keys():
                index_cons =  sparsity_matrix['jacIndexCons']
                index_vars =  sparsity_matrix['jacIndexVars']
                non_linear_constraints[idx_cons]['jacIndexCons'] = index_cons
                non_linear_constraints[idx_cons]['jacIndexVars'] = index_vars
                non_linear_constraints[idx_cons]['jac'] = block_constraint.jac
            elif callable(block_constraint.jac):
                non_linear_constraints[idx_cons][
                    'jacIndexCons'] = KN_DENSE_ROWMAJOR
                non_linear_constraints[idx_cons]['jac'] = block_constraint.jac
                
            #---- Hessian
            if 'hessIndexVars1' in sparsity_matrix.keys():
                index_vars1 =  sparsity_matrix['hessIndexVars1']
                index_vars2 =  sparsity_matrix['hessIndexVars2']
                non_linear_constraints[idx_cons]['hessIndexVars1'] = index_vars1
                non_linear_constraints[idx_cons]['hessIndexVars2'] = index_vars2
                non_linear_constraints[idx_cons]['hess'] = block_constraint.hess
            elif callable(block_constraint.hess):
                non_linear_constraints[idx_cons][
                    'hessIndexVars1'] = KN_DENSE_ROWMAJOR
                non_linear_constraints[idx_cons][
                    'hessIndexVars2'] = KN_DENSE_ROWMAJOR
                non_linear_constraints[idx_cons]['hess'] = block_constraint.hess
                
        #---- Finite difference estimation with sparsity pattern
        else:
            sparsity_matrix = block_constraint.finite_diff_jac_sparsity
            if not issparse(sparsity_matrix):
                sparsity_matrix = csr_matrix(sparsity_matrix)
            index_cons = sparsity_matrix.nonzero()[0] + idx_cons
            index_vars = sparsity_matrix.nonzero()[1]
            non_linear_constraints[idx_cons]['jacIndexCons'] = index_cons
            non_linear_constraints[idx_cons]['jacIndexVars'] = index_vars
            non_linear_constraints[idx_cons]['jac'] = None

    else:
        # if jac is not defined it is automatically put to "2-point" and note None
        if (block_constraint.jac == '2-point'):
            # Knitro equivalent is gradopt_forward
            KN_set_int_param(kc, "gradopt", 2)
        elif (block_constraint.jac == '3-point'):
            # Knitro equivalent is gradopt_central
            KN_set_int_param(kc, "gradopt", 3)
        elif (block_constraint.jac == 'cs'):
            KN_set_int_param(kc, "gradopt", 3)
            print('Complete steps scheme does not exist on Knitro, the \
                   Jacobian is calculated with "3-point" or central scheme')
        elif callable(block_constraint.jac):
            non_linear_constraints[idx_cons]['jacIndexCons'] = KN_DENSE_ROWMAJOR
            non_linear_constraints[idx_cons]['jac'] = block_constraint.jac
        else:
            raise ValueError(
                "Knitro-Python error: Jacobian is not callable or \
                the finite difference methods is not defined as 2-point, \
                 3-point or cs")

        #---- Constraints Hessian
        #if hess is not defined it is automatically put to "BFGS"
        if isinstance(block_constraint.hess, BFGS):
            KN_set_int_param(kc, "hessopt", 2)
        elif isinstance(block_constraint.hess, SR1):
            KN_set_int_param(kc, "hessopt", 3)
        elif callable(block_constraint.hess):
            non_linear_constraints[idx_cons]['hessIndexVars1'] = KN_DENSE_ROWMAJOR
            non_linear_constraints[idx_cons]['hess'] = block_constraint.hess
        elif block_constraint.hess in ['2-point', '3-point', 'cs']:
            KN_set_int_param(kc, "hessopt", 2)
            print(
                'Complete steps, 2-point and 3-point schemes do not exist on \
                    Knitro, the Hessian is computed with BFGS scheme')
        else:
            raise ValueError("Knitro-Python error: Hessian is not callable or \
                the finite difference methods is not defined as 2-point, \
                 3-point, cs, BFGS or SR1")


def _load_variables(kc, x0, bounds=None):
    '''Load variables bounds into KN_context kc.

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    x0: ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), where ‘n’ is the
        number of independent variables

    bounds: sequence or Bounds, optional
        Bounds on variables.
        There are two ways to specify the bounds:
            - Instance of 'scipy.optimize._constraints.Bounds' class.
            - Sequence of (min, max) pairs for each element in x.
              None is used to specify no bound.
    '''

    n = x0.size

    #---- Add variables
    KN_add_vars(kc, n)

    #---- Define an initial point.
    KN_set_var_primal_init_values(kc, xInitVals=np.array(x0, ndmin=1))

    #---- Skip bounds definition if undefined
    if bounds is None:
        return

    #---- Set the variables bounds
    if bounds is not None:
        #---- Sequence type
        if isinstance(bounds, (tuple, list)):
            bounds_tuple = np.copy(bounds)
            bounds = \
                {'lb': [-KN_INFINITY if (lb == -np.inf) else lb for (lb, ub) in bounds],
                 'ub': [KN_INFINITY if (ub == np.inf) else ub for (lb, ub) in bounds]}
            #if only one lower and one upper bound are set, it is the same bounds
            #for all the variables
            if np.array(bounds_tuple).ndim == 1:
                (lb, ub) = bounds_tuple
                bounds = {'lb': np.repeat(lb, n), 'ub': np.repeat(ub, n)}
            KN_set_var_lobnds(kc, xLoBnds=np.array(bounds['lb']))
            KN_set_var_upbnds(kc, xUpBnds=np.array(bounds['ub']))

        #---- Bounds type
        elif isinstance(bounds, Bounds):
            if isinstance(bounds.lb, (int, float)):
                bounds.lb = np.repeat(bounds.lb, n)
            if isinstance(bounds.ub, (int, float)):
                bounds.ub = np.repeat(bounds.ub, n)
            if 'lb' in bounds.__dict__.keys():
                bounds.lb = \
                    [-KN_INFINITY if (lb == -np.inf) else lb for lb in bounds.lb]
                KN_set_var_lobnds(kc, xLoBnds=np.array(bounds.lb))
            if 'ub' in bounds.__dict__.keys():
                bounds.ub = \
                    [KN_INFINITY if (ub == np.inf) else ub for ub in bounds.ub]
                KN_set_var_upbnds(kc, xUpBnds=np.array(bounds.ub))


def _load_objective(kc,
                    fun,
                    jac=None,
                    hess=None,
                    hessp=None,
                    constraints=None,
                    HV_linear_operator=False,
                    **options):
    '''Load objective into objective dictionary.

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    fun: callable
        The objective function to be minimized.
            fun(x, *args) -> float
        where x is an 1-D array with shape (n,) and args is a tuple of the
        fixed parameters needed to completely specify the function.

    jac: {callable, bool}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:
            jac(x, *args) -> array_like, shape (n,)
        where x is an array with shape (n,) and args is a tuple with the fixed
        parameters. If jac is a Boolean and is True, fun is assumed to return the
        gradient along with the objective function
        (class:scipy.optimize.MemoizeJac). If False, the gradient will be
        estimated using ‘2-point’ finite difference estimation.

    hess:  {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. If it is callable, it should
        return the Hessian matrix:
            hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)
        where x is a (n,) ndarray and args is a tuple with the fixed parameters.
        The keywords {‘2-point’, ‘3-point’, ‘cs’} select a finite difference
        scheme for numerical estimation. Or, objects implementing
        HessianUpdateStrategy interface can be used to approximate the Hessian.

    hessp: callable, optional
        Hessian of objective function times an arbitrary vector p. Only one of
        hessp or hess needs to be given. If hess is provided, then hessp will be
        ignored. hessp must compute the Hessian times an arbitrary vector:
            hessp(x, p, *args) ->  ndarray shape (n,)
        where x is a (n,) ndarray, p is an arbitrary vector with dimension (n,)
        and args is a tuple with the fixed parameters.

    constraints:  {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Available constraints are:
            - class:'scipy.optimize.LinearConstraint'
            - class:'scipy.optimize.NonlinearConstraint'
        Structure containing all the :class:'scipy.optimize.LinearConstraint' or
        class:'scipy.optimize.NonlinearConstraint' related information.

    options: dict, optional
        A dictionary of solver options.
        Generic options :
            `maxiter` : int
            Maximum number of iterations to perform.
            `disp` : bool
            Set to True to print convergence messages.
        Additional option :
            Example sparsity pattern with form :
            `sparsity` = {'jac_sparse': ndarray, 'hess_sparse': ndarray}
            where jac_sparse is the matrix of the jacobian without the
            coefficient but with 1 if the an element is non zero and zero
            otherwise

    HV_linear_operator: Bool
        If set to true, it means that hessian of constraints and objective is
        computed with the hessian vector and a LinearOperator object

    Returns
    -------
    objective: dict
        A dictionnary with fun, jac, hess and the indices according to the
        sparsity pattern if it is given
        
    Options can be added in the scipy.optimize.minimize function.
    Defining option 'sparse' means the user is providing the objective 
    gradient and/or hessian sparsity pattern .
    The user should provide a dictionnary
    'sparse':{'objGradIndexVars': The n_vars nonzero indices of 
              the objective gradient,
              'hessIndexVars1': Store nnzh index of each nonzero
              in the Hessian of the Lagrangian,
              'hessIndexVars2': Store nnzh index of of each 
              nonzero in the Hessian of the Lagrangian }'''

    objective = {}

    # #---- Add a callback function "callbackEvalF" to evaluate the nonlinear
    objective['fun'] = fun
    objective['jac'] = jac
    objective['hess'] = hess
    objective['hessp'] = hessp
   
    sparsity = False
    #---- Sparsity pattern
    if 'sparse' in options:
        sparsity = True
        option_sparse = options['sparse']
        objective['sparsity'] = True

        #---- Jacobian sparse
        # if 'jac_sparse' in option_sparse.keys():
        #     jac_sparse = coo_matrix(options['sparse']['jac_sparse'])
        if 'objGradIndexVars' in option_sparse.keys():
            objective['objGradIndexVars'] = options['sparse']['objGradIndexVars']
        else:
            objective['objGradIndexVars'] = KN_DENSE

        #---- Hessian sparse
        # if 'hess_sparse' in option_sparse.keys():
        #     hess_sparse = coo_matrix(options['sparse']['hess_sparse'])
        #     objective['hessIndexVars1'] = hess_sparse.row
        #     objective['hessIndexVars2'] = hess_sparse.col
        #     objective['sparsity'] = True
        if 'hessIndexVars1' in option_sparse.keys():
            objective['hessIndexVars1'] = options['sparse']['hessIndexVars1']
            objective['hessIndexVars2'] = options['sparse']['hessIndexVars2']
        else:
            objective['hessIndexVars2'] = KN_DENSE_ROWMAJOR
    else:
        #---- Evaluate the jacobian of nonlinear
        if jac is not None:
            if (jac == '2-point'):
                KN_set_int_param(kc, "gradopt", 2)
            elif (jac == '3-point'):
                KN_set_int_param(kc, "gradopt", 3)
            elif (jac == False):
                # if false, it computes the gradient with 2-points algorithm
                KN_set_int_param(kc, "gradopt", 2)
            elif (jac == 'cs'):
                KN_set_int_param(kc, "gradopt", 3)
                print('Complete steps scheme does not exist on Knitro, the \
                       Jacobian is calculated with "3-point" or central scheme'
                      )
            else:
                objective['objGradIndexVars'] = KN_DENSE

        #---- Evaluate the hessian of nonlinear
        if hess is not None or hessp is not None:
            # default
            if isinstance(hess, BFGS):
                KN_set_int_param(kc, "hessopt", 2)
            if isinstance(hess, SR1):
                KN_set_int_param(kc, "hessopt", 3)
            # If the user select on scipy 2-point, 3-point or cs algorithm, the
            # hessian will be computed with BFGS algorithm. The 2-point, 3-point
            # or cs algorithm do not exist on Knitro.
            elif hess in {'2-point', '3-point', 'cs'}:
                print(
                    'Complete steps, 2-point and 3-point schemes do not exist on \
                    Knitro, the Hessian is computed with BFGS scheme')
                KN_set_int_param(kc, "hessopt", 2)
            #---- Hessian product vector
            # if there are nonlinear constraints, they must be LinearOperator
            # objects
            elif (hessp is not None) and (hess is None):
                objective['hessp'] = hessp
                KN_set_int_param(kc, "hessopt", 5)

            objective['hessIndexVars1'] = KN_DENSE_ROWMAJOR

    return objective


def _hessian_vector_check(hess=None, hessp=None, constraints=None):
    '''
    Check if the Hessian given in the constraints and the objective are both
    LinearOperator object

    Parameters
    ----------

    hess :  {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. If it is callable, it should
        return the Hessian matrix:
            hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)
        where x is a (n,) ndarray and args is a tuple with the fixed parameters.
        The keywords {‘2-point’, ‘3-point’, ‘cs’} select a finite difference
        scheme for numerical estimation. Or, objects implementing
        HessianUpdateStrategy interface can be used to approximate the Hessian.

    hessp: callable, optional
        Hessian of objective function times an arbitrary vector p. Only one of
        hessp or hess needs to be given. If hess is provided, then hessp will be
        ignored. hessp must compute the Hessian times an arbitrary vector:
            hessp(x, p, *args) ->  ndarray shape (n,)
        where x is a (n,) ndarray, p is an arbitrary vector with dimension (n,)
        and args is a tuple with the fixed parameters.

    constraints:  {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Available constraints are:
            - class:'scipy.optimize.LinearConstraint'
            - class:'scipy.optimize.NonlinearConstraint'
        Structure containing all the :class:'scipy.optimize.LinearConstraint' or
        class:'scipy.optimize.NonlinearConstraint' related information.
    '''
    #---- Check if the constraint hessian is an hessian product vector
    if constraints is not None:
        constraint_nonlinear = \
            [cons for cons in constraints if isinstance(cons, NonlinearConstraint)]
        nb_constraints_linear_op = \
            sum([isinstance(cons.hess, LinearOperator)  \
                 for cons in constraint_nonlinear \
                 if (cons.hess is not None)])

        if len(constraint_nonlinear) > 0 and hessp is not None:
            raise ValueError("Knitro-Python error: hessian vector \
                can be used with hessp function if there are nonlinear \
                constraints")

        elif (len(constraint_nonlinear) > 0
              and nb_constraints_linear_op != len(constraint_nonlinear)
              and isinstance(hess, LinearOperator)):
            raise ValueError("Knitro-Python error: hessian \
                        constraints must be a LinearOperator")

        elif (nb_constraints_linear_op == len(constraint_nonlinear)
              and isinstance(hess, LinearOperator)):
            # Active hessian vector product mode on Knitro
            KN_set_int_param(kc, "hessopt", 5)
            return True
    return False


def _compute_hessian(current_hessian, x, evalResult, sigma=1, sparsity=False):
    ''' Compute Hessian of the Lagrangian (for nonlinear constraints and
    objective)

    Parameters
    ----------
    current_hessian: ndarray, sparse matrix, linear operator
        Hessian matrix compute at the current point

    x:  ndarray, shape(n,)
        Current point

    evalResult : callback object

    sigma: int, float
        Sigma of the Lagrangian

    sparsity: bool
        If sparsity is True, it means that the sparsity pattern is given and
        we can use it to compute the matrix.

    '''
    #---- The sparsity pattern is given
    if sparsity:
        evalResult.hess = (current_hessian.data).copy()


    #---- Problem of size 1
    elif isinstance(x, (int, float)) or len(x) == 1:
        evalResult.hess = np.array(sigma * current_hessian, ndmin=1, copy=True)
    #---- Sparse matrix
    elif issparse(current_hessian):
        if sparsity:
            #Check if the hessian is triangular
            hessian_tri = current_hessian.toarray()
            row, col = current_hessian.nonzero()
            if not (row <= col).all():
                print('The hessian is not triangular')
                current_hessian = current_hessian.toarray()
                current_hessian = np.triu(current_hessian, k=0)
                current_hessian = csc_matrix(current_hessian).data
                current_hessian = current_hessian * sigma
                evalResult.hess = current_hessian.copy()
            else:
                current_hessian = current_hessian.data
                evalResult.hess = (current_hessian * sigma).copy()
        else:
            # If the sparsity pattern is not given we have to give the zeroes
            print('The sparsity pattern is not given. \
                   The hessian will be densified')
            current_hessian = np.array(current_hessian.toarray())
            current_hessian = \
                np.array(current_hessian[np.triu_indices(len(current_hessian))],
                copy=True)
            evalResult.hess = current_hessian * sigma

    #---- Linear operator
    elif isinstance(current_hessian, LinearOperator):
        current_hessian = current_hessian.matmat(np.eye(len(x)))
        #triangular matrix
        current_hessian = \
            np.array(current_hessian[np.triu_indices(len(current_hessian))])
        current_hessian = current_hessian * sigma
        evalResult.hess = np.array([item if isinstance(item,(int,float)) \
                                    else item[0] for item in current_hessian],
                                    copy=True)
    #---- Dense array
    elif isinstance(current_hessian, (np.ndarray, np.generic, list)):
        current_hessian = \
            np.array(current_hessian[np.triu_indices(len(current_hessian))])
        evalResult.hess = (current_hessian * sigma).copy()
    else:
        raise ValueError("Knitro-Python error: Hessian must be an \
            numpy.ndarray or csc_matrix or LinearOperator")




def _make_all_callbacks(kc, objective, non_linear_constraints, m,
                        nb_linear_constraint, callback):
    '''Load objective and non linear constraints into KN_context kc

    Parameters
    ----------
    kc: KN_context
        Knitro solver context object to be updated

    objective: dict
        A dictionnary with fun, jac, hess and the indices according to the
        sparsity pattern if it is given

    non_linear_constraints: dict
        Dictionnary containing constraints per indice, with all the elements
        of the non linear constraint (jac, hess, sparsity, indices,...)

    m: int
        Number of constraints

    nb_linear_constraint: int
        Number of linear constraints

    callback: callable, optional
        Called after each iteration.
            callback(xk)
        where xk is the current parameter vector.
    '''

    # Evaluate nonlinear objective
    def callbackEvalF(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalF incorrectly called with eval type %d" \
                % evalRequest.type)
            return -1
        x = evalRequest.x

        evalResult.obj = objective['fun'](x).copy()
        return 0

    # Evaluate nonlinear constraints
    def callbackEvalFCons(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print("*** callbackEvalF incorrectly called with eval type %d" \
                % evalRequest.type)
            return -1
        x = evalRequest.x

        for idx in non_linear_constraints:
            cons = non_linear_constraints[idx]
            if nb_linear_constraint == 0:
                idx_range = cons["indexCons"]
            else:
                idx_range = cons["indexCons"] - nb_linear_constraint
            evalResult.c[idx_range] = cons['fun'](x).copy()
        return 0

    # Evaluate gradient of nonlinear objective
    def callbackEvalG(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print("*** callbackEvalG incorrectly called with eval type %d" \
                 % evalRequest.type)
            return -1
        x = evalRequest.x

        # if issparse(objective['jac'](x)) and sparsity_obj:
        if sparsity_obj:
            grad = objective['jac'](x).data
        else :
            grad = objective['jac'](x)
        # print(grad)
        evalResult.objGrad = grad.copy()
        return 0

    # Evaluate gradient of nonlinear constraints
    def callbackEvalGCons(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print("*** callbackEvalG incorrectly called with eval type %d" \
                 % evalRequest.type)
            return -1
        x = evalRequest.x

        for idx in non_linear_constraints:
            cons = non_linear_constraints[idx]
            userJac = cons['jac'](x)

            if nb_linear_constraint != 0:
                idx = idx - nb_linear_constraint

            #---- Sparse matrix
            if sparsity_cons:
                userJac = userJac.data
                index_1 = len(np.array(userJac).flatten()) * idx
                index_2 = len(np.array(userJac).flatten()) * (idx + 1)
                index = np.arange(index_1, index_2)
                evalResult.jac[index] = userJac.copy()
            if issparse(userJac):
                userJac = userJac.toarray()
                print('Warning: No sparsity is provided. A densed matrix will \
                       be used.')
                index_1 = len(np.array(userJac).flatten()) * idx
                index_2 = len(np.array(userJac).flatten()) * (idx + 1)
                index = np.arange(index_1, index_2)
                evalResult.jac[index] = np.array(
                    userJac, copy=True).flatten()
        #---- Dense array
            elif (isinstance(userJac,
                             (np.generic, np.ndarray, list, int, float))
                  or len(x) == 1):
                # here jac is a matrix
                # we put all the rows into a list to make this list flatt (and not
                # have list of lists)
                index_1 = len(np.array(userJac).flatten()) * idx
                index_2 = len(np.array(userJac).flatten()) * (idx + 1)
                index = np.arange(index_1, index_2)
                evalResult.jac[index] = np.array(
                    userJac, ndmin=1, copy=True).flatten()
            else:
                raise ValueError("Knitro-Python error: gradient must be an \
                        numpy.ndarray ")

        evalResult.jac = evalResult.jac.flatten().copy()
        return 0

    # Evaluate the hessian of the nonlinear objective.
    def callbackEvalHess(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALH and \
           evalRequest.type != KN_RC_EVALH_NO_F \
           and evalRequest.type != KN_RC_EVALHV:
            print("*** callbackEvalH incorrectly called with eval type %d" \
             % evalRequest.type)
            return -1
        x = evalRequest.x

        # Evaluate the hessian of the nonlinear objective.
        # Note: Since the Hessian is symmetric, we only provide the
        #       nonzero elements in the upper triangle (plus diagonal).
        #       These are provided in row major ordering as specified
        #       by the setting KN_DENSE_ROWMAJOR in "KN_set_cb_hess()".
        # Note: The Hessian terms for the quadratic constraints
        #       will be added internally by Knitro to form
        #       the full Hessian of the Lagrangian.

        #---- Hessian
        if evalRequest.type == KN_RC_EVALH:
            sigma = evalRequest.sigma
            current_hessian = objective['hess'](x)
            _compute_hessian(current_hessian, x, evalResult, sigma, sparsity=sparsity)
        #---- Hessian product vector
        if evalRequest.type == KN_RC_EVALHV:
            vec = evalRequest.vec
            if objective['hessp'] is not None:
                current_hessian = objective['hessp'](x, vec)
                if isinstance(current_hessian, (np.ndarray, np.generic)):
                    evalResult.hessVec = np.array(current_hessian, copy=True)
                else:
                    raise ValueError(
                        "Knitro-Python error: Hessian Vector must \
                                      be an numpy.ndarray")
            elif HV_linear_operator:
                current_hessian = objective['hess'](x).matvec(vec)
                evalResult.hessVec = current_hessian.copy()
            else:
                raise ValueError("Knitro-Python error: Hessian Vector must \
                                      be an numpy.ndarray or a LinearOperator")

        return 0

    # Evaluate the hessian of the nonlinear constraints.
    def callbackEvalHessCons(kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALH \
           and evalRequest.type != KN_RC_EVALH_NO_F \
           and evalRequest.type != KN_RC_EVALHV:
            print("*** callbackEvalH incorrectly called with eval type %d" \
                  % evalRequest.type)
            return -1
        x = evalRequest.x

        # Note: Since the Hessian is symmetric, we only provide the
        #       nonzero elements in the upper triangle (plus diagonal).
        #       These are provided in row major ordering as specified
        #       by the setting KN_DENSE_ROWMAJOR in "KN_set_cb_hess()".
        # Note: The Hessian terms for the quadratic constraints
        #       will be added internally by Knitro to form
        #       the full Hessian of the Lagrangian.

        lambda_ = evalRequest.lambda_

        for idx in non_linear_constraints:
            cons = non_linear_constraints[idx]
            idx_range = cons["indexCons"]
            current_hessian = cons['hess'](x, lambda_)
            if evalRequest.type == KN_RC_EVALHV:
                if HV_linear_operator:
                    current_hessian = cons['hess'](x).matvec(vec)
                    evalResult.hessVec = current_hessian.copy()
                else:
                    raise ValueError(
                        "Knitro-Python error: Hessian Vector must \
                                          be an numpy.ndarray or a LinearOperator"
                    )
            else:
                _compute_hessian(current_hessian, x, evalResult, sparsity=sparsity)

        return 0

    def callbackNewPoint(kc, x, lambda_, userParams):
        # If callback is defined by the usier, this function is called at each
        # iteration.
        # The function callback is applied at each iteration at the current
        # variables and the current objective.
        # For example, this can useful to print objective and variables at each
        # iteration to follow the evolution.
        callback(x)
        return 0

    sparsity_cons = False

    #---- Add a callback function defined by the usear and called at each iteration
    if callback is not None:
        KN_set_newpt_callback(kc, callbackNewPoint)

    #---- Add a callback function "callbackEvalF" to evaluate the nonlinear
    # objective
    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=callbackEvalF)

    #---- Add a callback function "callbackEvalFCons" to evaluate the nonlinear
    # constraints
    num_constraints = len(non_linear_constraints)
    if num_constraints > 0:
        idx_cons = np.arange(int(nb_linear_constraint), int(m))
        cbC = KN_add_eval_callback(
            kc, indexCons=idx_cons, funcCallback=callbackEvalFCons)

    #---- Get info from the non linear constraints
    jacIndexCons = KN_DENSE_ROWMAJOR
    hessIndexVars1 = KN_DENSE_ROWMAJOR
    for idx in non_linear_constraints:
        cons = non_linear_constraints[idx]
        #---- Finite difference
        if 'finite_diff_rel_step' in cons.keys():
            KN_set_cb_relstepsizes(
                kc, cb, xRelStepSizes=cons['finite_diff_rel_step'])
        #---- Sparsity pattern : get the indices
        if 'sparsity' in cons.keys():
            if cons['sparsity']:
                sparsity_cons = True
                jacIndexCons = cons['jacIndexCons']
                jacIndexVars = cons['jacIndexVars']

                if 'hess' in cons.keys():
                    cons_hessIndexVars1 = cons['hessIndexVars1']
                    cons_hessIndexVars2 = cons['hessIndexVars2']

        #---- Get the index for the jacobian
        if 'jac' in cons.keys() and not sparsity_cons:
            jacIndexCons = cons['jacIndexCons']

        #---- Get the index for the hessian
        if 'hess' in cons.keys() and not sparsity_cons:
            hessIndexVars1 = cons['hessIndexVars1']

    #---- Sparsity pattern for the objective
    sparsity_obj = False
    if 'sparsity' in objective.keys() and objective['sparsity']:
        sparsity_obj = True

    # Sparsity is used only if constraints and objectives have the sparsity pattern
    sparsity = sparsity_obj and sparsity_cons

    #---- Get the index for the objective gradient
    if objective['jac'] is not None:
        if 'objGradIndexVars' not in objective.keys():
            objective['objGradIndexVars'] = KN_DENSE
        objGradIndexVars = objective['objGradIndexVars']
        #---- Add a callback function to evaluate the  derivative for nonlinear
        # objective
        KN_set_cb_grad(
            kc,
            cb,
            objGradIndexVars=objGradIndexVars,
            gradCallback=callbackEvalG)

    #---- Add a callback function to evaluate the derivative for nonlinear
    # constraints
    # get the number of gradient given for the non linear constraints
    num_grad = len([True  for i in non_linear_constraints.keys() \
                      if 'grad' in non_linear_constraints[i].keys() \
                      if non_linear_constraints[i]['jac']!= None])
    if num_constraints > 0 and num_grad == num_constraints:
        if sparsity_cons:
            KN_set_cb_grad(
                kc,
                cbC,
                jacIndexCons=jacIndexCons,
                jacIndexVars=jacIndexVars,
                gradCallback=callbackEvalGCons)
        else:
            KN_set_cb_grad(
                kc,
                cbC,
                jacIndexCons=jacIndexCons,
                gradCallback=callbackEvalGCons)

    #---- Get the index for the objective hessian
    # get the number of hessian given for the non linear constraints
    num_hess = len([True  for i in non_linear_constraints.keys() \
                  if 'hess' in non_linear_constraints[i].keys() \
                  if non_linear_constraints[i]['hess']!= None])
    if objective['hess'] is not None:
        if 'hessIndexVars1' not in objective.keys():
            objective['hessIndexVars1'] = KN_DENSE_ROWMAJOR

        #---- Add a callback function to evaluate the second derivative for nonlinear
        if sparsity_obj:
            KN_set_cb_hess(
                kc,
                cb,
                hessIndexVars1=objective['hessIndexVars1'],
                hessIndexVars2=objective['hessIndexVars2'],
                hessCallback=callbackEvalHess)
            if num_constraints > 0 and num_hess == num_constraints and sparsity:
                KN_set_cb_hess(
                    kc,
                    cbC,
                    hessIndexVars1=cons_hessIndexVars1,
                    hessIndexVars2=cons_hessIndexVars2,
                    hessCallback=callbackEvalHessCons)
        else:
            KN_set_cb_hess(
                kc,
                cb,
                hessIndexVars1=objective['hessIndexVars1'],
                hessCallback=callbackEvalHess)
            if num_constraints > 0 and num_hess == num_constraints:
                KN_set_cb_hess(
                    kc,
                    cbC,
                    hessIndexVars1=objective['hessIndexVars1'],
                    hessCallback=callbackEvalHessCons)


#----------------------------------------------------------------------------
#     KNITRO SCIPY MINIMIZE FUNCTION
#----------------------------------------------------------------------------
def kn_minimize(fun,
                x0,
                args=(),
                jac=None,
                hess=None,
                hessp=None,
                bounds=None,
                constraints=(),
                callback=None,
                **options):
    '''
    Solve a minimization problem using Artelys Knitro.

    This function is designed to use scipy.optimize.minimize using the
    solver Artelys Knitro.
    Note that the user should try to provide as much
    information as possible as it usualy leads to better performance
    in Knitro.

    Parameters
    ----------
    fun: callable
        The objective function to be minimized.
            fun(x, *args) -> float
        where x is an 1-D array with shape (n,) and args is a tuple of the
        fixed parameters needed to completely specify the function.

    x0: ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), where ‘n’ is the
        number of independent variables.

    args: tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (fun, jac and hess functions).

    jac: {callable, bool}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:
            jac(x, *args) -> array_like, shape (n,)
        where x is an array with shape (n,) and args is a tuple with the fixed
        parameters. If jac is a Boolean and is True, fun is assumed to return the
        gradient along with the objective function
        (class:scipy.optimize.MemoizeJac). If False, the gradient will be
        estimated using ‘2-point’ finite difference estimation. For Knitro,
        '2-point' corresponds to a forward finite-difference approximation
        of the objective and constraint gradients

    hess :  {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. If it is callable, it should
        return the Hessian matrix:
            hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)
        where x is a (n,) ndarray and args is a tuple with the fixed parameters.
        The keywords {‘2-point’, ‘3-point’, ‘cs’} select a finite difference
        scheme for numerical estimation. Or, objects implementing
        HessianUpdateStrategy interface can be used to approximate the Hessian.
        For Knitro, '2-point' corresponds to a forward finite-difference
        approximation of the objective and constraint gradients.
        For Knitro, '3-point' corresponds to a centered finite-difference
        approximation of the objective and constraint gradients.

    hessp: callable, optional
        Hessian of objective function times an arbitrary vector p. Only one of
        hessp or hess needs to be given. If hess is provided, then hessp will be
        ignored. hessp must compute the Hessian times an arbitrary vector:
            hessp(x, p, *args) ->  ndarray shape (n,)
        where x is a (n,) ndarray, p is an arbitrary vector with dimension (n,)
        and args is a tuple with the fixed parameters.

    bounds: sequence or Bounds, optional
        Bounds on variables.
        There are two ways to specify the bounds:
            - Instance of 'scipy.optimize._constraints.Bounds' class.
            - Sequence of (min, max) pairs for each element in x. None is used
              to specify no bound.

    constraints:  {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Available constraints are:
            - class:'scipy.optimize.LinearConstraint'
            - class:'scipy.optimize.NonlinearConstraint'
        Structure containing all the :class:'scipy.optimize.LinearConstraint' or
        class:'scipy.optimize.NonlinearConstraint' related information.

    callback: callable, optional
        Called after each iteration.
            callback(xk)
        where xk is the current parameter vector.

    options: dict, optional
        A dictionary of solver options.
        Generic options :
            `maxiter` : int
            Maximum number of iterations to perform.
            `disp` : bool
            Set to True to print convergence messages.

    Returns
    -------
    res: OptimizeResult object
        The optimization result represented as a
        class:'scipy.optimize.OptimizeResult' object. Important attributes are:
        x the solution array, success a Boolean flag indicating if the optimizer
        exited successfully and message which describes the cause of the
        termination.

    '''
    success = False

    with KN_new() as kc:
        #---- Initialize Knitro with the problem definition.

        # Add the variables and set their bounds.
        # Note: any unset lower bounds are assumed to be
        # unbounded below and any unset upper bounds are
        # assumed to be unbounded above.
        n = x0.size
        _load_options(kc, options)
        _load_variables(kc, x0, bounds)

        #Check if have to use hessian vector product for constraints and objective
        HV_linear_operator = _hessian_vector_check(hess, hessp, constraints)

        objective = _load_objective(kc, fun, jac, hess, hessp, constraints,
                                    HV_linear_operator, **options)
        (nb_cons, nb_linear_constraint, nb_non_linear_constraint,
                   non_linear_constraints )= \
            _load_constraints(kc, n, constraints, HV_linear_operator)

        #Load objective and non linear constraints into KN_context kc
        _make_all_callbacks(kc, objective, non_linear_constraints, nb_cons,
                            nb_linear_constraint, callback)

        #---- Solve the problem.
        # Return status codes are defined in "knitro.py" and described
        # in the Knitro manual.
        nStatus = KN_solve(kc)

        # An example of obtaining solution information.
        nStatus, objSol, x, lambda_ = KN_get_solution(kc)
        tcpu = KN_get_solve_time_cpu(kc)
        treal = KN_get_solve_time_real(kc)
        print("Total CPU time           = %f" % tcpu)
        print("Total real time          = %f" % treal)

        nit = KN_get_number_iters(kc)
        nfev = KN_get_number_FC_evals(kc)
        njev = KN_get_number_GA_evals(kc)
        nhev = KN_get_number_H_evals(kc)

        #--- Same print as in scipy.optimize.minimize
        if nStatus == 0:
            success = True
            print("Optimization terminated successfully.", )
        elif nStatus in [-199, -100]:
            print("A feasible approximate solution was found.")
        elif nStatus in [-299, -200]:
            print("Knitro terminated at an infeasible point.")
        elif nStatus in [-301, -300]:
            print("The problem was determined to be unbounded.")
        elif nStatus in [-499, -400]:
            print("Knitro terminated because it reached a pre-defined limit.")
        elif nStatus in [-599, -500]:
            print(
                "Knitro terminated with an input error or some non-standard error."
            )

        print("            Current function value:", objSol)
        print("            Iterations:", nit)
        print("            Function evaluations:", nfev)
        print("            Gradient evaluations:", njev)
        print("Solution:", [xi for xi in x])

    return OptimizeResult(
        fun=objSol,
        x=x,
        success=success,
        status=nStatus,
        nit=nit,
        nfev=nfev,
        njev=njev,
        nhev=nhev)
