#*******************************************************
#* Copyright (c) 2020 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++  Artelys Knitro 12.2 Python API
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''Usage of Knitro Python Single call function is described here.

Alternatively to the Python interface of the Knitro callable
library functions, the user has the possibility to directly pass
an optimization problem structure to Knitro and solve it in a
single call fashion using the :meth:`optimize` method.

The definition of the problem structure is allowed using dedicated
Knitro structures: :class:`.Variables`, :class:`.Objective`,
:class:`.Constraints`, :class:`.ComplementarityConstraints` and
:class:`.Callback`.
'''

from .wrapper import *


#----------------------------------------------------------------------------
#     KNITRO OBJECTS
#----------------------------------------------------------------------------

class Variables:
    '''Store mathematical definition of the variables.

    nV: int
        Number of variables.

    xTypes: array, optional
        The type of the variables (e.g. :envvar:`KN_VARTYPE_CONTINUOUS`,
        :envvar:`KN_VARTYPE_BINARY`, :envvar:`KN_VARTYPE_INTEGER`).
        There are two ways to specify the types:

        1. array, shape (nV, 1): define all the variables type.

        2. array, shape (2, n_vars): define the type for a subset of `n_vars`
        variables using two arrays. The first array defines
        the variables index and the second one their types. For example,
        variables 0 and 2 are defined as binary::

            xTypes=[[0, 2], [KN_VARTYPE_BINARY, KN_VARTYPE_BINARY]]

        If not set, variables are assumed to be continuous by default.

    xLoBnds: array, optional
        The lower bounds of the variables. There are two
        ways to specify the lower bounds:

        1. array, shape (nV, 1): define all the variables lower bound.

        2. array, shape (2, n_vars): define the lower bound for `n_vars`
        variables using two arrays. The first
        array defines the variables index and the second one their
        lower bounds. For examples, variables 0 and 2 have their lower bounds
        set to 0::

            xLoBnds=[[0, 2], [0, 0]]

        If not set, lower bounds are assumed to be :envvar:`-KN_INFINITY`.

    xUpBnds: array, optional
        The upper bounds of the variables. There are two
        ways to specify the upper bounds:

        1. array, shape (nV, 1): define all the variables upper bound.

        2. array, shape (2, n_vars): define the upper bound for `n_vars`
        variables using two arrays. The first
        array defines the variables index and the second one their
        upper bounds. For example, variables 0 and 2 have their upper bounds
        set to 10::

            xUpBnds=[[0, 2], [10, 10]]

        If not set, upper bounds are assumed to be :envvar:`KN_INFINITY`.

    xInitVals: array, optional
        The initial primal values of the variables. There are two
        ways to specify the initial values:

        1. array, shape (nV, 1): define all the variables initial values.

        2. array, shape (2, n_vars): define the initial values for `n_vars`
        variables using two arrays. The first
        array defines the variables index and the second one their
        values. For example, variables 0 and 2 are initially set to -1 and 3::

            xInitVals=[[0, 2], [-1, 3]]

        If not set, variables may be initialized as 0 or initialized
        by Knitro based on some initialization strategy
        (perhaps determined by a user option).

    lambdaInitVals: array, optional
        The initial dual values of the variables (i.e. the Lagrange
        multipliers corresponding to the bounded variables).
        There are two ways to specify the initial dual values:

        1. array, shape (nV, 1): defines all the variables initial values.

        2. array, shape (2, n_vars): defines the initial values for `n_vars`
        variables only using two arrays. The first
        array defines the variables index and the second one their
        values. For example, variables 0 and 2 are initially set to -1 and 3::

            lambdaInitVals=[[0, 2], [-1, 3]]

        If not set, dual variables may be initialized as 0 or initialized
        by Knitro based on some initialization strategy
        (perhaps determined by a user option).

    xScale: array, optional
        Define scaling of variables to perform a linear
        scaling around specified centering values::

            x[i] = xScaleFactors[i] * xScaled[i] + xScaleCenters[i]

        These scaling factors should try to represent the `typical`
        values of the `x` variables so that the scaled variables
        (`xScaled`) used internally by Knitro are close to one. The values
        for xScaleFactors should be positive. If a non-positive
        value is specified, that variable will not be scaled.
        There are two ways to specify the scaling value:

        1. array, shape (2, nV): define all the variables xScaleFactors and
        xScaleCenters as two arrays.

        2. array, shape (3, n_vars): define the scaling values for `n_vars`
        variables only using three arrays. The first
        array defines the variables index, the second one their scaling
        factor and the third one their scaling center. For example,
        variables 0 and 2 only are scaled::

            xScale=[[0, 2], [2, 2], [0, 0]]

    xHonorBnds: array, optional
        Indicates whether to enforce satisfaction of simple
        variable bounds throughout the optimization (e.g
        :envvar:`KN_HONORBNDS_NO`, :envvar:`KN_HONORBNDS_ALWAYS`,
        :envvar:`KN_HONORBNDS_INITPT`). There are two ways to specify
        the honor bounds values:

        1. array, shape (nV, 1): defines all the variables honor bounds.

        2. array, shape (2, n_vars): defines the honor bounds for `n_vars`
        variables only using two arrays. The first
        array defines the variables index and the second one their
        values. For example, we enforce that variables 0 and 2 satisfy their
        bounds at initial point::

            xHonorBnds=[[0, 2], [KN_HONORBNDS_INITPT, KN_HONORBNDS_INITPT]]

    xProperties: array, optional
        Specify some properties of the variables. Currently this routine
        is only used to mark variables as linear (e.g :envvar:`KN_VAR_LINEAR`).
        There are two ways to specify the properties:

        1. array, shape (nV, 1): define all the variables properties.

        2. array, shape (2, n_vars): define the properties for `n_vars`
        variables using two arrays. The first
        array defines the variables index and the second one their
        properties. For example, we mark variables 0 and 2 as linear::

            xProperties=[[0, 2], [KN_VAR_LINEAR, KN_VAR_LINEAR]]

        Variables are assumed to be nonlinear variables by default.

    xNames: array, optional
        Specify the name of the variables so that Knitro can internally
        print out these names. There are two ways to specify the
        names of the variables:

        1. array, shape (nV, 1): define all the variables names.

        2. array, shape (2, n_vars): define the names for `n_vars`
        variables using two arrays. The first
        array defines the variables index and the second one their
        names. For example, we name variables 0 and 2::

            xNames=[[0, 2], ['var0', 'var2']]).

    '''
    def __init__(self, nV, xLoBnds=None, xUpBnds=None, xTypes=None,
                 xInitVals=None, lambdaInitVals=None, xScale=None,
                 xHonorBnds=None, xProperties=None, xNames=None):
        self.nV              = nV
        self.xTypes          = xTypes
        self.xLoBnds         = xLoBnds
        self.xUpBnds         = xUpBnds
        self.xInitVals       = xInitVals
        self.lambdaInitVals  = lambdaInitVals
        self.xScale          = xScale
        self.xHonorBnds      = xHonorBnds
        self.xProperties     = xProperties
        self.xNames          = xNames


class Constraints:
    '''Store mathematical definition of the contraints.

    nC: int
        The number of constraints

    cType: array, optional
        The type of the constraints (e.g. :envvar:`KN_CONTYPE_GENERAL`,
        :envvar:`KN_CONTYPE_LINEAR`, :envvar:`KN_CONTYPE_QUADRATIC).`
        There are two ways to specify the types:

        1. array, shape (nC, 1): define all the constraints types.

        2. array, shape (2, n_cons): define the types for `n_cons`
        constraints using two arrays. The first array defines
        the constraints index and the second one their types.
        For example, constraints 0 and 2 are defined as linear::

            cType=[[0, 2], [KN_CONTYPE_LINEAR, KN_CONTYPE_LINEAR]]

        If not set, constraints are assumed to be general.

    cLoBnds: array, optional
        The lower bounds of the constraints. There are two
        ways to specify the lower bounds:

        1. array, shape (nC, 1): define all the constraints lower bounds.

        2. array, shape (2, n_cons): define the lower bounds for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        lower bounds. For example, constraints 0 and 2 have their
        lower bounds set to 0::

            cLoBnds=[[0, 2], [0, 0]]

        If not set, lower bounds are assumed to be :envvar:`-KN_INFINITY`.

    cUpBnds: array, optional
        The upper bounds of the constraints. There are two
        ways to specify the upper bounds:

        1. array, shape (nC, 1): define all the constraints upper bounds.

        2. array, shape (2, n_cons): define the upper bound for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        upper bounds. For example, constraints 0 and 2 have their upper
        bounds set to 10::

            cUpBnds=[[0, 2], [10, 10]]

        If not set, upper bounds are assumed to be :envvar:`KN_INFINITY`.

    cEqBnds: array, optional
        The equality bounds of the constraints. There are two
        ways to specify the equality bounds:

        1. array, shape (nC, 1): define all the constraints equality bounds.

        2. array, shape (2, n_cons): define the equality bound for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        equality bounds. For example, constraints 0 and 2 have their
        equality bounds set to 10::

            cEqBnds=[[0, 2], [10, 10]]

    cConstant: array, optional
        Add constants to the body of constraint functions. There are two
        ways to specify the constants:

        1. array, shape (nC, 1): define all the constraints constants.

        2. array, shape (2, n_cons): define the constant structure for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        values. For example, constraints 0 and 2 have a constant term equal to
        respectively -1 and 3::

            cConstant=[[0, 2], [-1, 3]]

    cLinear: array, optional
        Add linear structure to the constraints. There are two
        ways to specify the linear structure:

        1. array, shape (nC, 1): define all the constraints linear structure.

        2. array, shape (3, n_cons): define the linear structure for `n_cons`
        constraints using three arrays. The first
        array defines the constraint index, the second one the variable
        index and the third one the coefficient values. For example, constraints
        1 and 2 have one linear structure respectively equal to 2*x[0]
        and -x[3]::

            cLinear=[[1, 2], [0, 3], [2, -1]]

    cQuadratic: array, optional
        Add quadratic structure to the constraint functions. There are two
        ways to specify the quadratic structure:

        1. array, shape (nC, 1): define all the constraints quadratic structure.

        2. array, shape (4, n_cons): define the quadratic structure for `n_cons`
        constraints using four arrays. The first
        array defines the constraint index, the second one the first
        variable index, the third one the second variable index and
        the last one the coefficient values. For example, constraint 1
        has one quadratic structure equal to 2*x[0]*x[1]::

            cQuadratic=[[1], [0], [1], [2]]

    lambdaInitVals: array, optional
        The initial dual values of the constraints (i.e. the Lagrange
        multipliers for the constraints). There are two ways to specify
        the initial dual values:

        1. array, shape (nC, 1): define all the constraints initial values.

        2. array, shape (2, n_cons): define the initial values for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        values. For example, constraints 0 and 2 are initially set
        to -1 and 3::

            lambdaInitVals=[[0, 2], [-1, 3]]

        If not set, constraints dual variables may be initialized as 0
        or initialized by Knitro based on some initialization strategy
        (perhaps determined by a user option).

    cScaleFactors: array, optional
        Set an array of constraint scaling values to perform a scaling::

            cScaled[i] = cScaleFactors[i] * c[i]

        for each constraint. These scaling factors should try to
        represent the `typical` values of the inverse of the constraint
        values `c` so that the scaled constraints (`cScaled`) used
        internally by Knitro are close to one. The values
        for cScaleFactors should be positive. If a non-positive value
        is specified, that constraint will use either the standard Knitro
        scaling (:envvar:`KN_SCALE_USER_INTERNAL`), or no scaling
        (:envvar:`KN_SCALE_USER_NONE`).
        Scaling factors for standard constraints can be provided with
        `cScaleFactors`, while scalings for complementarity constraint can
        be specified with `ccScaleFactors` (see Complementarity constraints).
        There are two ways to specify the scaling value:

        1. array, shape (nC, 1): define all the constraints cScaleFactors.

        2. array, shape (2, n_cons): define the scaling values for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their scaling
        factor (e.g constraints 0 and 2 only are scaled)::

            cScaleFactors=[[0, 2], [2, 2]

    cNames: array, optional
        Specify the names of the constraints so that Knitro can internally
        print out them. There are two ways to specify the
        constraints names:

        1. array, shape (nC, 1): define all the constraints names.

        2. array, shape (2, n_cons): define the names for `n_cons`
        constraints using two arrays. The first
        array defines the constraints index and the second one their
        names. For example, we name constraints 0 and 2::

            cNames=[[0, 2], ['con0', 'con2']]

    '''
    def __init__(self, nC, cType=None, cLoBnds=None, cUpBnds=None,
                 cEqBnds=None, cConstant=None, cLinear=None,
                 cQuadratic=None, lambdaInitVals=None, cScaleFactors=None,
                 cNames=None):
        self.nC             = nC
        self.cType          = cType
        self.cLoBnds        = cLoBnds
        self.cUpBnds        = cUpBnds
        self.cEqBnds        = cEqBnds
        self.cConstant      = cConstant
        self.cLinear        = cLinear
        self.cQuadratic     = cQuadratic
        self.lambdaInitVals = lambdaInitVals
        self.cScaleFactors  = cScaleFactors
        self.cNames         = cNames


class Objective:
    '''Store mathematical definition of the problem objective.

    objGoal: int, optional
        Set the objective goal (:envvar:`KN_OBJGOAL_MINIMIZE` or
        :envvar:`KN_OBJGOAL_MAXIMIZE`). If not set, the default
        goal is set to minimization.

    objConstant: float, optional
        Add a constant to the objective function.

    objLinear: array shape (2, n_vars), optional
        Add linear structure to the objective function.
        The first array defines `n_vars` variables index and the second one
        `n_vars` coefficient values.
        For example, two linear structures are added
        to the objective, 2*x[0] and -x[3]::

            objLinear=[[[0, 3], [2, -1]]

    objQuadratic: array shape (3,n_vars), optional
        Add quadratic structure to the objective function.
        The first two arrays define `n_vars` variables indices and the
        last one `n_vars` coefficient values.
        For example, one quadratic structure is added
        to the objective 2*x[0]*x[1]::

            objQuadratic=[[0], [1], [2]]

    objScaleFactor: float, optional
        Set a scaling value for the objective function::

            objScaled = objScaleFactor * obj

        This scaling factor should try to represent the `typical`
        value of the inverse of the objective function value `obj` so
        that the scaled objective (`objScaled`) used internally by
        Knitro is close to one. The value for objScaleFactor
        should be positive. If a non-positive value is specified, then
        the objective will use either the standard Knitro scaling
        (:envvar:`KN_SCALE_USER_INTERNAL`), or no scaling
        (:envvar:`KN_SCALE_USER_NONE`).

    '''
    def __init__(self, objGoal=None, objConstant=None, objLinear=None,
                 objQuadratic=None, objScaleFactor=None):
        self.objGoal        = objGoal
        self.objConstant    = objConstant
        self.objLinear      = objLinear
        self.objQuadratic   = objQuadratic
        self.objScaleFactor = objScaleFactor


class ComplementarityConstraints:
    '''Store mathematical definition of the complementarity constraints.

    nCc: int
        The number of complementarity constraints

    ccTypes: array, optional
        Specify the type of complementarity:

           * :envvar:`KN_CCTYPE_VARVAR`: two (non-negative) variables
           * :envvar:`KN_CCTYPE_VARCON`: a variable and a constraint
           * :envvar:`KN_CCTYPE_CONCON`: two constraints

        Note: Currently only KN_CCTYPE_VARVAR is supported.
        The other `ccTypes` will be added in future releases.
        The length of ccTypes must be equal to indexComps1 and
        indexComps2 length.

    indexComps1: array, optional
        Specify the variable indice of the first element of
        each complementarity constraint. Each pair (indexComps1, indexComps2)
        defines a complementarity constraint between the two variables.
        The two array of variable indices must be of equal length, and
        contain matching pairs of variable indices.

    indexComps2: array, optional
        Specify the variable indice of the second element of
        each complementarity constraint. Each pair (indexComps1, indexComps2)
        defines a complementarity constraint between the two variables.
        The two array of variable indices must be of equal length, and
        contain matching pairs of variable indices.

    ccScaleFactors: array, optional
        Set an array of constraint scaling values to perform a scaling::

            ccScaled[i] = ccScaleFactors[i] * cc[i]

        for each constraint. These scaling factors should try to
        represent the `typical` values of the inverse of the complementarity
        constraint values `c` so that the scaled complementarity constraints
        (`cScaled`) used internally by Knitro are close to one. The values
        for ccScaleFactors should be positive. If a non-positive value
        is specified, that constraint will use either the standard Knitro
        scaling (:envvar:`KN_SCALE_USER_INTERNAL`), or no scaling
        (:envvar:`KN_SCALE_USER_NONE`). There are two ways to specify
        the scaling value:

        1. array, shape (nCc, 1): define all the constraints ccScaleFactors.

        2. array, shape (2, n_cons): define the scaling values for `n_cons`
        complementarity constraints using two arrays. The first
        array defines the constraints index and the second one their
        scaling factor. For example, complementarity constraints 0
        and 2 only are scaled::

            ccScaleFactors=[[0, 2], [2, 2]]

    cNames: array, optional
        Specify the name of the complementarity constraints so that
        Knitro can internally print out these names. There are two
        ways to specify the complementarity constraints names:

        1. array, shape (nCc, 1): define all the constraints names.

        2. array, shape (2, n_cons): define the names for `n_cons`
        complementarity constraints only using two arrays. The first
        array defines the constraints index and the second one their
        names. For example, we name complementarity constraints 0 and 2::

            cNames=[[0, 2], ['compCon0', 'compCon2']]

    '''
    def __init__(self, ccTypes=None, indexComps1=None, indexComps2=None,
                 ccScaleFactors=None, cNames=None):
        self.ccTypes        = ccTypes
        self.indexComps1    = indexComps1
        self.indexComps2    = indexComps2
        self.ccScaleFactors = ccScaleFactors
        self.cNames         = cNames


class Callback:
    '''Store the definition of optimization callbacks.

    funcCallback: Function
        Function evaluating the objective and any constraint
        parts involved in this callback. If `evalFCGA` is equal to `True`,
        the callback should also evaluate the relevant first
        derivatives/gradients.

    evalObj: boolean, optional
        Indicate whether any part of the objective function is evaluated
        in the callback. If not specified, `evalObj` is set to `False`.
        If both, `evalObj` and `indexCons` are not specified, the
        funcCallback is considered to provide a generic callback
        evaluating the objective function and all constraints.

    indexCons: array of shape (n_cons,), optional
        Index of the `n_cons` constraints evaluated in the callback.
        If not specified, indexCons is set to `None`. If both `evalObj` and
        `indexCons` are not specified, the funcCallback is considered
        to provide a callback for objective function and all constraints.

    evalFCGA: boolean, optional
        Indicate if the first derivatives/gradients are also
        evaluated in the funcCallBack.

        Note: It is generally more efficient and recommended
        to have separate callback routines for functions and
        gradients since a gradient evaluation is not always
        needed for every function evaluations. However, in some
        cases, it may be more convenient to compute them together
        if most of the work for computing the gradients is already
        done while evaluating the function.

    gradCallback: Function, optional
        Function evaluating the components of the first derivatives/gradients
        of the objective and the constraint involved in this callback.
        If not specified, Knitro will approximate
        the gradient using finite-differencing. However, we recommend
        providing callbacks to evaluate the exact gradients whenever
        possible as this can drastically improve the performance of Knitro.
        If `evalFCGA` is `True`, the first derivatives are expected to be
        evaluated in the funcCallback and gradCallback is not required.

    objGradIndexVars: array of shape (n_vars,), optionnal
        The `n_vars` nonzero indices of the objective gradient. If the
        objective gradient evaluated in gradCallback is dense,
        `objGradIndexVars` should be set to :envvar:`KN_DENSE`.

    jacIndexCons: array of shape (nnzj,), optional
        Store `nnzj` indexes (row) of each nonzero in the Jacobian of
        the constraints involved in this callback. If the Jacobian
        evaluated in gradCallback is dense, jacIndexCons should be set
        to :envvar:`KN_DENSE_ROWMAJOR` to provide the full Jacobian in
        row major order (i.e. ordered by rows/constraints), or
        :envvar:`KN_DENSE_COLMAJOR` to provide the full Jacobian
        in column major order (i.e. ordered by columns/variables).
        The user should always try to define the sparsity
        structure for the Jacobian (`jacIndexCons`, `jacIndexVars`).
        Even when using finite-difference approximations to compute
        the gradients, knowing the sparse structure of the Jacobian
        can allow Knitro to compute these finite-difference
        approximations faster.

    jacIndexVars: array of shape (nnzj,), optional
        Store `nnzj` index (column) of each nonzero in the Jacobian of
        the constraints parts involved in this callback. If the Jacobian
        evaluated in gradCallback is dense, jacIndexVars is not required.

    hessCallback: Function, optional
        Function evaluating the components of the Hessian of the Lagrangian
        corresponding to the objective and any constraint
        involved in this callback. If not specified, Knitro will
        approximate the Hessian by finite-difference.
        However, providing a callback for the
        exact Hessian (as well as the non-zero sparsity structure) can
        greatly improve Knitro performance and is recommended if possible.

    hessIndexVars1: array of shape (nnzh,), optional
        Store `nnzh` index of each nonzero in the Hessian of the
        Lagrangian. If the Hessian evaluated in hessCallback is dense,
        hessIndexVars1 should be set to :envvar:`KN_DENSE_ROWMAJOR`
        to provide the full upper triangular Hessian in row major
        order, or :envvar:`KN_DENSE_COLMAJOR` to provide the full
        upper triangular Hessian in column major order.
        Note that the Hessian is symmetric, so the lower
        triangular components are the same as the upper triangular
        components with row and column indices swapped.

    hessIndexVars2: array of shape (nnzh,), optional
        Store `nnzh` index of of each nonzero in the Hessian of the
        Lagrangian. If the Hessian evaluated in hessCallback is dense,
        hessIndexVars2 is not required.

    hessianNoFAllow: boolean, optional
        Specify that the user is able to provide evaluations of the
        Hessian matrix without the objective component. Turned off
        by default but should be enabled if possible.

    '''
    def __init__(self, funcCallback, evalObj=None, indexCons=None,
                 evalFCGA=False, gradCallback=None,
                 objGradIndexVars=None, jacIndexCons=None,
                 jacIndexVars=None, hessCallback=None,
                 hessIndexVars1=None, hessIndexVars2=None,
                 hessianNoFAllow=False):
        self.funcCallback     = funcCallback
        self.evalObj          = evalObj
        self.indexCons        = indexCons
        self.evalFCGA         = evalFCGA
        self.gradCallback     = gradCallback
        self.objGradIndexVars = objGradIndexVars
        self.jacIndexCons     = jacIndexCons
        self.jacIndexVars     = jacIndexVars
        self.hessCallback     = hessCallback
        self.hessIndexVars1   = hessIndexVars1
        self.hessIndexVars2   = hessIndexVars2
        self.hessianNoFAllow  = hessianNoFAllow


class Solution:
    '''Store the solution returned by Knitro.

    status: int
        The solution status return codes are organized as follows.

        * 0: the final solution satisfies the termination conditions for verifying optimality.

        * -100 to -199: a feasible approximate solution was found.

        * -200 to -299: Knitro terminated at an infeasible point.

        * -300: the problem was determined to be unbounded.

        * | -400 to -499: Knitro terminated because it reached a pre-defined limit
          | (-40x codes indicate that a feasible point was found before reaching the
          | limit, while -41x codes indicate that no feasible point was found before
          | reaching the limit).

        * -500 to -599: Knitro terminated with an input error or some non-standard error.

    obj: double
        Final objective value.

    x: array
        Primal variables optimal values.

    lambdaVals: array
        Dual variables optimal values.

    iter: int
        Number of iterations.

    numFCevals: int
        Number of function evaluations.

    numGAevals: int
        Number of gradient evaluations.

    numHevals: int
        Number of Hessian evaluations.

    numHVevals: int
        Number of Hessian-vector products evaluations.

    absFeasError: double
        Absolute feasibility error at the solution.

    relFeasError: double
        Relative feasibility error at the solution.

    absOptError: double
        Absolute optimality error at the solution.

    relOptError: double
        Relative optimality error at the solution.

    solveTimeCPU: double
        Retrieve Knitro solve time as CPU time.

    solveTimeReal: double
        Retrieve Knitro solve time as real time.
    '''
    def __init__(self, kc):
        self.status, self.obj, self.x, self.lambdaVals = KN_get_solution(kc)
        self.iter          = KN_get_number_iters(kc)
        self.numFCevals    = KN_get_number_FC_evals(kc)
        self.numGAevals    = KN_get_number_GA_evals(kc)
        self.numHevals     = KN_get_number_H_evals(kc)
        self.numHVevals    = KN_get_number_HV_evals(kc)
        self.absFeasError  = KN_get_abs_feas_error(kc)
        self.relFeasError  = KN_get_rel_feas_error(kc)
        self.absOptError   = KN_get_rel_opt_error(kc)
        self.relOptError   = KN_get_rel_opt_error(kc)
        self.solveTimeCPU  = KN_get_solve_time_cpu(kc)
        self.solveTimeReal = KN_get_solve_time_real(kc)

#----------------------------------------------------------------------------
#     KNITRO LOAD FUNCTION
#----------------------------------------------------------------------------

def _load_options(kc, algorithm, options, optionsFile):
    '''Load Knitro solver options into KN_context `kc`.
    Regarding the options setting, the `algorithm` argument
    takes precedence over the `option` dictionnary which itself
    takes precedence over the `optionsFile`.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    algorithm : int or str, optional
        The algorithms to be used by knitro:
        (1) `interior-direct`
        (2) `interior-cg`
        (3) `active-set`
        (4) `sqp`

    options : dict, optional
        A dictionary of solver options.

    optionsFile : str, optional
        A file containing solver options.

    '''
    #---- Load the option file
    if isinstance(optionsFile, str):
        KN_load_param_file(kc, optionsFile)

    #---- Load the options argument
    if options is not None:
        for option in options:
            if isinstance(options.get(option), int):
                KN_set_int_param(kc, option, options.get(option))
            elif isinstance(options.get(option), float):
                KN_set_double_param(kc, option, options.get(option))
            elif isinstance(options.get(option), str):
                KN_set_char_param(kc, option, options.get(option))

    #---- Set the algorithm
    if algorithm is not None:
        if isinstance(algorithm, str):
            algorithm = algorithm.lower()
        if algorithm == 'auto' or algorithm == 0:
            KN_set_int_param(kc, "algorithm", 0)
        elif algorithm == 'interior-direct' or algorithm == 1:
            KN_set_int_param(kc, "algorithm", 1)
        elif algorithm == 'interior-cg' or algorithm == 2:
            KN_set_int_param(kc, "algorithm", 2)
        elif algorithm == 'active-set' or algorithm == 3:
            KN_set_int_param(kc, "algorithm", 3)
        elif algorithm == 'sqp' or algorithm == 4:
            KN_set_int_param(kc, "algorithm", 4)
        elif algorithm == 'multi' or algorithm == 5:
            KN_set_int_param(kc, "algorithm", 5)
        else:
            raise ValueError("Knitro-Python error: " +
                             "incorrect value for option algorithm!")


def _load_variables(kc, variables):
    '''Load variables into KN_context kc

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    variables : Variables object
        Structure containing all the variables related information

    '''
    #---- Add the variables.
    xIndices = KN_add_vars(kc, nV=variables.nV)

    #---- Variables bounds
    if variables.xLoBnds is not None:
        try:  # xLoBnds defined for a subset of the variables
            nV = len(variables.xLoBnds[0])
            KN_set_var_lobnds(kc, indexVars=variables.xLoBnds[0],
                              xLoBnds=variables.xLoBnds[1])
        except TypeError:
            KN_set_var_lobnds(kc, xLoBnds=variables.xLoBnds)

    if variables.xUpBnds is not None:
        try:  # xUpBnds defined for a subset of the variables
            nV = len(variables.xUpBnds[0])
            KN_set_var_upbnds(kc, indexVars=variables.xUpBnds[0],
                              xUpBnds=variables.xUpBnds[1])
        except TypeError:
            KN_set_var_upbnds(kc, xUpBnds=variables.xUpBnds)

    #---- Variables type
    if variables.xTypes is not None:
        try:  # xTypes defined for a subset of the variables
            nV = len(variables.xTypes[0])
            KN_set_var_types(kc, indexVars=variables.xTypes[0],
                             xTypes=variables.xTypes[1])
        except TypeError:
            KN_set_var_types(kc, xTypes=variables.xTypes)

    #---- Variables primal/dual initial values
    if variables.xInitVals is not None:
        try:  # xInitVals defined for a subset of the variables
            nV = len(variables.xInitVals[0])
            KN_set_var_primal_init_values(kc,
                                          indexVars=variables.xInitVals[0],
                                          xInitVals=variables.xInitVals[1])
        except TypeError:
            KN_set_var_primal_init_values(kc, xInitVals=variables.xInitVals)

    if variables.lambdaInitVals is not None:
        try:  # lambdaInitVals defined for a subset of the variables
            nV = len(variables.lambdaInitVals[0])
            KN_set_var_dual_init_values(kc,
                                        indexVars=variables.lambdaInitVals[0],
                                        lambdaInitVals=variables.lambdaInitVals[1])
        except TypeError:
            KN_set_var_dual_init_values(kc,
                                        lambdaInitVals=variables.lambdaInitVals)

    #---- Variables scaling
    if variables.xScale is not None:
        nV = len(variables.xScale)
        if nV == 2:
            KN_set_var_scalings(kc, xScaleFactors=variables.xScale[0],
                                xScaleCenters=variables.xScale[1])
        elif nV == 3: # scaling defined for a subset of the variables
            KN_set_var_scalings(kc, indexVars=variables.xScale[0],
                                xScaleFactors=variables.xScale[1],
                                xScaleCenters=variables.xScale[2])
        else:
            raise ValueError("Knitro-Python error: " +
                             "wrong number of arguments for variable scaling!")

    #---- Variables honor bounds
    if variables.xHonorBnds is not None:
        try:  # xHonorBnds defined for a subset of the variables
            nV = len(variables.xHonorBnds[0])
            KN_set_var_honorbnds(kc, indexVars=variables.xHonorBnds[0],
                                 xHonorBnds = variables.xHonorBnds[1])
        except TypeError:
            KN_set_var_honorbnds(kc, xHonorBnds=variables.xHonorBnds)

    #---- Variables properties
    if variables.xProperties is not None:
        try:  # xProperties defined for a subset of the variables
            nV = len(variables.xProperties[0])
            KN_set_var_properties(kc, indexVars=variables.xProperties[0],
                                  xProperties=variables.xProperties[1])
        except TypeError:
            KN_set_var_properties(kc, xProperties=variables.xProperties)

    #---- Variables names
    if variables.xNames is not None:
        # xNames defined for a subset of the variables
        if not isinstance(variables.xNames[0], str):
            KN_set_var_names(kc, indexVars=variables.xNames[0],
                             xNames=variables.xNames[1])
        else:
            KN_set_var_names(kc, xNames=variables.xNames)


def _load_objective(kc, objective):
    '''Load objective into KN_context kc.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    objective : Objective object
        Structure containing all the objective related information

    '''
    #---- Objective goal
    if objective.objGoal is not None:
        objGoal = objective.objGoal
        if isinstance(objGoal, str):
            objGoal = objGoal.lower()
        if objGoal == 'minimize' or objGoal == KN_OBJGOAL_MINIMIZE:
            KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)
        elif objGoal == 'maximize' or objGoal == KN_OBJGOAL_MAXIMIZE:
            KN_set_obj_goal(kc, KN_OBJGOAL_MAXIMIZE)

    #---- Objective structure
    if objective.objConstant is not None:
        KN_add_obj_constant(kc, constant=objective.objConstant)

    if objective.objLinear is not None:
        KN_add_obj_linear_struct(kc, indexVars=objective.objLinear[0],
                                 coefs=objective.objLinear[1])

    if objective.objQuadratic is not None:
        KN_add_obj_quadratic_struct(kc, indexVars1=objective.objQuadratic[0],
                                    indexVars2=objective.objQuadratic[1],
                                    coefs=objective.objQuadratic[2])

    #---- Objective scaling
    if objective.objScaleFactor is not None:
        KN_set_obj_scaling(kc, objScaleFactor=objective.objScaleFactor)


def _load_constraints(kc, constraints):
    '''Load constraints information into KN_context kc.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    constraints : Constraints object
        Structure containing all the constraints related information

    '''
    #---- Add the constraints.
    KN_add_cons(kc, nC=constraints.nC)

    if constraints is not None:
        #---- Constraints bounds
        if constraints.cLoBnds is not None:
            try:  # cLoBnds defined for a subset of the constraints
                nC = len(constraints.cLoBnds[0])
                KN_set_con_lobnds(kc, indexCons=constraints.cLoBnds[0],
                                  cLoBnds=constraints.cLoBnds[1])
            except TypeError:
                KN_set_con_lobnds(kc, cLoBnds=constraints.cLoBnds)

        if constraints.cUpBnds is not None:
            try:  # cUpBnds defined for a subset of the constraints
                nC = len(constraints.cUpBnds[0])
                KN_set_con_upbnds(kc, indexCons=constraints.cUpBnds[0],
                                  cUpBnds=constraints.cUpBnds[1])
            except TypeError:
                KN_set_con_upbnds(kc, cUpBnds=constraints.cUpBnds)

        if constraints.cEqBnds is not None:
            try:  # cEqBnds defined for a subset of the constraints
                nC = len(constraints.cEqBnds[0])
                KN_set_con_eqbnds(kc, indexCons=constraints.cEqBnds[0],
                                  cEqBnds=constraints.cEqBnds[1])
            except TypeError:
                KN_set_con_eqbnds(kc, cEqBnds=constraints.cEqBnds)

        #---- Constraints dual initial values
        if constraints.lambdaInitVals is not None:
            KN_set_con_dual_init_values(kc, lambdaInitVals=lambdaInitVals)

        #---- Constraints scaling
        if constraints.cScaleFactors is not None:
            try:  # cScaleFactors defined for a subset of the constraints
                nC = len(constraints.cScaleFactors[0])
                KN_set_con_scalings(kc, indexCons=constraints.cScaleFactors[0],
                                    cScaleFactors=constraints.cScaleFactors[1])
            except TypeError:
                KN_set_con_scalings(kc, cScaleFactors=constraints.cScaleFactors)

        #---- Constraints names
        if constraints.cNames is not None:
            # cNames defined for a subset of the constraints
            if not isinstance(constraints.cNames[0], str):
                KN_set_con_names(kc, indexCons=constraints.cNames[0],
                                 cNames=constraints.cNames[1])
            else:
                KN_set_con_names(kc, cNames=constraints.cNames)

        #---- Constraints structure
        if constraints.cConstant is not None:
            constStruct = constraints.cConstant
            KN_add_con_constants (kc,
                                  indexCons=constraints.cConstant[0],
                                  constants=constraints.cConstant[1])

        if constraints.cLinear is not None:
            KN_add_con_linear_struct (kc,
                                      indexCons=constraints.cLinear[0],
                                      indexVars=constraints.cLinear[1],
                                      coefs=constraints.cLinear[2])

        if constraints.cQuadratic is not None:
            KN_add_con_quadratic_struct(kc,
                                        indexCons=constraints.cQuadratic[0],
                                        indexVars1=constraints.cQuadratic[1],
                                        indexVars2=constraints.cQuadratic[2],
                                        coefs=constraints.cQuadratic[3])


def _load_single_callback(kc, callback):
    '''Load a single callback into KN_context kc.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    callbacks : Callback object
        Structure containing the callback to be loaded.

    '''
    cb = KN_add_eval_callback(kc, evalObj=callback.evalObj,
                              indexCons=callback.indexCons,
                              funcCallback=callback.funcCallback)

    if callback.evalFCGA:
        KN_set_int_param(kc, KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES)
        KN_set_cb_grad(kc, cb,
                       objGradIndexVars=callback.objGradIndexVars,
                       jacIndexCons=callback.jacIndexCons,
                       jacIndexVars=callback.jacIndexVars)

    if callback.gradCallback is not None:
        KN_set_cb_grad(kc, cb,
                       objGradIndexVars=callback.objGradIndexVars,
                       jacIndexCons=callback.jacIndexCons,
                       jacIndexVars=callback.jacIndexVars,
                       gradCallback=callback.gradCallback)

    if callback.hessCallback is not None:
        KN_set_cb_hess(kc, cb,
                       hessIndexVars1=callback.hessIndexVars1,
                       hessIndexVars2=callback.hessIndexVars2,
                       hessCallback=callback.hessCallback)

        if callback.hessianNoFAllow:
            KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)


def _load_callbacks(kc, callbacks):
    '''Load callbacks information into KN_context kc.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    callbacks : array of Callback object
        List containing all the callbacks related information.

    '''
    for callback in callbacks:
        _load_single_callback(kc, callback)


def _load_complementarity_constraints(kc, compConstraints):
    '''Load complementarity constraints information into KN_context kc.

    Parameters
    ----------
    kc : KN_context
        Knitro solver context object to be updated

    compConstraints : ComplementarityConstraints object
        Structure containing all the complementarity
        constraints related information
    '''
    KN_set_compcons(kc,
                    ccTypes=compConstraints.ccTypes,
                    indexComps1=compConstraints.indexComps1,
                    indexComps2=compConstraints.indexComps2)

    #---- Complementarity constraints scaling
    if compConstraints.ccScaleFactors is not None:
        # scaling defined for a subset of the compConstraints
        try:
            nCC = len(compConstraints.ccScaleFactors[0])
            KN_set_compcon_scalings(kc,
                                    indexCompCons=compConstraints.ccScaleFactors[0],
                                    ccScaleFactors=compConstraints.ccScaleFactors[1])
        except TypeError:
            KN_set_compcon_scalings(kc, ccScaleFactors=compConstraints.ccScaleFactors)

    #---- Complementarity constraints name
    if compConstraints.cNames is not None:
        # cNames defined for a subset of the compConstraints
        if not isinstance(compConstraints.cNames[0], str):
            KN_set_compcon_names(kc,
                                 indexCompCons=compConstraints.cNames[0],
                                 cNames=compConstraints.cNames[1])
        else:
            KN_set_compcon_names(kc, cNames=compConstraints.cNames)


#----------------------------------------------------------------------------
#     KNITRO SINGLE CALL FUNCTION
#----------------------------------------------------------------------------

def optimize(variables, objective=None, constraints=None, compConstraints=None,
           callbacks=None, algorithm=None, options=None, optionsFile=None):
    '''
    Solve an optimization problem using Artelys Knitro.

    This function is designed to provide maximal flexibility to the
    end-user. Note that the user should try to provide as much
    information as possible as it usualy leads to improve the
    performance of Knitro.

    Parameters
    ----------
    variables: Variables object
        Structure containing all the :class:`knitro.Variables` related information.

    objective: Objective object, optionnal
        Structure containing all the :class:`knitro.Objective` related information.

    constraints: Constraints object, optional
        Structure containing all the :class:`knitro.Constraints` related information

    compConstraints: ComplementarityConstraints object, optional
        Structure containing all the :class:`knitro.ComplementarityConstraints`
        related information.

    callbacks: array, Callback object, optional
        Structure containing all the :class:`knitro.Callback` related information.

    algorithm: int or str, optional
        The algorithms to be used by knitro:

        (1) `interior-direct`

        (2) `interior-cg`

        (3) `active-set`

        (4) `sqp`

    options: dict, optional
        A dictionary of solver options.

    optionsFile: str, optional
        A file containing solver options.

    Returns
    -------
    solution: Solution object
        Structure containing solve related information.
        See :class:`.Solution` for a detailed description.

    Notes
    -----
    Regarding the options setting, :py:attr:`~algorithm` takes
    precedence over the :py:attr:`~options` dictionnary which itself
    takes precedence over the :py:attr:`~optionsFile`.
    '''
    #---- Create a new Knitro solver instance.
    with KN_new() as kc:
        #--- Load the solver options
        _load_options(kc, algorithm, options, optionsFile)

        #---- Load the variable information
        _load_variables(kc, variables)

        #---- Load the objective information
        if objective is not None:
            _load_objective(kc, objective)

        #---- Load the constraint information
        if constraints is not None:
            _load_constraints(kc, constraints)

        #---- Load complementarity constraints
        if compConstraints is not None:
            _load_complementarity_constraints(kc, compConstraints)

        #--- Load the callback information
        if callbacks is not None:
            if isinstance(callbacks, list):
                _load_callbacks(kc, callbacks)
            elif isinstance(callbacks, Callback):
                _load_single_callback(kc, callbacks)

        #---- Solve the problem.
        KN_solve(kc)

        #---- Retrieve the solution information
        sol = Solution(kc)

    return sol
