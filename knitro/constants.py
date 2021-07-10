#*******************************************************
#* Copyright (c) 2020 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++  Artelys Knitro 12.2 Python API
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


'''Definition of Artelys Knitro constants to match include/knitro.h.
'''

import sys

#---- PROBLEM DEFINITION CONSTANTS
KN_INFINITY                     = sys.float_info.max    # Same as DBL_MAX in C

KN_PARAMTYPE_INTEGER            = 0
KN_PARAMTYPE_FLOAT              = 1
KN_PARAMTYPE_STRING             = 2

KN_OBJGOAL_MINIMIZE             = 0
KN_OBJGOAL_MAXIMIZE             = 1

KN_OBJTYPE_CONSTANT             = -1
KN_OBJTYPE_GENERAL              = 0
KN_OBJTYPE_LINEAR               = 1
KN_OBJTYPE_QUADRATIC            = 2

KN_CONTYPE_CONSTANT             = -1
KN_CONTYPE_GENERAL              = 0
KN_CONTYPE_LINEAR               = 1
KN_CONTYPE_QUADRATIC            = 2
KN_CONTYPE_CONIC                = 3

KN_RSDTYPE_CONSTANT             = -1
KN_RSDTYPE_GENERAL              = 0
KN_RSDTYPE_LINEAR               = 1

KN_CCTYPE_VARVAR                = 0
KN_CCTYPE_VARCON                = 1
KN_CCTYPE_CONCON                = 2

KN_VARTYPE_CONTINUOUS           = 0
KN_VARTYPE_INTEGER              = 1
KN_VARTYPE_BINARY               = 2

KN_VAR_LINEAR                   = 1

KN_OBJ_CONVEX                   =  1
KN_OBJ_CONCAVE                  =  2
KN_OBJ_CONTINUOUS               =  4
KN_OBJ_DIFFERENTIABLE           =  8
KN_OBJ_TWICE_DIFFERENTIABLE     = 16
KN_OBJ_NOISY                    = 32
KN_OBJ_NONDETERMINISTIC         = 64

KN_CON_CONVEX                   =  1
KN_CON_CONCAVE                  =  2
KN_CON_CONTINUOUS               =  4
KN_CON_DIFFERENTIABLE           =  8
KN_CON_TWICE_DIFFERENTIABLE     = 16
KN_CON_NOISY                    = 32
KN_CON_NONDETERMINISTIC         = 64

KN_DENSE                        = -1
KN_DENSE_ROWMAJOR               = -2
KN_DENSE_COLMAJOR               = -3

KN_RC_EVALFC                    =  1
KN_RC_EVALGA                    =  2
KN_RC_EVALH                     =  3
KN_RC_EVALHV                    =  7
KN_RC_EVALH_NO_F                =  8
KN_RC_EVALHV_NO_F               =  9
KN_RC_EVALR                     = 10
KN_RC_EVALRJ                    = 11
KN_RC_EVALFCGA                  = 12

#---- KNITRO RETURN CODES.
KN_RC_OPTIMAL_OR_SATISFACTORY   = 0
KN_RC_OPTIMAL                   = 0
KN_RC_NEAR_OPT                  = -100
KN_RC_FEAS_XTOL                 = -101
KN_RC_FEAS_NO_IMPROVE           = -102
KN_RC_FEAS_FTOL                 = -103
KN_RC_INFEASIBLE                = -200
KN_RC_INFEAS_XTOL               = -201
KN_RC_INFEAS_NO_IMPROVE         = -202
KN_RC_INFEAS_MULTISTART         = -203
KN_RC_INFEAS_CON_BOUNDS         = -204
KN_RC_INFEAS_VAR_BOUNDS         = -205
KN_RC_UNBOUNDED                 = -300
KN_RC_UNBOUNDED_OR_INFEAS       = -301
KN_RC_ITER_LIMIT_FEAS           = -400
KN_RC_TIME_LIMIT_FEAS           = -401
KN_RC_FEVAL_LIMIT_FEAS          = -402
KN_RC_MIP_EXH_FEAS              = -403
KN_RC_MIP_TERM_FEAS             = -404
KN_RC_MIP_SOLVE_LIMIT_FEAS      = -405
KN_RC_MIP_NODE_LIMIT_FEAS       = -406
KN_RC_ITER_LIMIT_INFEAS         = -410
KN_RC_TIME_LIMIT_INFEAS         = -411
KN_RC_FEVAL_LIMIT_INFEAS        = -412
KN_RC_MIP_EXH_INFEAS            = -413
KN_RC_MIP_SOLVE_LIMIT_INFEAS    = -415
KN_RC_MIP_NODE_LIMIT_INFEAS     = -416
KN_RC_CALLBACK_ERR              = -500
KN_RC_LP_SOLVER_ERR             = -501
KN_RC_EVAL_ERR                  = -502
KN_RC_OUT_OF_MEMORY             = -503
KN_RC_USER_TERMINATION          = -504
KN_RC_OPEN_FILE_ERR             = -505
KN_RC_BAD_N_OR_F                = -506
KN_RC_BAD_CONSTRAINT            = -507
KN_RC_BAD_JACOBIAN              = -508
KN_RC_BAD_HESSIAN               = -509
KN_RC_BAD_CON_INDEX             = -510
KN_RC_BAD_JAC_INDEX             = -511
KN_RC_BAD_HESS_INDEX            = -512
KN_RC_BAD_CON_BOUNDS            = -513
KN_RC_BAD_VAR_BOUNDS            = -514
KN_RC_ILLEGAL_CALL              = -515
KN_RC_BAD_KCPTR                 = -516
KN_RC_NULL_POINTER              = -517
KN_RC_BAD_INIT_VALUE            = -518
KN_RC_BAD_LICENSE               = -520
KN_RC_BAD_PARAMINPUT            = -521
KN_RC_LINEAR_SOLVER_ERR         = -522
KN_RC_DERIV_CHECK_FAILED        = -523
KN_RC_DERIV_CHECK_TERMINATE     = -524
KN_RC_OVERFLOW_ERR              = -525
KN_RC_BAD_SIZE                  = -526
KN_RC_BAD_VARIABLE              = -527
KN_RC_BAD_VAR_INDEX             = -528
KN_RC_BAD_OBJECTIVE             = -529
KN_RC_BAD_OBJ_INDEX             = -530
KN_RC_BAD_RESIDUAL              = -531
KN_RC_BAD_RSD_INDEX             = -532
KN_RC_INTERNAL_ERROR            = -600

#---- KNITRO PARAMETERS.
KN_PARAM_NEWPOINT               = 1001
KN_NEWPOINT_NONE                    = 0
KN_NEWPOINT_SAVEONE                 = 1
KN_NEWPOINT_SAVEALL                 = 2
KN_PARAM_HONORBNDS              = 1002
KN_HONORBNDS_AUTO                   = -1
KN_HONORBNDS_NO                     = 0
KN_HONORBNDS_ALWAYS                 = 1
KN_HONORBNDS_INITPT                 = 2
KN_PARAM_ALGORITHM              = 1003
KN_PARAM_ALG                    = 1003
KN_ALG_AUTOMATIC                    = 0
KN_ALG_AUTO                         = 0
KN_ALG_BAR_DIRECT                   = 1
KN_ALG_BAR_CG                       = 2
KN_ALG_ACT_CG                       = 3
KN_ALG_ACT_SQP                      = 4
KN_ALG_MULTI                        = 5
KN_PARAM_BAR_MURULE             = 1004
KN_BAR_MURULE_AUTOMATIC             = 0
KN_BAR_MURULE_AUTO                  = 0
KN_BAR_MURULE_MONOTONE              = 1
KN_BAR_MURULE_ADAPTIVE              = 2
KN_BAR_MURULE_PROBING               = 3
KN_BAR_MURULE_DAMPMPC               = 4
KN_BAR_MURULE_FULLMPC               = 5
KN_BAR_MURULE_QUALITY               = 6
KN_PARAM_BAR_FEASIBLE           = 1006
KN_BAR_FEASIBLE_NO                  = 0
KN_BAR_FEASIBLE_STAY                = 1
KN_BAR_FEASIBLE_GET                 = 2
KN_BAR_FEASIBLE_GET_STAY            = 3
KN_PARAM_GRADOPT                = 1007
KN_GRADOPT_EXACT                    = 1
KN_GRADOPT_FORWARD                  = 2
KN_GRADOPT_CENTRAL                  = 3
KN_GRADOPT_USER_FORWARD             = 4
KN_GRADOPT_USER_CENTRAL             = 5
KN_PARAM_HESSOPT                = 1008
KN_HESSOPT_AUTO                     = 0
KN_HESSOPT_EXACT                    = 1
KN_HESSOPT_BFGS                     = 2
KN_HESSOPT_SR1                      = 3
KN_HESSOPT_FINITE_DIFF              = 4
KN_HESSOPT_PRODUCT_FINDIFF          = 4
KN_HESSOPT_PRODUCT                  = 5
KN_HESSOPT_LBFGS                    = 6
KN_HESSOPT_GAUSS_NEWTON             = 7
KN_PARAM_BAR_INITPT             = 1009
KN_BAR_INITPT_AUTO                  = 0
KN_BAR_INITPT_CONVEX                = 1
KN_BAR_INITPT_NEARBND               = 2
KN_BAR_INITPT_CENTRAL               = 3
KN_PARAM_ACT_LPSOLVER           = 1012
KN_ACT_LPSOLVER_INTERNAL            = 1
KN_ACT_LPSOLVER_CPLEX               = 2
KN_ACT_LPSOLVER_XPRESS              = 3
KN_PARAM_CG_MAXIT               = 1013
KN_PARAM_MAXIT                  = 1014
KN_PARAM_OUTLEV                 = 1015
KN_OUTLEV_NONE                      = 0
KN_OUTLEV_SUMMARY                   = 1
KN_OUTLEV_ITER_10                   = 2
KN_OUTLEV_ITER                      = 3
KN_OUTLEV_ITER_VERBOSE              = 4
KN_OUTLEV_ITER_X                    = 5
KN_OUTLEV_ALL                       = 6
KN_PARAM_OUTMODE                = 1016
KN_OUTMODE_SCREEN                   = 0
KN_OUTMODE_FILE                     = 1
KN_OUTMODE_BOTH                     = 2
KN_PARAM_SCALE                  = 1017
KN_SCALE_NEVER                      = 0
KN_SCALE_NO                         = 0
KN_SCALE_USER_INTERNAL              = 1
KN_SCALE_USER_NONE                  = 2
KN_SCALE_INTERNAL                   = 3
KN_PARAM_SOC                    = 1019
KN_SOC_NO                           = 0
KN_SOC_MAYBE                        = 1
KN_SOC_YES                          = 2
KN_PARAM_DELTA                  = 1020
KN_PARAM_BAR_FEASMODETOL        = 1021
KN_PARAM_FEASTOL                = 1022
KN_PARAM_FEASTOLABS             = 1023
KN_PARAM_MAXTIMECPU             = 1024
KN_PARAM_BAR_INITMU             = 1025
KN_PARAM_OBJRANGE               = 1026
KN_PARAM_OPTTOL                 = 1027
KN_PARAM_OPTTOLABS              = 1028
KN_PARAM_LINSOLVER_PIVOTTOL     = 1029
KN_PARAM_XTOL                   = 1030
KN_PARAM_DEBUG                  = 1031
KN_DEBUG_NONE                       = 0
KN_DEBUG_PROBLEM                    = 1
KN_DEBUG_EXECUTION                  = 2
KN_PARAM_MULTISTART             = 1033
KN_PARAM_MSENABLE               = 1033
KN_MULTISTART_NO                    = 0
KN_MULTISTART_YES                   = 1
KN_PARAM_MSMAXSOLVES            = 1034
KN_PARAM_MSMAXBNDRANGE          = 1035
KN_PARAM_MSMAXTIMECPU           = 1036
KN_PARAM_MSMAXTIMEREAL          = 1037
KN_PARAM_LMSIZE                 = 1038
KN_PARAM_BAR_MAXCROSSIT         = 1039
KN_PARAM_MAXTIMEREAL            = 1040
KN_PARAM_CG_PRECOND             = 1041
KN_CG_PRECOND_NONE                  = 0
KN_CG_PRECOND_CHOL                  = 1
KN_PARAM_BLASOPTION             = 1042
KN_BLASOPTION_KNITRO                = 0
KN_BLASOPTION_INTEL                 = 1
KN_BLASOPTION_DYNAMIC               = 2
KN_PARAM_BAR_MAXREFACTOR        = 1043
KN_PARAM_LINESEARCH_MAXTRIALS   = 1044
KN_PARAM_BLASOPTIONLIB          = 1045
KN_PARAM_OUTAPPEND              = 1046
KN_OUTAPPEND_NO                     = 0
KN_OUTAPPEND_YES                    = 1
KN_PARAM_OUTDIR                 = 1047
KN_PARAM_CPLEXLIB               = 1048
KN_PARAM_BAR_PENRULE            = 1049
KN_BAR_PENRULE_AUTO                 = 0
KN_BAR_PENRULE_SINGLE               = 1
KN_BAR_PENRULE_FLEX                 = 2
KN_PARAM_BAR_PENCONS            = 1050
KN_BAR_PENCONS_AUTO                 = -1
KN_BAR_PENCONS_NONE                 = 0
KN_BAR_PENCONS_ALL                  = 2
KN_BAR_PENCONS_EQUALITIES           = 3
KN_BAR_PENCONS_INFEAS               = 4
KN_PARAM_MSNUMTOSAVE            = 1051
KN_PARAM_MSSAVETOL              = 1052
KN_PARAM_PRESOLVEDEBUG          = 1053
KN_PRESOLVEDBG_NONE                 = 0
KN_PRESOLVEDBG_BASIC                = 1
KN_PRESOLVEDBG_VERBOSE              = 2
KN_PARAM_MSTERMINATE            = 1054
KN_MSTERMINATE_MAXSOLVES            = 0
KN_MSTERMINATE_OPTIMAL              = 1
KN_MSTERMINATE_FEASIBLE             = 2
KN_MSTERMINATE_ANY                  = 3
KN_PARAM_MSSTARTPTRANGE         = 1055
KN_PARAM_INFEASTOL              = 1056
KN_PARAM_LINSOLVER              = 1057
KN_LINSOLVER_AUTO                   = 0
KN_LINSOLVER_INTERNAL               = 1
KN_LINSOLVER_HYBRID                 = 2
KN_LINSOLVER_DENSEQR                = 3
KN_LINSOLVER_MA27                   = 4
KN_LINSOLVER_MA57                   = 5
KN_LINSOLVER_MKLPARDISO             = 6
KN_LINSOLVER_MA97                   = 7
KN_LINSOLVER_MA86                   = 8
KN_PARAM_BAR_DIRECTINTERVAL     = 1058
KN_PARAM_PRESOLVE               = 1059
KN_PRESOLVE_NO                      = 0
KN_PRESOLVE_NONE                    = 0  # DEPRECATED
KN_PRESOLVE_YES                     = 1
KN_PRESOLVE_BASIC                   = 1  # DEPRECATED
KN_PRESOLVE_ADVANCED                = 2  # DEPRECATED
KN_PARAM_PRESOLVE_TOL           = 1060
KN_PARAM_BAR_SWITCHRULE         = 1061
KN_BAR_SWITCHRULE_AUTO              = -1
KN_BAR_SWITCHRULE_NEVER             = 0
KN_BAR_SWITCHRULE_MODERATE          = 2
KN_BAR_SWITCHRULE_AGGRESSIVE        = 3
KN_PARAM_HESSIAN_NO_F           = 1062
KN_HESSIAN_NO_F_FORBID              = 0
KN_HESSIAN_NO_F_ALLOW               = 1
KN_PARAM_MA_TERMINATE           = 1063
KN_MA_TERMINATE_ALL                 = 0
KN_MA_TERMINATE_OPTIMAL             = 1
KN_MA_TERMINATE_FEASIBLE            = 2
KN_MA_TERMINATE_ANY                 = 3
KN_PARAM_MA_MAXTIMECPU          = 1064
KN_PARAM_MA_MAXTIMEREAL         = 1065
KN_PARAM_MSSEED                 = 1066
KN_PARAM_MA_OUTSUB              = 1067
KN_MA_OUTSUB_NONE                   = 0
KN_MA_OUTSUB_YES                    = 1
KN_PARAM_MS_OUTSUB              = 1068
KN_MS_OUTSUB_NONE                   = 0
KN_MS_OUTSUB_YES                    = 1
KN_PARAM_XPRESSLIB              = 1069
KN_PARAM_TUNER                  = 1070
KN_TUNER_OFF                        = 0
KN_TUNER_ON                         = 1
KN_PARAM_TUNER_OPTIONSFILE      = 1071
KN_PARAM_TUNER_MAXTIMECPU       = 1072
KN_PARAM_TUNER_MAXTIMEREAL      = 1073
KN_PARAM_TUNER_OUTSUB           = 1074
KN_TUNER_OUTSUB_NONE                = 0
KN_TUNER_OUTSUB_SUMMARY             = 1
KN_TUNER_OUTSUB_ALL                 = 2
KN_PARAM_TUNER_TERMINATE        = 1075
KN_TUNER_TERMINATE_ALL              = 0
KN_TUNER_TERMINATE_OPTIMAL          = 1
KN_TUNER_TERMINATE_FEASIBLE         = 2
KN_TUNER_TERMINATE_ANY              = 3
KN_PARAM_LINSOLVER_OOC          = 1076
KN_LINSOLVER_OOC_NO                 = 0
KN_LINSOLVER_OOC_MAYBE              = 1
KN_LINSOLVER_OOC_YES                = 2
KN_PARAM_BAR_RELAXCONS          = 1077
KN_BAR_RELAXCONS_NONE               = 0
KN_BAR_RELAXCONS_EQS                = 1
KN_BAR_RELAXCONS_INEQS              = 2
KN_BAR_RELAXCONS_ALL                = 3
KN_PARAM_MSDETERMINISTIC        = 1078
KN_MSDETERMINISTIC_NO               = 0
KN_MSDETERMINISTIC_YES              = 1
KN_PARAM_BAR_REFINEMENT         = 1079
KN_BAR_REFINEMENT_NO                = 0
KN_BAR_REFINEMENT_YES               = 1
KN_PARAM_DERIVCHECK             = 1080
KN_DERIVCHECK_NONE                  = 0
KN_DERIVCHECK_FIRST                 = 1
KN_DERIVCHECK_SECOND                = 2
KN_DERIVCHECK_ALL                   = 3
KN_PARAM_DERIVCHECK_TYPE        = 1081
KN_DERIVCHECK_FORWARD               = 1
KN_DERIVCHECK_CENTRAL               = 2
KN_PARAM_DERIVCHECK_TOL         = 1082
KN_PARAM_LINSOLVER_INEXACT      = 1083
KN_LINSOLVER_INEXACT_NO             = 0
KN_LINSOLVER_INEXACT_YES            = 1
KN_PARAM_LINSOLVER_INEXACTTOL   = 1084
KN_PARAM_MAXFEVALS              = 1085
KN_PARAM_FSTOPVAL               = 1086
KN_PARAM_DATACHECK              = 1087
KN_DATACHECK_NO                     = 0
KN_DATACHECK_YES                    = 1
KN_PARAM_DERIVCHECK_TERMINATE   = 1088
KN_DERIVCHECK_STOPERROR             = 1
KN_DERIVCHECK_STOPALWAYS            = 2
KN_PARAM_BAR_WATCHDOG           = 1089
KN_BAR_WATCHDOG_NO                  = 0
KN_BAR_WATCHDOG_YES                 = 1
KN_PARAM_FTOL                   = 1090
KN_PARAM_FTOL_ITERS             = 1091
KN_PARAM_ACT_QPALG              = 1092
KN_ACT_QPALG_AUTO                   = 0
KN_ACT_QPALG_BAR_DIRECT             = 1
KN_ACT_QPALG_BAR_CG                 = 2
KN_ACT_QPALG_ACT_CG                 = 3
KN_PARAM_BAR_INITPI_MPEC        = 1093
KN_PARAM_XTOL_ITERS             = 1094
KN_PARAM_LINESEARCH             = 1095
KN_LINESEARCH_AUTO                  = 0
KN_LINESEARCH_BACKTRACK             = 1
KN_LINESEARCH_INTERPOLATE           = 2
KN_LINESEARCH_WEAKWOLFE             = 3
KN_PARAM_OUT_CSVINFO            = 1096
KN_OUT_CSVINFO_NO                   = 0
KN_OUT_CSVINFO_YES                  = 1
KN_PARAM_INITPENALTY            = 1097
KN_PARAM_ACT_LPFEASTOL          = 1098
KN_PARAM_CG_STOPTOL             = 1099
KN_PARAM_RESTARTS               = 1100
KN_PARAM_RESTARTS_MAXIT         = 1101
KN_PARAM_BAR_SLACKBOUNDPUSH     = 1102
KN_PARAM_CG_PMEM                = 1103
KN_PARAM_BAR_SWITCHOBJ          = 1104
KN_BAR_SWITCHOBJ_NONE               = 0
KN_BAR_SWITCHOBJ_SCALARPROX         = 1
KN_BAR_SWITCHOBJ_DIAGPROX           = 2
KN_PARAM_OUTNAME                = 1105
KN_PARAM_OUT_CSVNAME            = 1106
KN_PARAM_ACT_PARAMETRIC         = 1107
KN_ACT_PARAMETRIC_NO                = 0
KN_ACT_PARAMETRIC_MAYBE             = 1
KN_ACT_PARAMETRIC_YES               = 2
KN_PARAM_ACT_LPDUMPMPS          = 1108
KN_ACT_LPDUMPMPS_NO                 = 0
KN_ACT_LPDUMPMPS_YES                = 1
KN_PARAM_ACT_LPALG              = 1109
KN_ACT_LPALG_DEFAULT                = 0
KN_ACT_LPALG_PRIMAL                 = 1
KN_ACT_LPALG_DUAL                   = 2
KN_ACT_LPALG_BARRIER                = 3
KN_PARAM_ACT_LPPRESOLVE         = 1110
KN_ACT_LPPRESOLVE_OFF               = 0
KN_ACT_LPPRESOLVE_ON                = 1
KN_PARAM_ACT_LPPENALTY          = 1111
KN_ACT_LPPENALTY_AUTO               = 0
KN_ACT_LPPENALTY_ALL                = 1
KN_ACT_LPPENALTY_NONLINEAR          = 2
KN_ACT_LPPENALTY_DYNAMIC            = 3
KN_PARAM_BNDRANGE               = 1112
KN_PARAM_BAR_CONIC_ENABLE       = 1113
KN_BAR_CONIC_ENABLE_NONE            = 0
KN_BAR_CONIC_ENABLE_SOC             = 1
KN_PARAM_CONVEX                 = 1114
KN_CONVEX_AUTO                      = -1
KN_CONVEX_NO                        = 0
KN_CONVEX_YES                       = 1
KN_PARAM_OUT_HINTS              = 1115
KN_OUT_HINTS_NO                     = 0
KN_OUT_HINTS_YES                    = 1
KN_PARAM_EVAL_FCGA              = 1116
KN_EVAL_FCGA_NO                     = 0
KN_EVAL_FCGA_YES                    = 1
KN_PARAM_BAR_MAXCORRECTORS      = 1117
KN_PARAM_STRAT_WARM_START       = 1118
KN_STRAT_WARM_START_NO              = 0
KN_STRAT_WARM_START_YES             = 1
KN_PARAM_FINDIFF_TERMINATE      = 1119
KN_FINDIFF_TERMINATE_NONE           = 0
KN_FINDIFF_TERMINATE_ERREST         = 1
KN_PARAM_CPUPLATFORM            = 1120
KN_CPUPLATFORM_AUTO                 = -1
KN_CPUPLATFORM_COMPATIBLE           = 1
KN_CPUPLATFORM_SSE2                 = 2
KN_CPUPLATFORM_AVX                  = 3
KN_CPUPLATFORM_AVX2                 = 4
KN_CPUPLATFORM_AVX512               = 5  # EXPERIMENTAL
KN_PARAM_PRESOLVE_PASSES        = 1121
KN_PARAM_PRESOLVE_LEVEL         = 1122
KN_PRESOLVE_LEVEL_AUTO              = -1
KN_PRESOLVE_LEVEL_1                 = 1
KN_PRESOLVE_LEVEL_2                 = 2
KN_PARAM_FINDIFF_RELSTEPSIZE    = 1123
KN_PARAM_INFEASTOL_ITERS        = 1124
KN_PARAM_PRESOLVEOP_TIGHTEN     = 1125
KN_PRESOLVEOP_TIGHTEN_AUTO          = -1
KN_PRESOLVEOP_TIGHTEN_NONE          = 0
KN_PRESOLVEOP_TIGHTEN_VARBND        = 1
KN_PARAM_BAR_LINSYS             = 1126
KN_BAR_LINSYS_AUTO                  = -1
KN_BAR_LINSYS_FULL                  = 0
KN_BAR_LINSYS_COMPACT1              = 1
KN_BAR_LINSYS_COMPACT2              = 2
KN_PARAM_PRESOLVE_INITPT        = 1127
KN_PRESOLVE_INITPT_AUTO             = -1
KN_PRESOLVE_INITPT_NOSHIFT          = 0
KN_PRESOLVE_INITPT_LINSHIFT         = 1
KN_PRESOLVE_INITPT_ANYSHIFT         = 2
KN_PARAM_ACT_QPPENALTY          = 1128
KN_ACT_QPPENALTY_AUTO               = -1
KN_ACT_QPPENALTY_NONE               = 0
KN_ACT_QPPENALTY_ALL                = 1

#---- KNITRO MIP PARAMETERS.
KN_PARAM_MIP_METHOD             = 2001
KN_MIP_METHOD_AUTO                  = 0
KN_MIP_METHOD_BB                    = 1
KN_MIP_METHOD_HQG                   = 2
KN_MIP_METHOD_MISQP                 = 3
KN_PARAM_MIP_BRANCHRULE         = 2002
KN_MIP_BRANCH_AUTO                  = 0
KN_MIP_BRANCH_MOSTFRAC              = 1
KN_MIP_BRANCH_PSEUDOCOST            = 2
KN_MIP_BRANCH_STRONG                = 3
KN_PARAM_MIP_SELECTRULE         = 2003
KN_MIP_SEL_AUTO                     = 0
KN_MIP_SEL_DEPTHFIRST               = 1
KN_MIP_SEL_BESTBOUND                = 2
KN_MIP_SEL_COMBO_1                  = 3
KN_PARAM_MIP_INTGAPABS          = 2004
KN_PARAM_MIP_INTGAPREL          = 2005
KN_PARAM_MIP_MAXTIMECPU         = 2006
KN_PARAM_MIP_MAXTIMEREAL        = 2007
KN_PARAM_MIP_MAXSOLVES          = 2008
KN_PARAM_MIP_INTEGERTOL         = 2009
KN_PARAM_MIP_OUTLEVEL           = 2010
KN_MIP_OUTLEVEL_NONE                = 0
KN_MIP_OUTLEVEL_ITERS               = 1
KN_MIP_OUTLEVEL_ITERSTIME           = 2
KN_MIP_OUTLEVEL_ROOT                = 3
KN_PARAM_MIP_OUTINTERVAL        = 2011
KN_PARAM_MIP_OUTSUB             = 2012
KN_MIP_OUTSUB_NONE                  = 0
KN_MIP_OUTSUB_YES                   = 1
KN_MIP_OUTSUB_YESPROB               = 2
KN_PARAM_MIP_DEBUG              = 2013
KN_MIP_DEBUG_NONE                   = 0
KN_MIP_DEBUG_ALL                    = 1
KN_PARAM_MIP_IMPLICATNS         = 2014
KN_MIP_IMPLICATNS_NO                = 0
KN_MIP_IMPLICATNS_YES               = 1
KN_PARAM_MIP_GUB_BRANCH         = 2015
KN_MIP_GUB_BRANCH_NO                = 0
KN_MIP_GUB_BRANCH_YES               = 1
KN_PARAM_MIP_KNAPSACK           = 2016
KN_MIP_KNAPSACK_NO                  = 0
KN_MIP_KNAPSACK_NONE                = 0
KN_MIP_KNAPSACK_INEQ                = 1
KN_MIP_KNAPSACK_LIFTED              = 2
KN_MIP_KNAPSACK_ALL                 = 3
KN_PARAM_MIP_ROUNDING           = 2017
KN_MIP_ROUND_AUTO                   = -1
KN_MIP_ROUND_NONE                   = 0
KN_MIP_ROUND_HEURISTIC              = 2
KN_MIP_ROUND_NLP_SOME               = 3
KN_MIP_ROUND_NLP_ALWAYS             = 4
KN_PARAM_MIP_ROOTALG            = 2018
KN_MIP_ROOTALG_AUTO                 = 0
KN_MIP_ROOTALG_BAR_DIRECT           = 1
KN_MIP_ROOTALG_BAR_CG               = 2
KN_MIP_ROOTALG_ACT_CG               = 3
KN_MIP_ROOTALG_ACT_SQP              = 4
KN_MIP_ROOTALG_MULTI                = 5
KN_PARAM_MIP_LPALG              = 2019
KN_MIP_LPALG_AUTO                   = 0
KN_MIP_LPALG_BAR_DIRECT             = 1
KN_MIP_LPALG_BAR_CG                 = 2
KN_MIP_LPALG_ACT_CG                 = 3
KN_PARAM_MIP_TERMINATE          = 2020
KN_MIP_TERMINATE_OPTIMAL            = 0
KN_MIP_TERMINATE_FEASIBLE           = 1
KN_PARAM_MIP_MAXNODES           = 2021
KN_PARAM_MIP_HEURISTIC          = 2022
KN_MIP_HEURISTIC_AUTO               = -1
KN_MIP_HEURISTIC_NONE               = 0
KN_MIP_HEURISTIC_FEASPUMP           = 2
KN_MIP_HEURISTIC_MPEC               = 3
KN_PARAM_MIP_HEUR_MAXIT         = 2023
KN_PARAM_MIP_HEUR_MAXTIMECPU    = 2024
KN_PARAM_MIP_HEUR_MAXTIMEREAL   = 2025
KN_PARAM_MIP_PSEUDOINIT         = 2026
KN_MIP_PSEUDOINIT_AUTO              = 0
KN_MIP_PSEUDOINIT_AVE               = 1
KN_MIP_PSEUDOINIT_STRONG            = 2
KN_PARAM_MIP_STRONG_MAXIT       = 2027
KN_PARAM_MIP_STRONG_CANDLIM     = 2028
KN_PARAM_MIP_STRONG_LEVEL       = 2029
KN_PARAM_MIP_INTVAR_STRATEGY    = 2030
KN_MIP_INTVAR_STRATEGY_NONE         = 0
KN_MIP_INTVAR_STRATEGY_RELAX        = 1
KN_MIP_INTVAR_STRATEGY_MPEC         = 2
KN_PARAM_MIP_RELAXABLE          = 2031
KN_MIP_RELAXABLE_NONE               = 0
KN_MIP_RELAXABLE_ALL                = 1
KN_PARAM_MIP_NODEALG            = 2032
KN_MIP_NODEALG_AUTO                 = 0
KN_MIP_NODEALG_BAR_DIRECT           = 1
KN_MIP_NODEALG_BAR_CG               = 2
KN_MIP_NODEALG_ACT_CG               = 3
KN_MIP_NODEALG_ACT_SQP              = 4
KN_MIP_NODEALG_MULTI                = 5
KN_PARAM_MIP_HEUR_TERMINATE     = 2033
KN_MIP_HEUR_TERMINATE_FEASIBLE      = 1
KN_MIP_HEUR_TERMINATE_LIMIT         = 2
KN_PARAM_MIP_SELECTDIR          = 2034
KN_MIP_SELECTDIR_DOWN               = 0
KN_MIP_SELECTDIR_UP                 = 1
KN_PARAM_MIP_CUTFACTOR          = 2035
KN_PARAM_MIP_ZEROHALF           = 2036
KN_MIP_ZEROHALF_NONE                = 0
KN_MIP_ZEROHALF_ROOT                = 1
KN_MIP_ZEROHALF_TREE                = 2
KN_MIP_ZEROHALF_ALL                 = 3
KN_PARAM_MIP_MIR                = 2037
KN_MIP_MIR_AUTO                     = -1
KN_MIP_MIR_NONE                     = 0
KN_MIP_MIR_TREE                     = 1
KN_MIP_MIR_NLP                      = 2
KN_PARAM_MIP_CLIQUE             = 2038
KN_MIP_CLIQUE_NONE                  = 0
KN_MIP_CLIQUE_ROOT                  = 1
KN_MIP_CLIQUE_TREE                  = 2
KN_MIP_CLIQUE_ALL                   = 3

#---- KNITRO MULTITHREAD PARAMETERS.
KN_PARAM_PAR_NUMTHREADS         = 3001
KN_PARAM_PAR_CONCURRENT_EVALS   = 3002
KN_PAR_CONCURRENT_EVALS_NO          = 0
KN_PAR_CONCURRENT_EVALS_YES         = 1
KN_PARAM_PAR_BLASNUMTHREADS     = 3003
KN_PARAM_PAR_LSNUMTHREADS       = 3004
KN_PARAM_PAR_MSNUMTHREADS       = 3005
KN_PAR_MSNUMTHREADS_AUTO            = 0
KN_PARAM_PAR_CONICNUMTHREADS    = 3006
