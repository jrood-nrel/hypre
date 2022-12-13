/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_PARCSR_LS_HEADER
#define NALU_HYPRE_PARCSR_LS_HEADER

#include "NALU_HYPRE_utilities.h"
#include "NALU_HYPRE_seq_mv.h"
#include "NALU_HYPRE_parcsr_mv.h"
#include "NALU_HYPRE_IJ_mv.h"
#include "NALU_HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup ParCSRSolvers ParCSR Solvers
 *
 * These solvers use matrix/vector storage schemes that are taylored
 * for general sparse matrix systems.
 *
 * @memo Linear solvers for sparse matrix systems
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Solvers
 *
 * @{
 **/

struct hypre_Solver_struct;
/**
 * The solver object.
 **/

#ifndef NALU_HYPRE_SOLVER_STRUCT
#define NALU_HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *NALU_HYPRE_Solver;
#endif

typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToParSolverFcn)(NALU_HYPRE_Solver,
                                             NALU_HYPRE_ParCSRMatrix,
                                             NALU_HYPRE_ParVector,
                                             NALU_HYPRE_ParVector);

#ifndef NALU_HYPRE_MODIFYPC
#define NALU_HYPRE_MODIFYPC
typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToModifyPCFcn)(NALU_HYPRE_Solver,
                                            NALU_HYPRE_Int,
                                            NALU_HYPRE_Real);
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BoomerAMG Solver and Preconditioner
 *
 * Parallel unstructured algebraic multigrid solver and preconditioner
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGCreate(NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDestroy(NALU_HYPRE_Solver solver);

/**
 * Set up the BoomerAMG solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetup(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

/**
 * Solve the system or apply AMG as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSolve(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

/**
 * Solve the transpose system \f$A^T x = b\f$ or apply AMG as a preconditioner
 * to the transpose system . Note that this function should only be used
 * when preconditioning CGNR with BoomerAMG. It can only be used with
 * Jacobi smoothing (relax_type 0 or 7) and without CF smoothing,
 * i.e relax_order needs to be set to 0.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSolveT(NALU_HYPRE_Solver       solver,
                                NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector    b,
                                NALU_HYPRE_ParVector    x);

/**
 * Recovers old default for coarsening and interpolation, i.e Falgout
 * coarsening and untruncated modified classical interpolation.
 * This option might be preferred for 2 dimensional problems.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOldDefault(NALU_HYPRE_Solver       solver);

/**
 * Returns the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetResidual(NALU_HYPRE_Solver     solver,
                                     NALU_HYPRE_ParVector *residual);

/**
 * Returns the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetNumIterations(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int          *num_iterations);

/*
 * Returns cumulative num of nonzeros for A and P operators
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetCumNnzAP(NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Real   *cum_nnz_AP);

/*
 * Activates cumulative num of nonzeros for A and P operators. 
 * Needs to be set to a positive number for activation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCumNnzAP(NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Real    cum_nnz_AP);

/**
 * Returns the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                      NALU_HYPRE_Real   *rel_resid_norm);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1, i.e. a scalar system.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumFunctions(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int          num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDofFunc(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *dof_func);

/**
 * (Optional) Set the type convergence checking
 * 0: (default) norm(r)/norm(b), or norm(r) when b == 0
 * 1: nomr(r) / norm(r_0)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetConvergeType(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    type);

/**
 * (Optional) Set the convergence tolerance, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 0.
 * The default is 1.e-6.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetTol(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real   tol);

/**
 * (Optional) Sets maximum number of iterations, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 1.
 * The default is 20.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxIter(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int          max_iter);

/**
 * (Optional)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMinIter(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    min_iter);

/**
 * (Optional) Sets maximum size of coarsest grid.
 * The default is 9.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxCoarseSize(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    max_coarse_size);

/**
 * (Optional) Sets minimum size of coarsest grid.
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMinCoarseSize(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    min_coarse_size);

/**
 * (Optional) Sets maximum number of multigrid levels.
 * The default is 25.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxLevels(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    max_levels);

/**
 * (Optional) Sets cut factor for choosing isolated points
 * during coarsening according to the rows' density. The default is 0.
 * If nnzrow > coarsen_cut_factor*avg_nnzrow, where avg_nnzrow is the
 * average number of nonzeros per row of the global matrix, holds for
 * a given row, it is set as fine, and interpolation weights are not computed.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Int    coarsen_cut_factor);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2D Laplace operators, 0.25 is a good value, for 3D Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetStrongThreshold(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Real   strong_threshold);

/**
 * (Optional) The strong threshold for R is strong connections used
 * in building an approximate ideal restriction.
 * Default value is 0.25.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetStrongThresholdR(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real   strong_threshold);

/**
 * (Optional) The filter threshold for R is used to eliminate small entries
 * of the approximate ideal restriction after building it.
 * Default value is 0.0, which disables filtering.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFilterThresholdR(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real   filter_threshold);

/**
 * (Optional) Deprecated. This routine now has no effect.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real   S_commpkg_switch);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If \e max_row_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxRowSum(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real    max_row_sum);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for \e coarsen_type:
 *
 *    - 0  : CLJP-coarsening (a parallel coarsening algorithm using independent sets.
 *    - 1  : classical Ruge-Stueben coarsening on each processor, no boundary treatment
             (not recommended!)
 *    - 3  : classical Ruge-Stueben coarsening on each processor, followed by a third pass,
             which adds coarse points on the boundaries
 *    - 6  : Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
 *           generated by 1 as its first independent set)
 *    - 7  : CLJP-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 8  : PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating
 *           lower complexities than CLJP, might also lead to slower convergence)
 *    - 9  : PMIS-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 10 : HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed
 *           by PMIS using the interior C-points generated as its first independent set)
 *    - 11 : one-pass Ruge-Stueben coarsening on each processor, no boundary treatment
             (not recommended!)
 *    - 21 : CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer
 *    - 22 : CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer
 *
 * The default is 10.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoarsenType(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    coarsen_type);

/**
 * (Optional) Defines the non-Galerkin drop-tolerance
 * for sparsifying coarse grid operators and thus reducing communication.
 * Value specified here is set on all levels.
 * This routine should be used before NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol, which
 * then can be used to change individual levels if desired
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNonGalerkinTol (NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Real  nongalerkin_tol);

/**
 * (Optional) Defines the level specific non-Galerkin drop-tolerances
 * for sparsifying coarse grid operators and thus reducing communication.
 * A drop-tolerance of 0.0 means to skip doing non-Galerkin on that
 * level.  The maximum drop tolerance for a level is 1.0, although
 * much smaller values such as 0.03 or 0.01 are recommended.
 *
 * Note that if the user wants to set a  specific tolerance on all levels,
 * NALU_HYPRE_BooemrAMGSetNonGalerkinTol should be used. Individual levels
 * can then be changed using this routine.
 *
 * In general, it is safer to drop more aggressively on coarser levels.
 * For instance, one could use 0.0 on the finest level, 0.01 on the second level and
 * then using 0.05 on all remaining levels. The best way to achieve this is
 * to set 0.05 on all levels with NALU_HYPRE_BoomerAMGSetNonGalerkinTol and then
 * change the tolerance on level 0 to 0.0 and the tolerance on level 1 to 0.01
 * with NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol.
 * Like many AMG parameters, these drop tolerances can be tuned.  It is also common
 * to delay the start of the non-Galerkin process further to a later level than
 * level 1.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param nongalerkin_tol [IN] level specific drop tolerance
 * @param level [IN] level on which drop tolerance is used
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol (NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Real   nongalerkin_tol,
                                                 NALU_HYPRE_Int  level);

/**
 * (Optional) Defines the non-Galerkin drop-tolerance (old version)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNonGalerkTol (NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    nongalerk_num_tol,
                                          NALU_HYPRE_Real  *nongalerk_tol);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMeasureType(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    measure_type);

/**
 * (Optional) Defines the number of levels of aggressive coarsening.
 * The default is 0, i.e. no aggressive coarsening.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggNumLevels(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    agg_num_levels);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1. Larger numbers lead to less aggressive
 * coarsening.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumPaths(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    num_paths);

/**
 * (optional) Defines the number of pathes for CGC-coarsening.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCGCIts (NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    its);

/**
 * (Optional) Sets whether to use the nodal systems coarsening.
 * Should be used for linear systems generated from systems of PDEs.
 * The default is 0 (unknown-based coarsening,
 *                   only coarsens within same function).
 * For the remaining options a nodal matrix is generated by
 * applying a norm to the nodal blocks and applying the coarsening
 * algorithm to this matrix.
 *    - 1 : Frobenius norm
 *    - 2 : sum of absolute values of elements in each block
 *    - 3 : largest element in each block (not absolute value)
 *    - 4 : row-sum norm
 *    - 6 : sum of all values in each block
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNodal(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    nodal);
/**
 * (Optional) Sets whether to give special treatment to diagonal elements in
 * the nodal systems version.
 * The default is 0.
 * If set to 1, the diagonal entry is set to the negative sum of all off
 * diagonal entries.
 * If set to 2, the signs of all diagonal entries are inverted.
 */
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNodalDiag(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    nodal_diag);


/*
 * (Optional) Sets whether to keep same sign in S for nodal > 0
 * The default is 0, i.e., discard those elements.
 */
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetKeepSameSign(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    keep_same_sign);

/**
 * (Optional) Defines which parallel interpolation operator is used.
 * There are the following options for \e interp_type:
 *
 *    - 0  : classical modified interpolation
 *    - 1  : LS interpolation (for use with GSMG)
 *    - 2  : classical modified interpolation for hyperbolic PDEs
 *    - 3  : direct interpolation (with separation of weights) (also for GPU use)
 *    - 4  : multipass interpolation
 *    - 5  : multipass interpolation (with separation of weights)
 *    - 6  : extended+i interpolation (also for GPU use)
 *    - 7  : extended+i (if no common C neighbor) interpolation
 *    - 8  : standard interpolation
 *    - 9  : standard interpolation (with separation of weights)
 *    - 10 : classical block interpolation (for use with nodal systems version only)
 *    - 11 : classical block interpolation (for use with nodal systems version only)
 *           with diagonalized diagonal blocks
 *    - 12 : FF interpolation
 *    - 13 : FF1 interpolation
 *    - 14 : extended interpolation (also for GPU use)
 *    - 15 : interpolation with adaptive weights (GPU use only)
 *    - 16 : extended interpolation in matrix-matrix form
 *    - 17 : extended+i interpolation in matrix-matrix form
 *    - 18 : extended+e interpolation in matrix-matrix form
 *
 * The default is ext+i interpolation (interp_type 6) trunctated to at most 4
 * elements per row. (see NALU_HYPRE_BoomerAMGSetPMaxElmts).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpType(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    interp_type);

/**
 * (Optional) Defines a truncation factor for the interpolation. The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetTruncFactor(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real   trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 4. To turn off truncation, it needs to be set to 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPMaxElmts(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    P_max_elmts);

/**
 * (Optional) Defines whether separation of weights is used
 * when defining strength for standard interpolation or
 * multipass interpolation.
 * Default: 0, i.e. no separation of weights used.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSepWeight(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    sep_weight);

/**
 * (Optional) Defines the interpolation used on levels of aggressive coarsening
 * The default is 4, i.e. multipass interpolation.
 * The following options exist:
 *
 *    - 1 : 2-stage extended+i interpolation
 *    - 2 : 2-stage standard interpolation
 *    - 3 : 2-stage extended interpolation
 *    - 4 : multipass interpolation
 *    - 5 : 2-stage extended interpolation in matrix-matrix form
 *    - 6 : 2-stage extended+i interpolation in matrix-matrix form
 *    - 7 : 2-stage extended+e interpolation in matrix-matrix form
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggInterpType(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    agg_interp_type);

/**
 * (Optional) Defines the truncation factor for the
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggTruncFactor(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real   agg_trunc_factor);

/**
 * (Optional) Defines the truncation factor for the
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Real   agg_P12_trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggPMaxElmts(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    agg_P_max_elmts);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    agg_P12_max_elmts);

/**
 * (Optional) Allows the user to incorporate additional vectors
 * into the interpolation for systems AMG, e.g. rigid body modes for
 * linear elasticity problems.
 * This can only be used in context with nodal coarsening and still
 * requires the user to choose an interpolation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVectors (NALU_HYPRE_Solver     solver,
                                           NALU_HYPRE_Int        num_vectors,
                                           NALU_HYPRE_ParVector *interp_vectors );

/**
 * (Optional) Defines the interpolation variant used for
 * NALU_HYPRE_BoomerAMGSetInterpVectors:
 *    - 1 : GM approach 1
 *    - 2 : GM approach 2  (to be preferred over 1)
 *    - 3 : LN approach
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecVariant (NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int    var );

/**
 * (Optional) Defines the maximal elements per row for Q, the additional
 * columns added to the original interpolation matrix P, to reduce complexity.
 * The default is no truncation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecQMax (NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    q_max );

/**
 * (Optional) Defines a truncation factor for Q, the additional
 * columns added to the original interpolation matrix P, to reduce complexity.
 * The default is no truncation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc (NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real   q_trunc );

/**
 * (Optional) Specifies the use of GSMG - geometrically smooth
 * coarsening and interpolation. Currently any nonzero value for
 * gsmg will lead to the use of GSMG.
 * The default is 0, i.e. (GSMG is not used)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGSMG(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    gsmg);

/**
 * (Optional) Defines the number of sample vectors used in GSMG
 * or LS interpolation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumSamples(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    num_samples);
/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set \e cycle_type to 1, for a W-cycle
 *  set \e cycle_type to 2. The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleType(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    cycle_type);
/**
 * (Optional) Specifies the use of Full multigrid cycle.
 * The default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFCycle( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    fcycle  );

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * classical additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAdditive(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    addlvl);

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * mult-additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAdditive(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    addlvl);

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * simplified mult-additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSimple(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    addlvl);

/**
 * (Optional) Defines last level where additive, mult-additive
 * or simple cycle is used.
 * The multiplicative approach is used on levels > add_last_lvl.
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddLastLvl(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    add_last_lvl);

/**
 * (Optional) Defines the truncation factor for the
 * smoothed interpolation used for mult-additive or simple method.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(NALU_HYPRE_Solver solver,
                                               NALU_HYPRE_Real   add_trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * smoothed interpolation used for mult-additive or simple method.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Int    add_P_max_elmts);
/**
 * (Optional) Defines the relaxation type used in the (mult)additive cycle
 * portion (also affects simple method.)
 * The default is 18 (L1-Jacobi).
 * Currently the only other option allowed is 0 (Jacobi) which should be
 * used in combination with NALU_HYPRE_BoomerAMGSetAddRelaxWt.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddRelaxType(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    add_rlx_type);

/**
 * (Optional) Defines the relaxation weight used for Jacobi within the
 * (mult)additive or simple cycle portion.
 * The default is 1.
 * The weight only affects the Jacobi method, and has no effect on L1-Jacobi
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetAddRelaxWt(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Real    add_rlx_wt);

/**
 * (Optional) Sets maximal size for agglomeration or redundant coarse grid solve.
 * When the system is smaller than this threshold, sequential AMG is used
 * on process 0 or on all remaining active processes (if redundant = 1 ).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSeqThreshold(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    seq_threshold);
/**
 * (Optional) operates switch for redundancy. Needs to be used with
 * NALU_HYPRE_BoomerAMGSetSeqThreshold. Default is 0, i.e. no redundancy.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRedundant(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    redundant);

/**
 * (Optional) Defines the number of sweeps for the fine and coarse grid,
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use NALU_HYPRE_BoomerAMGSetNumSweeps or NALU_HYPRE_BoomerAMGSetCycleNumSweeps instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumGridSweeps(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int    *num_grid_sweeps);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to \e num_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumSweeps(NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int     num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for \e k:
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleNumSweeps(NALU_HYPRE_Solver  solver,
                                           NALU_HYPRE_Int     num_sweeps,
                                           NALU_HYPRE_Int     k);

/**
 * (Optional) Defines which smoother is used on the fine and coarse grid,
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use NALU_HYPRE_BoomerAMGSetRelaxType or NALU_HYPRE_BoomerAMGSetCycleRelaxType instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGridRelaxType(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int    *grid_relax_type);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is \f$\ell_1\f$-Gauss-Seidel, forward solve (13)
 * on the down cycle and backward solve (14) on the up cycle.
 *
 * There are the following options for \e relax_type:
 *
 *    - 0  : Jacobi
 *    - 1  : Gauss-Seidel, sequential (very slow!)
 *    - 2  : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    - 3  : hybrid Gauss-Seidel or SOR, forward solve
 *    - 4  : hybrid Gauss-Seidel or SOR, backward solve
 *    - 5  : hybrid chaotic Gauss-Seidel (works only with OpenMP)
 *    - 6  : hybrid symmetric Gauss-Seidel or SSOR
 *    - 8  : \f$\ell_1\f$-scaled hybrid symmetric Gauss-Seidel
 *    - 9  : Gaussian elimination (only on coarsest level)
 *    - 13 : \f$\ell_1\f$ Gauss-Seidel, forward solve
 *    - 14 : \f$\ell_1\f$ Gauss-Seidel, backward solve
 *    - 15 : CG (warning - not a fixed smoother - may require FGMRES)
 *    - 16 : Chebyshev
 *    - 17 : FCF-Jacobi
 *    - 18 : \f$\ell_1\f$-scaled jacobi
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxType(NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int     relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of \e relax_type see
 * description of NALU_HYPRE_BoomerAMGSetRelaxType). Options for \e k are
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCycleRelaxType(NALU_HYPRE_Solver  solver,
                                           NALU_HYPRE_Int     relax_type,
                                           NALU_HYPRE_Int     k);

/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for \e relax_order:
 *
 *    - 0 : the points are relaxed in natural or lexicographic order on each processor
 *    - 1 : CF-relaxation is used, i.e on the fine grid and the down cycle the
 *          coarse points are relaxed first, followed by the fine points; on the
 *          up cycle the F-points are relaxed first, followed by the C-points.
 *          On the coarsest level, if an iterative scheme is used, the points
 *          are relaxed in lexicographic order.
 *
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxOrder(NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int     relax_order);

/**
 * (Optional) Defines in which order the points are relaxed.
 *
 * Note: This routine will be phased out!!!!
 * Use NALU_HYPRE_BoomerAMGSetRelaxOrder instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGridRelaxPoints(NALU_HYPRE_Solver   solver,
                                            NALU_HYPRE_Int    **grid_relax_points);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR.
 *
 * Note: This routine will be phased out!!!!
 * Use NALU_HYPRE_BoomerAMGSetRelaxWt or NALU_HYPRE_BoomerAMGSetLevelRelaxWt instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxWeight(NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Real   *relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on all levels.
 *
 * Values for \e relax_weight are
 *    - > 0  : this assigns the given relaxation weight on all levels
 *    - = 0  : the weight is determined on each level with the estimate
 *             \f$3 \over {4\|D^{-1/2}AD^{-1/2}\|}\f$, where \f$D\f$ is the diagonal of \f$A\f$
 *             (this should only be used with Jacobi)
 *    - = -k : the relaxation weight is determined with at most k CG steps on each level
 *             (this should only be used for symmetric positive definite problems)
 *
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRelaxWt(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real    relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive \e relax_weight, the parameter is
 * determined on the given level as described for NALU_HYPRE_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelRelaxWt(NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Real    relax_weight,
                                         NALU_HYPRE_Int     level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR.
 * Note: This routine will be phased out!!!!
 * Use NALU_HYPRE_BoomerAMGSetOuterWt or NALU_HYPRE_BoomerAMGSetLevelOuterWt instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOmega(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Real   *omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 *
 * Values for \e omega are
 *    - > 0  : this assigns the same outer relaxation weight omega on each level
 *    - = -k : an outer relaxation weight is determined with at most k CG steps on each level
 *             (this only makes sense for symmetric positive definite problems and smoothers
 *              such as SSOR)
 *
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOuterWt(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real    omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for NALU_HYPRE_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevelOuterWt(NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Real    omega,
                                         NALU_HYPRE_Int     level);

/**
 * (Optional) Defines the Order for Chebyshev smoother.
 *  The default is 2 (valid options are 1-4).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyOrder(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    order);

/**
 * (Optional) Fraction of the spectrum to use for the Chebyshev smoother.
 *  The default is .3 (i.e., damp on upper 30% of the spectrum).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyFraction (NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real   ratio);

/**
 * (Optional) Defines whether matrix should be scaled.
 *  The default is 1 (i.e., scaled).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyScale (NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int   scale);

/**
 * (Optional) Defines which polynomial variant should be used.
 *  The default is 0 (i.e., scaled).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyVariant (NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int   variant);

/**
 * (Optional) Defines how to estimate eigenvalues.
 *  The default is 10 (i.e., 10 CG iterations are used to find extreme
 *  eigenvalues.) If eig_est=0, the largest eigenvalue is estimated
 *  using Gershgorin, the smallest is set to 0.
 *  If eig_est is a positive number n, n iterations of CG are used to
 *  determine the smallest and largest eigenvalue.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetChebyEigEst (NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int   eig_est);

/**
 * (Optional) Enables the use of more complex smoothers.
 * The following options exist for \e smooth_type:
 *
 *    - 4 : FSAI (routines needed to set: NALU_HYPRE_BoomerAMGSetFSAIMaxSteps,
 *          NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize, NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters,
 *          NALU_HYPRE_BoomerAMGSetFSAIKapTolerance)
 *    - 5 : ParILUK (routines needed to set: NALU_HYPRE_ILUSetLevelOfFill, NALU_HYPRE_ILUSetType)
 *    - 6 : Schwarz (routines needed to set: NALU_HYPRE_BoomerAMGSetDomainType,
 *          NALU_HYPRE_BoomerAMGSetOverlap, NALU_HYPRE_BoomerAMGSetVariant,
 *          NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight)
 *    - 7 : Pilut (routines needed to set: NALU_HYPRE_BoomerAMGSetDropTol,
 *          NALU_HYPRE_BoomerAMGSetMaxNzPerRow)
 *    - 8 : ParaSails (routines needed to set: NALU_HYPRE_BoomerAMGSetSym,
 *          NALU_HYPRE_BoomerAMGSetLevel, NALU_HYPRE_BoomerAMGSetFilter,
 *          NALU_HYPRE_BoomerAMGSetThreshold)
 *    - 9 : Euclid (routines needed to set: NALU_HYPRE_BoomerAMGSetEuclidFile)
 *
 * The default is 6.  Also, if no smoother parameters are set via the routines
 * mentioned in the table above, default values are used.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothType(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    smooth_type);

/**
 * (Optional) Sets the number of levels for more complex smoothers.
 * The smoothers,
 * as defined by NALU_HYPRE_BoomerAMGSetSmoothType, will be used
 * on level 0 (the finest level) through level \e smooth_num_levels-1.
 * The default is 0, i.e. no complex smoothers are used.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothNumLevels(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    smooth_num_levels);

/**
 * (Optional) Sets the number of sweeps for more complex smoothers.
 * The default is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    smooth_num_sweeps);

/**
 * (Optional) Defines which variant of the Schwarz method is used.
 * The following options exist for \e variant:
 *
 *    - 0 : hybrid multiplicative Schwarz method (no overlap across processor boundaries)
 *    - 1 : hybrid additive Schwarz method (no overlap across processor boundaries)
 *    - 2 : additive Schwarz method
 *    - 3 : hybrid multiplicative Schwarz method (with overlap across processor boundaries)
 *
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetVariant(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    variant);

/**
 * (Optional) Defines the overlap for the Schwarz method.
 * The following options exist for overlap:
 *
 *    - 0 : no overlap
 *    - 1 : minimal overlap (default)
 *    - 2 : overlap generated by including all neighbors of domain boundaries
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetOverlap(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    overlap);

/**
 * (Optional) Defines the type of domain used for the Schwarz method.
 * The following options exist for \e domain_type:
 *
 *    - 0 : each point is a domain
 *    - 1 : each node is a domain (only of interest in "systems" AMG)
 *    - 2 : each domain is generated by agglomeration (default)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDomainType(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    domain_type);

/**
 * (Optional) Defines a smoothing parameter for the additive Schwarz method.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real   schwarz_rlx_weight);

/**
 *  (Optional) Indicates that the aggregates may not be SPD for the Schwarz method.
 * The following options exist for \e use_nonsymm:
 *
 *    - 0 : assume SPD (default)
 *    - 1 : assume non-symmetric
**/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int    use_nonsymm);

/**
 * (Optional) Defines symmetry for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSym(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    sym);

/**
 * (Optional) Defines number of levels for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLevel(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    level);

/**
 * (Optional) Defines threshold for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetThreshold(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   threshold);

/**
 * (Optional) Defines filter for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFilter(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   filter);

/**
 * (Optional) Defines drop tolerance for PILUT.
 * For further explanation see description of PILUT.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDropTol(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   drop_tol);

/**
 * (Optional) Defines maximal number of nonzeros for PILUT.
 * For further explanation see description of PILUT.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetMaxNzPerRow(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    max_nz_per_row);

/**
 * (Optional) Defines name of an input file for Euclid parameters.
 * For further explanation see description of Euclid.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuclidFile(NALU_HYPRE_Solver  solver,
                                       char         *euclidfile);

/**
 * (Optional) Defines number of levels for ILU(k) in Euclid.
 * For further explanation see description of Euclid.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuLevel(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    eu_level);

/**
 * (Optional) Defines filter for ILU(k) for Euclid.
 * For further explanation see description of Euclid.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuSparseA(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   eu_sparse_A);

/**
 * (Optional) Defines use of block jacobi ILUT for Euclid.
 * For further explanation see description of Euclid.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetEuBJ(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    eu_bj);


/**
 * Defines type of ILU smoother to use
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUType( NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Int         ilu_type);

/**
 * Defines level k for ILU(k) smoother
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULevel( NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int         ilu_lfil);

/**
 * Defines max row nonzeros for ILUT smoother
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUMaxRowNnz( NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int         ilu_max_row_nnz);

/**
 * Defines number of iterations for ILU smoother on each level
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUMaxIter( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int         ilu_max_iter);

/**
 * Defines drop tolorance for iLUT smoother
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUDroptol( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Real        ilu_droptol);

/**
 * (Optional) Defines maximum number of steps for FSAI.
 * For further explanation see description of FSAI.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    max_steps);

/**
 * (Optional) Defines maximum step size for FSAI.
 * For further explanation see description of FSAI.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    max_step_size);

/**
 * (Optional) Defines maximum number of iterations for estimating the
 * largest eigenvalue of the FSAI preconditioned matrix (G^T * G * A).
 * For further explanation see description of FSAI.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    eig_max_iters);

/**
 * (Optional) Defines the kaporin dropping tolerance.
 * For further explanation see description of FSAI.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real   kap_tolerance);

/**
 * (Optional) Defines triangular solver for ILU(k,T) smoother: 0-iterative, 1-direct (default)
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUTriSolve( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int     ilu_tri_solve);

/**
 * (Optional) Defines number of lower Jacobi iterations for ILU(k,T) smoother triangular solve.
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULowerJacobiIters( NALU_HYPRE_Solver  solver,
                                                 NALU_HYPRE_Int     ilu_lower_jacobi_iters);

/**
 * (Optional) Defines number of upper Jacobi iterations for ILU(k,T) smoother triangular solve.
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILUUpperJacobiIters( NALU_HYPRE_Solver  solver,
                                                 NALU_HYPRE_Int     ilu_upper_jacobi_iters);

/**
 * Set Local Reordering paramter (1==RCM, 0==None)
 * For further explanation see description of ILU.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetILULocalReordering( NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Int    ilu_reordering_type);

/**
 * (Optional) Defines which parallel restriction operator is used.
 * There are the following options for restr_type:
 *
 *    - 0 : \f$P^T\f$ - Transpose of the interpolation operator
 *    - 1 : AIR-1 - Approximate Ideal Restriction (distance 1)
 *    - 2 : AIR-2 - Approximate Ideal Restriction (distance 2)
 *
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRestriction(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    restr_par);

/**
 * (Optional) Assumes the matrix is triangular in some ordering
 * to speed up the setup time of approximate ideal restriction.
 *
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetIsTriangular(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int   is_triangular);

/**
 * (Optional) Set local problem size at which GMRES is used over
 * a direct solve in approximating ideal restriction.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetGMRESSwitchR(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int   gmres_switch);

/**
 * (Optional) Defines the drop tolerance for the A-matrices
 * from the 2nd level of AMG.
 * The default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetADropTol( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Real    A_drop_tol  );

/**
 * (Optional) Drop the entries that are not on the diagonal and smaller than
 * its row norm: type 1: 1-norm, 2: 2-norm, -1: infinity norm
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetADropType( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int     A_drop_type  );

/**
 * (Optional) Name of file to which BoomerAMG will print;
 * cf NALU_HYPRE_BoomerAMGSetPrintLevel.  (Presently this is ignored).
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPrintFileName(NALU_HYPRE_Solver  solver,
                                          const char   *print_file_name);

/**
 * (Optional) Requests automatic printing of setup and solve information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 *    - 2 : print solve information
 *    - 3 : print both setup and solve information
 *
 * Note, that if one desires to print information and uses BoomerAMG as a
 * preconditioner, suggested \e print_level is 1 to avoid excessive output,
 * and use \e print_level of solver for solve phase information.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPrintLevel(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    print_level);

/**
 * (Optional) Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default to 0 for do nothing.  The latest
 * residual will be available if logging > 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetLogging(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    logging);


/**
 * (Optional)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDebugFlag(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    debug_flag);

/**
 * (Optional) This routine will be eliminated in the future.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGInitGridRelaxation(NALU_HYPRE_Int    **num_grid_sweeps_ptr,
                                            NALU_HYPRE_Int    **grid_relax_type_ptr,
                                            NALU_HYPRE_Int   ***grid_relax_points_ptr,
                                            NALU_HYPRE_Int      coarsen_type,
                                            NALU_HYPRE_Real **relax_weights_ptr,
                                            NALU_HYPRE_Int      max_levels);

/**
 * (Optional) If rap2 not equal 0, the triple matrix product RAP is
 * replaced by two matrix products.
 * (Required for triple matrix product generation on GPUs)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetRAP2(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    rap2);

/**
 * (Optional) If mod_rap2 not equal 0, the triple matrix product RAP is
 * replaced by two matrix products with modularized kernels
 * (Required for triple matrix product generation on GPUs)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetModuleRAP2(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    mod_rap2);

/**
 * (Optional) If set to 1, the local interpolation transposes will
 * be saved to use more efficient matvecs instead of matvecTs
 * (Recommended for efficient use on GPUs)
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetKeepTranspose(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    keepTranspose);

/**
 * NALU_HYPRE_BoomerAMGSetPlotGrids
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPlotGrids (NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    plotgrids);

/**
 * NALU_HYPRE_BoomerAMGSetPlotFilename
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPlotFileName (NALU_HYPRE_Solver  solver,
                                          const char   *plotfilename);

/**
 * NALU_HYPRE_BoomerAMGSetCoordDim
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoordDim (NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    coorddim);

/**
 * NALU_HYPRE_BoomerAMGSetCoordinates
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCoordinates (NALU_HYPRE_Solver  solver,
                                         float        *coordinates);

/**
 * (Optional) Get the coarse grid hierarchy. Assumes input/ output array is
 * preallocated to the size of the local matrix. On return, \e cgrid[i] returns
 * the last grid level containing node \e i.
 *
 * @param solver [IN] solver or preconditioner
 * @param cgrid [IN/ OUT] preallocated array. On return, contains grid hierarchy info.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGGetGridHierarchy(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int *cgrid );

#ifdef NALU_HYPRE_USING_DSUPERLU
/**
 * NALU_HYPRE_BoomerAMGSetDSLUThreshold
 *
 * Usage:
 *  Set slu_threshold >= max_coarse_size (from NALU_HYPRE_BoomerAMGSetMaxCoarseSize(...))
 *  to turn on use of superLU for the coarse grid solve. SuperLU is used if the
 *  coarse grid size > max_coarse_size and the grid level is < (max_num_levels - 1)
 *  (set with NALU_HYPRE_BoomerAMGSetMaxLevels(...)).
 **/

NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetDSLUThreshold (NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    slu_threshold);
#endif

/**
 * (Optional) Fix C points to be kept till a specified coarse level.
 *
 * @param solver [IN] solver or preconditioner
 * @param cpt_coarse_level [IN] coarse level up to which to keep C points
 * @param num_cpt_coarse [IN] number of C points to be kept
 * @param cpt_coarse_index [IN] indexes of C points to be kept
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCPoints(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int     cpt_coarse_level,
                                    NALU_HYPRE_Int     num_cpt_coarse,
                                    NALU_HYPRE_BigInt *cpt_coarse_index);

/**
 * (Optional) Deprecated function. Use NALU_HYPRE_BoomerAMGSetCPoints instead.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCpointsToKeep(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int  cpt_coarse_level,
                                          NALU_HYPRE_Int  num_cpt_coarse,
                                          NALU_HYPRE_BigInt *cpt_coarse_index);

/**
 * (Optional) Set fine points in the first level.
 *
 * @param solver [IN] solver or preconditioner
 * @param num_fpt [IN] number of fine points
 * @param fpt_index [IN] global indices of fine points
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetFPoints(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int     num_fpt,
                                    NALU_HYPRE_BigInt *fpt_index);

/**
 * (Optional) Set isolated fine points in the first level.
 * Interpolation weights are not computed for these points.
 *
 * @param solver [IN] solver or preconditioner
 * @param num_isolated_fpt [IN] number of isolated fine points
 * @param isolated_fpt_index [IN] global indices of isolated fine points
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetIsolatedFPoints(NALU_HYPRE_Solver  solver,
                                            NALU_HYPRE_Int     num_isolated_fpt,
                                            NALU_HYPRE_BigInt *isolated_fpt_index);

/**
 * (Optional) if Sabs equals 1, the strength of connection test is based
 * on the absolute value of the matrix coefficients
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetSabs (NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int Sabs );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BoomerAMGDD Solver and Preconditioner
 *
 * Communication reducing solver and preconditioner built on top of algebraic multigrid
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDCreate( NALU_HYPRE_Solver *solver );

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDDestroy( NALU_HYPRE_Solver solver );

/**
 * Set up the BoomerAMGDD solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSetup( NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x );

/**
 * Solve the system or apply AMG-DD as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGDDSolve( NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x );

/**
 * (Optional) Set the number of pre- and post-relaxations per level for
 * AMG-DD inner FAC cycles. Default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACNumRelax( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    amgdd_fac_num_relax );

/**
 * (Optional) Set the number of inner FAC cycles per AMG-DD iteration.
 * Default is 2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACNumCycles( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_num_cycles );

/**
 * (Optional) Set the cycle type for the AMG-DD inner FAC cycles.
 * 1 (default) = V-cycle, 2 = W-cycle, 3 = F-cycle
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACCycleType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_cycle_type );

/**
 * (Optional) Set the relaxation type for the AMG-DD inner FAC cycles.
 * 0 = Jacobi, 1 = Gauss-Seidel, 2 = ordered Gauss-Seidel, 3 (default) = C/F L1-scaled Jacobi
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACRelaxType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    amgdd_fac_relax_type );

/**
 * (Optional) Set the relaxation weight for the AMG-DD inner FAC cycles. Default is 1.0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetFACRelaxWeight( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   amgdd_fac_relax_weight );

/**
 * (Optional) Set the AMG-DD start level. Default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetStartLevel( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    start_level );

/**
 * (Optional) Set the AMG-DD padding. Default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetPadding( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    padding );

/**
 * (Optional) Set the AMG-DD number of ghost layers. Default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetNumGhostLayers( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    num_ghost_layers );

/**
 * (Optional) Pass a custom user-defined function as a relaxation method for the AMG-DD FAC cycles.
 * Function should have the following form, where amgdd_solver is of type hypre_ParAMGDDData* and level is the level on which to relax:
 * NALU_HYPRE_Int userFACRelaxation( NALU_HYPRE_Solver amgdd_solver, NALU_HYPRE_Int level )
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDSetUserFACRelaxation( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_param ) );

/**
 * (Optional) Get the underlying AMG hierarchy as a NALU_HYPRE_Solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetAMG( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Solver *amg_solver );

/**
 * Returns the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                               NALU_HYPRE_Real   *rel_resid_norm );

/**
 * Returns the number of iterations taken.
 **/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDDGetNumIterations( NALU_HYPRE_Solver   solver,
                                   NALU_HYPRE_Int     *num_iterations );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR FSAI Solver and Preconditioner
 *
 * An adaptive factorized sparse approximate inverse solver/preconditioner/smoother
 * that computes a sparse approximation G to the inverse of the lower cholesky
 * factor of A such that M^{-1} \approx G^T * G.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAICreate( NALU_HYPRE_Solver *solver );

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAIDestroy( NALU_HYPRE_Solver solver );

/**
 * Set up the FSAI solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetup( NALU_HYPRE_Solver       solver,
                           NALU_HYPRE_ParCSRMatrix A,
                           NALU_HYPRE_ParVector    b,
                           NALU_HYPRE_ParVector    x );

/**
 * Solve the system or apply FSAI as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISolve( NALU_HYPRE_Solver       solver,
                           NALU_HYPRE_ParCSRMatrix A,
                           NALU_HYPRE_ParVector    b,
                           NALU_HYPRE_ParVector    x );

/**
 * (Optional) Sets the algorithm type used to compute the lower triangular factor G
 *
 *      - 1: Native (can use OpenMP with static scheduling)
 *      - 2: OpenMP with dynamic scheduling
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetAlgoType( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    algo_type );

/**
 * (Optional) Sets the maximum number of steps for computing the sparsity
 * pattern of G
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxSteps( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    max_steps  );

/**
 * (Optional) Sets the maximum step size for computing the sparsity pattern of G
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxStepSize( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    max_step_size  );

/**
 * (Optional) Sets the kaporin gradient reduction factor for computing the
 *  sparsity pattern of G
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetKapTolerance( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real   kap_tolerance  );

/**
 * (Optional) Sets the relaxation factor for FSAI
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetOmega( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   omega );

/**
 * (Optional) Sets the maximum number of iterations (sweeps) for FSAI
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetMaxIterations( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    max_iterations );

/**
 * (Optional) Set number of iterations for computing maximum
 * eigenvalue of the preconditioned operator.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetEigMaxIters( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    eig_max_iters );

/**
 * (Optional) Set the convergence tolerance, if FSAI is used
 * as a solver. When using FSAI as a preconditioner, set the tolerance
 * to 0.0. The default is \f$10^{-6}\f$.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetTolerance( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   tolerance );

/**
 * (Optional) Requests automatic printing of setup information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetPrintLevel(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    print_level);

/**
 * (Optional) Use a zero initial guess. This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int NALU_HYPRE_FSAISetZeroGuess(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    zero_guess);


/**@}*/


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ParaSails Preconditioner
 *
 * Parallel sparse approximate inverse preconditioner for the
 * ParCSR matrix format.
 *
 * @{
 **/

/**
 * Create a ParaSails preconditioner.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsCreate(MPI_Comm      comm,
                                NALU_HYPRE_Solver *solver);

/**
 * Destroy a ParaSails preconditioner.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsDestroy(NALU_HYPRE_Solver solver);

/**
 * Set up the ParaSails preconditioner.  This function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetup(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

/**
 * Apply the ParaSails preconditioner.  This function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSolve(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

/**
 * Set the threshold and levels parameter for the ParaSails
 * preconditioner.  The accuracy and cost of ParaSails are
 * parameterized by these two parameters.  Lower values of the
 * threshold parameter and higher values of levels parameter
 * lead to more accurate, but more expensive preconditioners.
 *
 * @param solver [IN] Preconditioner object for which to set parameters.
 * @param thresh [IN] Value of threshold parameter, \f$0 \le\f$ thresh \f$\le 1\f$.
 *                    The default value is 0.1.
 * @param nlevels [IN] Value of levels parameter, \f$0 \le\f$ nlevels.
 *                     The default value is 1.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetParams(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   thresh,
                                   NALU_HYPRE_Int    nlevels);
/**
 * Set the filter parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set filter parameter.
 * @param filter [IN] Value of filter parameter.  The filter parameter is
 *                    used to drop small nonzeros in the preconditioner,
 *                    to reduce the cost of applying the preconditioner.
 *                    Values from 0.05 to 0.1 are recommended.
 *                    The default value is 0.1.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetFilter(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   filter);

/**
 * Set the symmetry parameter for the ParaSails preconditioner.
 *
 * Values for \e sym
 *    - 0 : nonsymmetric and/or indefinite problem, and nonsymmetric preconditioner
 *    - 1 : SPD problem, and SPD (factored) preconditioner
 *    - 2 : nonsymmetric, definite problem, and SPD (factored) preconditioner
 *
 * @param solver [IN] Preconditioner object for which to set symmetry parameter.
 * @param sym [IN] Symmetry parameter.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetSym(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    sym);

/**
 * Set the load balance parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the load balance
 *                    parameter.
 * @param loadbal [IN] Value of the load balance parameter,
 *                     \f$0 \le\f$ loadbal \f$\le 1\f$.  A zero value indicates that
 *                     no load balance is attempted; a value of unity indicates
 *                     that perfect load balance will be attempted.  The
 *                     recommended value is 0.9 to balance the overhead of
 *                     data exchanges for load balancing.  No load balancing
 *                     is needed if the preconditioner is very sparse and
 *                     fast to construct.  The default value when this
 *                     parameter is not set is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetLoadbal(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   loadbal);

/**
 * Set the pattern reuse parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the pattern reuse
 *                    parameter.
 * @param reuse [IN] Value of the pattern reuse parameter.  A nonzero value
 *                   indicates that the pattern of the preconditioner should
 *                   be reused for subsequent constructions of the
 *                   preconditioner.  A zero value indicates that the
 *                   preconditioner should be constructed from scratch.
 *                   The default value when this parameter is not set is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetReuse(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    reuse);

/**
 * Set the logging parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the logging
 *                    parameter.
 * @param logging [IN] Value of the logging parameter.  A nonzero value
 *                     sends statistics of the setup procedure to stdout.
 *                     The default value when this parameter is not set is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsSetLogging(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    logging);

/**
 * Build IJ Matrix of the sparse approximate inverse (factor).
 * This function explicitly creates the IJ Matrix corresponding to
 * the sparse approximate inverse or the inverse factor.
 * Example:  NALU_HYPRE_IJMatrix ij_A;
 *           NALU_HYPRE_ParaSailsBuildIJMatrix(solver, \&ij_A);
 *
 * @param solver [IN] Preconditioner object.
 * @param pij_A [OUT] Pointer to the IJ Matrix.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParaSailsBuildIJMatrix(NALU_HYPRE_Solver    solver,
                                       NALU_HYPRE_IJMatrix *pij_A);

/* ParCSRParaSails routines */

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsCreate(MPI_Comm      comm,
                                      NALU_HYPRE_Solver *solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetup(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix A,
                                     NALU_HYPRE_ParVector    b,
                                     NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSolve(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix A,
                                     NALU_HYPRE_ParVector    b,
                                     NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetParams(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Real   thresh,
                                         NALU_HYPRE_Int    nlevels);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetFilter(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Real   filter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetSym(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    sym);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetLoadbal(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Real   loadbal);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetReuse(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    reuse);

NALU_HYPRE_Int NALU_HYPRE_ParCSRParaSailsSetLogging(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    logging);

/**@}*/

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Euclid Preconditioner
 *
 * MPI Parallel ILU preconditioner
 *
 * Options summary:
 *
 *    | Option    | Default   | Synopsis                                      |
 *    | :-------- | --------- | :-------------------------------------------- |
 *    | -level    | 1         | ILU(k) factorization level                    |
 *    | -bj       | 0 (false) | Use Block Jacobi ILU instead of PILU          |
 *    | -eu_stats | 0 (false) | Print  internal timing and statistics         |
 *    | -eu_mem   | 0 (false) | Print  internal memory usage                  |
 *
 * @{
 **/

/**
 * Create a Euclid object.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidCreate(MPI_Comm      comm,
                             NALU_HYPRE_Solver *solver);

/**
 * Destroy a Euclid object.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidDestroy(NALU_HYPRE_Solver solver);

/**
 * Set up the Euclid preconditioner.  This function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetup(NALU_HYPRE_Solver       solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector    b,
                            NALU_HYPRE_ParVector    x);

/**
 * Apply the Euclid preconditioner. This function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSolve(NALU_HYPRE_Solver       solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector    b,
                            NALU_HYPRE_ParVector    x);

/**
 * Insert (name, value) pairs in Euclid's options database
 * by passing Euclid the command line (or an array of strings).
 * All Euclid options (e.g, level, drop-tolerance) are stored in
 * this database.
 * If a (name, value) pair already exists, this call updates the value.
 * See also: NALU_HYPRE_EuclidSetParamsFromFile.
 *
 * @param argc [IN] Length of argv array
 * @param argv [IN] Array of strings
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetParams(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int     argc,
                                char         *argv[]);

/**
 * Insert (name, value) pairs in Euclid's options database.
 * Each line of the file should either begin with a "\#",
 * indicating a comment line, or contain a (name value)
 * pair, e.g:
 *
   \verbatim
   >cat optionsFile
   \#sample runtime parameter file
   -blockJacobi 3
   -matFile     /home/hysom/myfile.euclid
   -doSomething true
   -xx_coeff -1.0
   \endverbatim
 *
 * See also: NALU_HYPRE_EuclidSetParams.
 *
 * @param filename[IN] Pathname/filename to read
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetParamsFromFile(NALU_HYPRE_Solver  solver,
                                        char         *filename);

/**
 * Set level k for ILU(k) factorization, default: 1
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetLevel(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    level);

/**
 * Use block Jacobi ILU preconditioning instead of PILU
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetBJ(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    bj);

/**
 * If \e eu_stats not equal 0, a summary of runtime settings and
 * timing information is printed to stdout.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetStats(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    eu_stats);

/**
 * If \e eu_mem not equal 0, a summary of Euclid's memory usage
 * is printed to stdout.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetMem(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    eu_mem);

/**
 * Defines a drop tolerance for ILU(k). Default: 0
 * Use with NALU_HYPRE_EuclidSetRowScale.
 * Note that this can destroy symmetry in a matrix.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetSparseA(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Real   sparse_A);

/**
 * If \e row_scale not equal 0, values are scaled prior to factorization
 * so that largest value in any row is +1 or -1.
 * Note that this can destroy symmetry in a matrix.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetRowScale(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    row_scale);

/**
 * uses ILUT and defines a drop tolerance relative to the largest
 * absolute value of any entry in the row being factored.
 **/
NALU_HYPRE_Int NALU_HYPRE_EuclidSetILUT(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   drop_tol);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Pilut Preconditioner
 *
 * @{
 **/

/**
 * Create a preconditioner object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutCreate(MPI_Comm      comm,
                                  NALU_HYPRE_Solver *solver);

/**
 * Destroy a preconditioner object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutDestroy(NALU_HYPRE_Solver solver);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetup(NALU_HYPRE_Solver       solver,
                                 NALU_HYPRE_ParCSRMatrix A,
                                 NALU_HYPRE_ParVector    b,
                                 NALU_HYPRE_ParVector    x);

/**
 * Precondition the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSolve(NALU_HYPRE_Solver       solver,
                                 NALU_HYPRE_ParCSRMatrix A,
                                 NALU_HYPRE_ParVector    b,
                                 NALU_HYPRE_ParVector    x);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetMaxIter(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    max_iter);

/**
 * (Optional)
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetDropTolerance(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Real   tol);

/**
 * (Optional)
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetFactorRowSize(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    size);


NALU_HYPRE_Int NALU_HYPRE_ParCSRPilutSetLogging(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    logging );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR AMS Solver and Preconditioner
 *
 * Parallel auxiliary space Maxwell solver and preconditioner
 *
 * @{
 **/

/**
 * Create an AMS solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSCreate(NALU_HYPRE_Solver *solver);

/**
 * Destroy an AMS solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSDestroy(NALU_HYPRE_Solver solver);

/**
 * Set up the AMS solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetup(NALU_HYPRE_Solver       solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector    b,
                         NALU_HYPRE_ParVector    x);

/**
 * Solve the system or apply AMS as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSolve(NALU_HYPRE_Solver       solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector    b,
                         NALU_HYPRE_ParVector    x);

/**
 * (Optional) Sets the problem dimension (2 or 3). The default is 3.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetDimension(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    dim);

/**
 * Sets the discrete gradient matrix \e G.
 * This function should be called before NALU_HYPRE_AMSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetDiscreteGradient(NALU_HYPRE_Solver       solver,
                                       NALU_HYPRE_ParCSRMatrix G);

/**
 * Sets the \e x, \e y and \e z coordinates of the vertices in the mesh.
 *
 * Either NALU_HYPRE_AMSSetCoordinateVectors() or NALU_HYPRE_AMSSetEdgeConstantVectors()
 * should be called before NALU_HYPRE_AMSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetCoordinateVectors(NALU_HYPRE_Solver    solver,
                                        NALU_HYPRE_ParVector x,
                                        NALU_HYPRE_ParVector y,
                                        NALU_HYPRE_ParVector z);

/**
 * Sets the vectors \e Gx, \e Gy and \e Gz which give the representations of
 * the constant vector fields (1,0,0), (0,1,0) and (0,0,1) in the
 * edge element basis.
 *
 * Either NALU_HYPRE_AMSSetCoordinateVectors() or NALU_HYPRE_AMSSetEdgeConstantVectors()
 * should be called before NALU_HYPRE_AMSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetEdgeConstantVectors(NALU_HYPRE_Solver    solver,
                                          NALU_HYPRE_ParVector Gx,
                                          NALU_HYPRE_ParVector Gy,
                                          NALU_HYPRE_ParVector Gz);

/**
 * (Optional) Set the (components of) the Nedelec interpolation matrix
 * \f$\Pi = [ \Pi^x, \Pi^y, \Pi^z ]\f$.
 *
 * This function is generally intended to be used only for high-order Nedelec
 * discretizations (in the lowest order case, \f$\Pi\f$ is constructed internally in
 * AMS from the discreet gradient matrix and the coordinates of the vertices),
 * though it can also be used in the lowest-order case or for other types of
 * discretizations (e.g. ones based on the second family of Nedelec elements).
 *
 * By definition, \f$\Pi\f$ is the matrix representation of the linear operator that
 * interpolates (high-order) vector nodal finite elements into the (high-order)
 * Nedelec space. The component matrices are defined as \f$\Pi^x \varphi = \Pi
 * (\varphi,0,0)\f$ and similarly for \f$\Pi^y\f$ and \f$\Pi^z\f$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of Pi should be node-based, i.e. the \f$x\f$/\f$y\f$/\f$z\f$
 * components of the first node (vertex or high-order dof) should be listed
 * first, followed by the \f$x\f$/\f$y\f$/\f$z\f$ components of the second node and so on
 * (see the documentation of NALU_HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before NALU_HYPRE_AMSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * \f$\{\Pi\}\f$ and \f$\{\Pi^x,\Pi^y,\Pi^z\}\f$ needs to be specified (though it is OK
 * to provide both).  If Pix is NULL, then scalar \f$\Pi\f$-based AMS cycles,
 * i.e. those with \e cycle_type > 10, will be unavailable. Similarly, AMS cycles
 * based on monolithic \f$\Pi\f$ (\e cycle_type < 10) require that Pi is not NULL.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetInterpolations(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix Pi,
                                     NALU_HYPRE_ParCSRMatrix Pix,
                                     NALU_HYPRE_ParCSRMatrix Piy,
                                     NALU_HYPRE_ParCSRMatrix Piz);

/**
 * (Optional) Sets the matrix \f$A_\alpha\f$ corresponding to the Poisson
 * problem with coefficient \f$\alpha\f$ (the curl-curl term coefficient in
 * the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of \f$\Pi^T\f$ is a block-diagonal version of \f$A_\Pi\f$. If this function is not
 * called, the coarse space solver on the range of \f$\Pi^T\f$ is constructed
 * as \f$\Pi^T A \Pi\f$ in NALU_HYPRE_AMSSetup(). See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaPoissonMatrix(NALU_HYPRE_Solver       solver,
                                         NALU_HYPRE_ParCSRMatrix A_alpha);

/**
 * (Optional) Sets the matrix \f$A_\beta\f$ corresponding to the Poisson
 * problem with coefficient \f$\beta\f$ (the mass term coefficient in the
 * Maxwell problem).
 *
 * If not given, the Poisson matrix will be computed in NALU_HYPRE_AMSSetup().
 * If the given matrix is NULL, we assume that \f$\beta\f$ is identically 0
 * and use two-level (instead of three-level) methods. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaPoissonMatrix(NALU_HYPRE_Solver       solver,
                                        NALU_HYPRE_ParCSRMatrix A_beta);

/**
 * (Optional) Set the list of nodes which are interior to a zero-conductivity
 * region. This way, a more robust solver is constructed, that can be iterated
 * to lower tolerance levels. A node is interior if its entry in the array is
 * 1.0. This function should be called before NALU_HYPRE_AMSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetInteriorNodes(NALU_HYPRE_Solver    solver,
                                    NALU_HYPRE_ParVector interior_nodes);

/**
 * (Optional) Set the frequency at which a projection onto the compatible
 * subspace for problems with zero-conductivity regions is performed. The
 * default value is 5.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetProjectionFrequency(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    projection_frequency);

/**
 * (Optional) Sets maximum number of iterations, if AMS is used
 * as a solver. To use AMS as a preconditioner, set the maximum
 * number of iterations to 1. The default is 20.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetMaxIter(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    maxit);

/**
 * (Optional) Set the convergence tolerance, if AMS is used
 * as a solver. When using AMS as a preconditioner, set the tolerance
 * to 0.0. The default is \f$10^{-6}\f$.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetTol(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   tol);

/**
 * (Optional) Choose which three-level solver to use. Possible values are:
 *
 *    - 1  : 3-level multiplicative solver (01210)
 *    - 2  : 3-level additive solver (0+1+2)
 *    - 3  : 3-level multiplicative solver (02120)
 *    - 4  : 3-level additive solver (010+2)
 *    - 5  : 3-level multiplicative solver (0102010)
 *    - 6  : 3-level additive solver (1+020)
 *    - 7  : 3-level multiplicative solver (0201020)
 *    - 8  : 3-level additive solver (0(1+2)0)
 *    - 11 : 5-level multiplicative solver (013454310)
 *    - 12 : 5-level additive solver (0+1+3+4+5)
 *    - 13 : 5-level multiplicative solver (034515430)
 *    - 14 : 5-level additive solver (01(3+4+5)10)
 *
 * The default is 1. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetCycleType(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is 1 (print residual norm at each step).
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetPrintLevel(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    print_level);

/**
 * (Optional) Sets relaxation parameters for \f$A\f$.
 * The defaults are 2, 1, 1.0, 1.0.
 *
 * The available options for \e relax_type are:
 *
 *    - 1 : \f$\ell_1\f$-scaled Jacobi
 *    - 2 : \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 3 : Kaczmarz
 *    - 4 : truncated version of \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 16 : Chebyshev
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetSmoothingOptions(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    relax_type,
                                       NALU_HYPRE_Int    relax_times,
                                       NALU_HYPRE_Real   relax_weight,
                                       NALU_HYPRE_Real   omega);

/**
 * (Optional) Sets AMG parameters for \f$B_\Pi\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGOptions(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    alpha_coarsen_type,
                                      NALU_HYPRE_Int    alpha_agg_levels,
                                      NALU_HYPRE_Int    alpha_relax_type,
                                      NALU_HYPRE_Real   alpha_strength_threshold,
                                      NALU_HYPRE_Int    alpha_interp_type,
                                      NALU_HYPRE_Int    alpha_Pmax);

/**
 * (Optional) Sets the coarsest level relaxation in the AMG solver for \f$B_\Pi\f$.
 * The default is 8 (l1-GS). Use 9, 19, 29 or 99 for a direct solver.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetAlphaAMGCoarseRelaxType(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int    alpha_coarse_relax_type);

/**
 * (Optional) Sets AMG parameters for \f$B_G\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGOptions(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    beta_coarsen_type,
                                     NALU_HYPRE_Int    beta_agg_levels,
                                     NALU_HYPRE_Int    beta_relax_type,
                                     NALU_HYPRE_Real   beta_strength_threshold,
                                     NALU_HYPRE_Int    beta_interp_type,
                                     NALU_HYPRE_Int    beta_Pmax);

/**
 * (Optional) Sets the coarsest level relaxation in the AMG solver for \f$B_G\f$.
 * The default is 8 (l1-GS). Use 9, 19, 29 or 99 for a direct solver.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSSetBetaAMGCoarseRelaxType(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Int    beta_coarse_relax_type);

/**
 * Returns the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSGetNumIterations(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Real   *rel_resid_norm);

/**
 * For problems with zero-conductivity regions, project the vector onto the
 * compatible subspace: \f$x = (I - G_0 (G_0^t G_0)^{-1} G_0^T) x\f$, where \f$G_0\f$ is
 * the discrete gradient restricted to the interior nodes of the regions with
 * zero conductivity. This ensures that x is orthogonal to the gradients in the
 * range of \f$G_0\f$.
 *
 * This function is typically called after the solution iteration is complete,
 * in order to facilitate the visualization of the computed field. Without it
 * the values in the zero-conductivity regions contain kernel components.
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSProjectOutGradients(NALU_HYPRE_Solver    solver,
                                       NALU_HYPRE_ParVector x);

/**
 * Construct and return the lowest-order discrete gradient matrix G using some
 * edge and vertex information. We assume that \e edge_vertex lists the edge
 * vertices consecutively, and that the orientation of all edges is consistent.
 *
 * If \e edge_orientation = 1, the edges are already oriented.
 *
 * If \e edge_orientation = 2, the orientation of edge i depends only
 * on the sign of \e edge_vertex[2*i+1] - \e edge_vertex[2*i].
 **/
NALU_HYPRE_Int NALU_HYPRE_AMSConstructDiscreteGradient(NALU_HYPRE_ParCSRMatrix  A,
                                             NALU_HYPRE_ParVector     x_coord,
                                             NALU_HYPRE_BigInt       *edge_vertex,
                                             NALU_HYPRE_Int           edge_orientation,
                                             NALU_HYPRE_ParCSRMatrix *G);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ADS Solver and Preconditioner
 *
 * Parallel auxiliary space divergence solver and preconditioner
 *
 * @{
 **/

/**
 * Create an ADS solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSCreate(NALU_HYPRE_Solver *solver);

/**
 * Destroy an ADS solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSDestroy(NALU_HYPRE_Solver solver);

/**
 * Set up the ADS solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetup(NALU_HYPRE_Solver       solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector    b,
                         NALU_HYPRE_ParVector    x);

/**
 * Solve the system or apply ADS as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSolve(NALU_HYPRE_Solver       solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector    b,
                         NALU_HYPRE_ParVector    x);

/**
 * Sets the discrete curl matrix \e C.
 * This function should be called before NALU_HYPRE_ADSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteCurl(NALU_HYPRE_Solver       solver,
                                   NALU_HYPRE_ParCSRMatrix C);

/**
 * Sets the discrete gradient matrix \e G.
 * This function should be called before NALU_HYPRE_ADSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetDiscreteGradient(NALU_HYPRE_Solver       solver,
                                       NALU_HYPRE_ParCSRMatrix G);

/**
 * Sets the \e x, \e y and \e z coordinates of the vertices in the mesh.
 * This function should be called before NALU_HYPRE_ADSSetup()!
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetCoordinateVectors(NALU_HYPRE_Solver    solver,
                                        NALU_HYPRE_ParVector x,
                                        NALU_HYPRE_ParVector y,
                                        NALU_HYPRE_ParVector z);

/**
 * (Optional) Set the (components of) the Raviart-Thomas (\f$\Pi_{RT}\f$) and the Nedelec
 * (\f$\Pi_{ND}\f$) interpolation matrices.
 *
 * This function is generally intended to be used only for high-order \f$H(div)\f$
 * discretizations (in the lowest order case, these matrices are constructed
 * internally in ADS from the discreet gradient and curl matrices and the
 * coordinates of the vertices), though it can also be used in the lowest-order
 * case or for other types of discretizations.
 *
 * By definition, \e RT_Pi and \e ND_Pi are the matrix representations of the linear
 * operators \f$\Pi_{RT}\f$ and \f$\Pi_{ND}\f$ that interpolate (high-order) vector
 * nodal finite elements into the (high-order) Raviart-Thomas and Nedelec
 * spaces. The component matrices are defined in both cases as \f$\Pi^x \varphi =
 * \Pi (\varphi,0,0)\f$ and similarly for \f$\Pi^y\f$ and \f$\Pi^z\f$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of \e RT_Pi and \e ND_Pi should be node-based, i.e. the
 * \f$x\f$/\f$y\f$/\f$z\f$ components of the first node (vertex or high-order dof) should be
 * listed first, followed by the \f$x\f$/\f$y\f$/\f$z\f$ components of the second node and
 * so on (see the documentation of NALU_HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before hypre_ADSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * \f$\{\Pi_{RT}\}\f$ and \f$\{\Pi_{RT}^x,\Pi_{RT}^y,\Pi_{RT}^z\}\f$ needs to be
 * specified (though it is OK to provide both).  If \e RT_Pix is NULL, then scalar
 * \f$\Pi\f$-based ADS cycles, i.e. those with \e cycle_type > 10, will be
 * unavailable. Similarly, ADS cycles based on monolithic \f$\Pi\f$ (\e cycle_type <
 * 10) require that \e RT_Pi is not NULL. The same restrictions hold for the sets
 * \f$\{\Pi_{ND}\}\f$ and \f$\{\Pi_{ND}^x,\Pi_{ND}^y,\Pi_{ND}^z\}\f$ -- only one of them
 * needs to be specified, and the availability of each enables different AMS
 * cycle type options.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetInterpolations(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix RT_Pi,
                                     NALU_HYPRE_ParCSRMatrix RT_Pix,
                                     NALU_HYPRE_ParCSRMatrix RT_Piy,
                                     NALU_HYPRE_ParCSRMatrix RT_Piz,
                                     NALU_HYPRE_ParCSRMatrix ND_Pi,
                                     NALU_HYPRE_ParCSRMatrix ND_Pix,
                                     NALU_HYPRE_ParCSRMatrix ND_Piy,
                                     NALU_HYPRE_ParCSRMatrix ND_Piz);
/**
 * (Optional) Sets maximum number of iterations, if ADS is used
 * as a solver. To use ADS as a preconditioner, set the maximum
 * number of iterations to 1. The default is 20.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetMaxIter(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    maxit);

/**
 * (Optional) Set the convergence tolerance, if ADS is used
 * as a solver. When using ADS as a preconditioner, set the tolerance
 * to 0.0. The default is \f$10^{-6}\f$.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetTol(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   tol);

/**
 * (Optional) Choose which auxiliary-space solver to use. Possible values are:
 *
 *    -  1 : 3-level multiplicative solver (01210)
 *    -  2 : 3-level additive solver (0+1+2)
 *    -  3 : 3-level multiplicative solver (02120)
 *    -  4 : 3-level additive solver (010+2)
 *    -  5 : 3-level multiplicative solver (0102010)
 *    -  6 : 3-level additive solver (1+020)
 *    -  7 : 3-level multiplicative solver (0201020)
 *    -  8 : 3-level additive solver (0(1+2)0)
 *    - 11 : 5-level multiplicative solver (013454310)
 *    - 12 : 5-level additive solver (0+1+3+4+5)
 *    - 13 : 5-level multiplicative solver (034515430)
 *    - 14 : 5-level additive solver (01(3+4+5)10)
 *
 * The default is 1. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetCycleType(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is 1 (print residual norm at each step).
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetPrintLevel(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    print_level);

/**
 * (Optional) Sets relaxation parameters for \f$A\f$.
 * The defaults are 2, 1, 1.0, 1.0.
 *
 * The available options for \e relax_type are:
 *
 *    - 1  : \f$\ell_1\f$-scaled Jacobi
 *    - 2  : \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 3  : Kaczmarz
 *    - 4  : truncated version of \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 16 : Chebyshev
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetSmoothingOptions(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    relax_type,
                                       NALU_HYPRE_Int    relax_times,
                                       NALU_HYPRE_Real   relax_weight,
                                       NALU_HYPRE_Real   omega);

/**
 * (Optional) Sets parameters for Chebyshev relaxation.
 * The defaults are 2, 0.3.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetChebySmoothingOptions(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    cheby_order,
                                            NALU_HYPRE_Int    cheby_fraction);

/**
 * (Optional) Sets AMS parameters for \f$B_C\f$.
 * The defaults are 11, 10, 1, 3, 0.25, 0, 0.
 * Note that \e cycle_type should be greater than 10, unless the high-order
 * interface of NALU_HYPRE_ADSSetInterpolations is being used!
 * See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetAMSOptions(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    cycle_type,
                                 NALU_HYPRE_Int    coarsen_type,
                                 NALU_HYPRE_Int    agg_levels,
                                 NALU_HYPRE_Int    relax_type,
                                 NALU_HYPRE_Real   strength_threshold,
                                 NALU_HYPRE_Int    interp_type,
                                 NALU_HYPRE_Int    Pmax);

/**
 * (Optional) Sets AMG parameters for \f$B_\Pi\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSSetAMGOptions(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    coarsen_type,
                                 NALU_HYPRE_Int    agg_levels,
                                 NALU_HYPRE_Int    relax_type,
                                 NALU_HYPRE_Real   strength_threshold,
                                 NALU_HYPRE_Int    interp_type,
                                 NALU_HYPRE_Int    Pmax);

/**
 * Returns the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSGetNumIterations(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_ADSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Real   *rel_resid_norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGCreate(MPI_Comm      comm,
                                NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetup(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSolve(NALU_HYPRE_Solver       solver,
                               NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector    b,
                               NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetTol(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetMaxIter(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    max_iter);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetStopCrit(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    stop_crit);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetTwoNorm(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    two_norm);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetRelChange(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    rel_change);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetPrecond(NALU_HYPRE_Solver            solver,
                                    NALU_HYPRE_PtrToParSolverFcn precond,
                                    NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                    NALU_HYPRE_Solver            precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetPrecond(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetLogging(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGSetPrintLevel(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetNumIterations(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                      NALU_HYPRE_Real   *norm);
/**
 * Returns the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRPCGGetResidual(NALU_HYPRE_Solver     solver,
                                     NALU_HYPRE_ParVector *residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRDiagScaleSetup(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix A,
                                     NALU_HYPRE_ParVector    y,
                                     NALU_HYPRE_ParVector    x);

/**
 * Solve routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRDiagScale(NALU_HYPRE_Solver       solver,
                                NALU_HYPRE_ParCSRMatrix HA,
                                NALU_HYPRE_ParVector    Hy,
                                NALU_HYPRE_ParVector    Hx);

/* Setup routine for on-processor triangular solve as preconditioning. */
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSetup(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix HA,
                                     NALU_HYPRE_ParVector    Hy,
                                     NALU_HYPRE_ParVector    Hx);

/* Solve routine for on-processor triangular solve as preconditioning. */
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSolve(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix HA,
                                     NALU_HYPRE_ParVector    Hy,
                                     NALU_HYPRE_ParVector    Hx);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESCreate(MPI_Comm      comm,
                                  NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetup(NALU_HYPRE_Solver       solver,
                                 NALU_HYPRE_ParCSRMatrix A,
                                 NALU_HYPRE_ParVector    b,
                                 NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSolve(NALU_HYPRE_Solver       solver,
                                 NALU_HYPRE_ParCSRMatrix A,
                                 NALU_HYPRE_ParVector    b,
                                 NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetKDim(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    k_dim);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetTol(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    max_iter);

/*
 * Obsolete
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetStopCrit(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    stop_crit);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetPrecond(NALU_HYPRE_Solver             solver,
                                      NALU_HYPRE_PtrToParSolverFcn  precond,
                                      NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                                      NALU_HYPRE_Solver             precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetLogging(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                            NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                        NALU_HYPRE_Real   *norm);
/**
 * Returns the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRGMRESGetResidual(NALU_HYPRE_Solver     solver,
                                       NALU_HYPRE_ParVector *residual);


/* ParCSR CO-GMRES, author: KS */

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESCreate(MPI_Comm      comm,
                                    NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetup(NALU_HYPRE_Solver       solver,
                                   NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector    b,
                                   NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSolve(NALU_HYPRE_Solver       solver,
                                   NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector    b,
                                   NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetKDim(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    k_dim);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetUnroll(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    unroll);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetCGS(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    cgs);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetTol(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    max_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetPrecond(NALU_HYPRE_Solver             solver,
                                        NALU_HYPRE_PtrToParSolverFcn  precond,
                                        NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                                        NALU_HYPRE_Solver             precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetLogging(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                              NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                          NALU_HYPRE_Real   *norm);

/**
 * Returns the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRCOGMRESGetResidual(NALU_HYPRE_Solver     solver,
                                         NALU_HYPRE_ParVector *residual);

/* end of parCSR CO-GMRES */

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESCreate(MPI_Comm      comm,
                                      NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetup(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix A,
                                     NALU_HYPRE_ParVector    b,
                                     NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSolve(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix A,
                                     NALU_HYPRE_ParVector    b,
                                     NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetKDim(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    k_dim);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetTol(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    max_iter);


NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetPrecond(NALU_HYPRE_Solver             solver,
                                          NALU_HYPRE_PtrToParSolverFcn  precond,
                                          NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                                          NALU_HYPRE_Solver             precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetLogging(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                            NALU_HYPRE_Real   *norm);

NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESGetResidual(NALU_HYPRE_Solver     solver,
                                           NALU_HYPRE_ParVector *residual);


NALU_HYPRE_Int NALU_HYPRE_ParCSRFlexGMRESSetModifyPC( NALU_HYPRE_Solver           solver,
                                            NALU_HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESCreate(MPI_Comm      comm,
                                   NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetup(NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSolve(NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetKDim(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    k_dim);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetAugDim(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    aug_dim);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetTol(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    max_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetPrecond(NALU_HYPRE_Solver             solver,
                                       NALU_HYPRE_PtrToParSolverFcn  precond,
                                       NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                                       NALU_HYPRE_Solver             precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetLogging(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                             NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                         NALU_HYPRE_Real   *norm);

NALU_HYPRE_Int NALU_HYPRE_ParCSRLGMRESGetResidual(NALU_HYPRE_Solver     solver,
                                        NALU_HYPRE_ParVector *residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABCreate(MPI_Comm      comm,
                                     NALU_HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetup(NALU_HYPRE_Solver       solver,
                                    NALU_HYPRE_ParCSRMatrix A,
                                    NALU_HYPRE_ParVector    b,
                                    NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSolve(NALU_HYPRE_Solver       solver,
                                    NALU_HYPRE_ParCSRMatrix A,
                                    NALU_HYPRE_ParVector    b,
                                    NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetTol(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real   a_tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetMinIter(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetMaxIter(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    max_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetStopCrit(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    stop_crit);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetPrecond(NALU_HYPRE_Solver            solver,
                                         NALU_HYPRE_PtrToParSolverFcn precond,
                                         NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                         NALU_HYPRE_Solver            precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetPrecond(NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetLogging(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    print_level);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetNumIterations(NALU_HYPRE_Solver  solver,
                                               NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                           NALU_HYPRE_Real   *norm);

NALU_HYPRE_Int NALU_HYPRE_ParCSRBiCGSTABGetResidual(NALU_HYPRE_Solver     solver,
                                          NALU_HYPRE_ParVector *residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Hybrid Solver
 *
 * @{
 **/

/**
 *  Create solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridCreate(NALU_HYPRE_Solver *solver);
/**
 *  Destroy solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridDestroy(NALU_HYPRE_Solver solver);

/**
 *  Setup the hybrid solver
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetup(NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x);

/**
 *  Solve linear system
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSolve(NALU_HYPRE_Solver       solver,
                                  NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector    b,
                                  NALU_HYPRE_ParVector    x);
/**
 *  Set the convergence tolerance for the Krylov solver. The default is 1.e-6.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetTol(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   tol);
/**
 *  Set the absolute convergence tolerance for the Krylov solver. The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real   tol);

/**
 *  Set the desired convergence factor
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetConvergenceTol(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Real   cf_tol);

/**
 *  Set the maximal number of iterations for the diagonally
 *  preconditioned solver
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetDSCGMaxIter(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    dscg_max_its);

/**
 *  Set the maximal number of iterations for the AMG
 *  preconditioned solver
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPCGMaxIter(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    pcg_max_its);

/*
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetSetupType(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    setup_type);

/**
 *  Set the desired solver type. There are the following options:
 *     -  1 : PCG (default)
 *     -  2 : GMRES
 *     -  3 : BiCGSTAB
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetSolverType(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    solver_type);

/**
 * (Optional) Set recompute residual (don't rely on 3-term recurrence).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRecomputeResidual( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int     recompute_residual );

/**
 * (Optional) Get recompute residual option.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetRecomputeResidual( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int    *recompute_residual );

/**
 * (Optional) Set recompute residual period (don't rely on 3-term recurrence).
 *
 * Recomputes residual after every specified number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRecomputeResidualP( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int     recompute_residual_p );

/**
 * (Optional) Get recompute residual period option.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetRecomputeResidualP( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int    *recompute_residual_p );

/**
 * Set the Krylov dimension for restarted GMRES.
 * The default is 5.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetKDim(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    k_dim);

/**
 * Set the type of norm for PCG.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetTwoNorm(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    two_norm);

/**
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetStopCrit(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    stop_crit);

/**
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetRelChange(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    rel_change);

/**
 * Set preconditioner if wanting to use one that is not set up by
 * the hybrid solver.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPrecond(NALU_HYPRE_Solver            solver,
                                       NALU_HYPRE_PtrToParSolverFcn precond,
                                       NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                       NALU_HYPRE_Solver            precond_solver);

/**
 * Set logging parameter (default: 0, no logging).
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetLogging(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    logging);

/**
 * Set print level (default: 0, no printing)
 * 2 will print residual norms per iteration
 * 10 will print AMG setup information if AMG is used
 * 12 both Setup information and iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPrintLevel(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int    print_level);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For elasticity problems, a larger strength threshold, such as 0.7 or 0.8,
 * is often better.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetStrongThreshold(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real   strong_threshold);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If \e max_row_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxRowSum(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   max_row_sum);

/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetTruncFactor(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Real   trunc_factor);


/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridSetPMaxElmts(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    P_max_elmts);

/**
 * (Optional) Defines the maximal number of levels used for AMG.
 * The default is 25.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxLevels(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    max_levels);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMeasureType(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    measure_type);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for \e coarsen_type:
 *
 *    - 0  : CLJP-coarsening (a parallel coarsening algorithm using independent sets).
 *    - 1  : classical Ruge-Stueben coarsening on each processor, no boundary treatment
 *    - 3  : classical Ruge-Stueben coarsening on each processor, followed by a third
 *           pass, which adds coarse points on the boundaries
 *    - 6  : Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse
 *           points generated by 1 as its first independent set)
 *    - 7  : CLJP-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 8  : PMIS-coarsening (a parallel coarsening algorithm using independent sets
 *           with lower complexities than CLJP, might also lead to slower convergence)
 *    - 9  : PMIS-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 10 : HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently,
 *           followed by PMIS using the interior C-points as its first independent set)
 *    - 11 : one-pass Ruge-Stueben coarsening on each processor, no boundary treatment
 *
 * The default is 10.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCoarsenType(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    coarsen_type);

/**
 * (Optional) Specifies which interpolation operator is used
 * The default is ext+i interpolation truncated to at most 4 elements per row.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetInterpType(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    interp_type);

/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set \e cycle_type to 1, for a W-cycle
 *  set \e cycle_type to 2. The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleType(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    cycle_type);

/*
 *
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetGridRelaxType(NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int    *grid_relax_type);

/*
 *
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetGridRelaxPoints(NALU_HYPRE_Solver   solver,
                                     NALU_HYPRE_Int    **grid_relax_points);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to \e num_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumSweeps(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for \e k:
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleNumSweeps(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    num_sweeps,
                                    NALU_HYPRE_Int    k);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is l1-Gauss-Seidel, forward solve on the down
 * cycle (13) and backward solve on the up cycle (14).
 *
 * There are the following options for \e relax_type:
 *
 *    - 0  : Jacobi
 *    - 1  : Gauss-Seidel, sequential (very slow!)
 *    - 2  : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    - 3  : hybrid Gauss-Seidel or SOR, forward solve
 *    - 4  : hybrid Gauss-Seidel or SOR, backward solve
 *    - 6  : hybrid symmetric Gauss-Seidel or SSOR
 *    - 8  : hybrid symmetric l1-Gauss-Seidel or SSOR
 *    - 13 : l1-Gauss-Seidel, forward solve
 *    - 14 : l1-Gauss-Seidel, backward solve
 *    - 18 : l1-Jacobi
 *    - 9  : Gaussian elimination (only on coarsest level)
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxType(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of \e relax_type see
 * description of NALU_HYPRE_BoomerAMGSetRelaxType). Options for k are
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleRelaxType(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    relax_type,
                                    NALU_HYPRE_Int    k);

/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for \e relax_order:
 *
 *    - 0 : the points are relaxed in natural or lexicographic order on each processor
 *    - 1 : CF-relaxation is used, i.e on the fine grid and the down cycle the
 *          coarse points are relaxed first, followed by the fine points; on the
 *          up cycle the F-points are relaxed first, followed by the C-points.
 *          On the coarsest level, if an iterative scheme is used, the points
 *          are relaxed in lexicographic order.
 *
 * The default is 0 (CF-relaxation).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxOrder(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    relax_order);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on all levels.
 *
 * Values for \e relax_wt are
 *    - > 0  : this assigns the given relaxation weight on all levels
 *    - = 0  : the weight is determined on each level with the estimate
 *             \f$3 \over {4\|D^{-1/2}AD^{-1/2}\|}\f$, where \f$D\f$ is the diagonal of \f$A\f$
 *             (this should only be used with Jacobi)
 *    - = -k : the relaxation weight is determined with at most k CG steps on each level
 *             (this should only be used for symmetric positive definite problems)
 *
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxWt(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real   relax_wt);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive \e relax_weight, the parameter is
 * determined on the given level as described for NALU_HYPRE_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetLevelRelaxWt(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   relax_wt,
                                  NALU_HYPRE_Int    level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 *
 * Values for \e outer_wt are
 *    - > 0  : this assigns the same outer relaxation weight omega on each level
 *    - = -k : an outer relaxation weight is determined with at most k CG steps on each level
 *             (this only makes sense for symmetric positive definite problems and smoothers
 *             such as SSOR)
 *
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetOuterWt(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real   outer_wt);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for NALU_HYPRE_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetLevelOuterWt(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   outer_wt,
                                  NALU_HYPRE_Int    level);

/**
 * (Optional) Defines the maximal coarse grid size.
 * The default is 9.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxCoarseSize(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    max_coarse_size);

/**
 * (Optional) Defines the minimal coarse grid size.
 * The default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMinCoarseSize(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    min_coarse_size);

/**
 * (Optional) enables redundant coarse grid size. If the system size becomes
 * smaller than seq_threshold, sequential AMG is used on all remaining processors.
 * The default is 0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetSeqThreshold(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    seq_threshold);

/**
 *
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxWeight(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Real   *relax_weight);

/**
 *
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetOmega(NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Real   *omega);

/**
 * (Optional) Defines the number of levels of aggressive coarsening,
 * starting with the finest level.
 * The default is 0, i.e. no aggressive coarsening.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetAggNumLevels(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    agg_num_levels);

/*
 * (Optional) Defines the interpolation used on levels of aggressive coarsening
 * The default is 4, i.e. multipass interpolation.
 * The following options exist:
 *
 *    - 1 : 2-stage extended+i interpolation
 *    - 2 : 2-stage standard interpolation
 *    - 3 : 2-stage extended interpolation
 *    - 4 : multipass interpolation
 *    - 5 : 2-stage extended interpolation in matrix-matrix form
 *    - 6 : 2-stage extended+i interpolation in matrix-matrix form
 *    - 7 : 2-stage extended+e interpolation in matrix-matrix form
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetAggInterpType( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    agg_interp_type);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1, which leads to the most aggressive coarsening.
 * Setting \e num_paths to 2 will increase complexity somewhat,
 * but can lead to better convergence.**/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumPaths(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    num_paths);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumFunctions(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetDofFunc(NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int    *dof_func);
/**
 * (Optional) Sets whether to use the nodal systems version.
 * The default is 0 (the unknown based approach).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNodal(NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    nodal);

/**
 * (Optional) Sets whether to store local transposed interpolation
 * The default is 0 (don't store).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetKeepTranspose(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    keepT);

/**
 * (Optional) Sets whether to use non-Galerkin option
 * The default is no non-Galerkin option
 * num_levels sets the number of levels where to use it
 * nongalerkin_tol contains the tolerances for <num_levels> levels
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNonGalerkinTol(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int   num_levels,
                                    NALU_HYPRE_Real *nongalerkin_tol);

/**
 * Retrieves the total number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetNumIterations(NALU_HYPRE_Solver  solver,
                                             NALU_HYPRE_Int    *num_its);

/**
 * Retrieves the number of iterations used by the diagonally scaled solver.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetDSCGNumIterations(NALU_HYPRE_Solver  solver,
                                                 NALU_HYPRE_Int    *dscg_num_its);

/**
 * Retrieves the number of iterations used by the AMG preconditioned solver.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetPCGNumIterations(NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Int    *pcg_num_its);

/**
 * Retrieves the final relative residual norm.
 **/
NALU_HYPRE_Int NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                         NALU_HYPRE_Real   *norm);

/* Is this a retired function? (RDF) */
NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumGridSweeps(NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int    *num_grid_sweeps);


NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetSetupSolveTime( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real  *time    );
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Schwarz Solver
 **/

NALU_HYPRE_Int NALU_HYPRE_SchwarzCreate(NALU_HYPRE_Solver *solver);

NALU_HYPRE_Int NALU_HYPRE_SchwarzDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetup(NALU_HYPRE_Solver       solver,
                             NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector    b,
                             NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSolve(NALU_HYPRE_Solver       solver,
                             NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector    b,
                             NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetVariant(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    variant);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetOverlap(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    overlap);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDomainType(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    domain_type);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetRelaxWeight(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   relax_weight);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDomainStructure(NALU_HYPRE_Solver    solver,
                                          NALU_HYPRE_CSRMatrix domain_structure);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetNumFunctions(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    num_functions);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetDofFunc(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *dof_func);

NALU_HYPRE_Int NALU_HYPRE_SchwarzSetNonSymm(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    use_nonsymm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name ParCSR CGNR Solver
 **/

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRCreate(MPI_Comm      comm,
                                 NALU_HYPRE_Solver *solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRDestroy(NALU_HYPRE_Solver solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetup(NALU_HYPRE_Solver       solver,
                                NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector    b,
                                NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSolve(NALU_HYPRE_Solver       solver,
                                NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector    b,
                                NALU_HYPRE_ParVector    x);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetTol(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Real   tol);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetMinIter(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    min_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetMaxIter(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    max_iter);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetStopCrit(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    stop_crit);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetPrecond(NALU_HYPRE_Solver            solver,
                                     NALU_HYPRE_PtrToParSolverFcn precond,
                                     NALU_HYPRE_PtrToParSolverFcn precondT,
                                     NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                     NALU_HYPRE_Solver            precond_solver);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetPrecond(NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Solver *precond_data);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRSetLogging(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    logging);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetNumIterations(NALU_HYPRE_Solver  solver,
                                           NALU_HYPRE_Int    *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                       NALU_HYPRE_Real   *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR MGR Solver
 *
 * Parallel multigrid reduction solver and preconditioner.
 * This solver or preconditioner is designed with systems of
 * PDEs in mind. However, it can also be used for scalar linear
 * systems, particularly for problems where the user can exploit
 * information from the physics of the problem. In this way, the
 * MGR solver could potentially be used as a foundation
 * for a physics-based preconditioner.
 *
 * @{
 **/

#ifdef NALU_HYPRE_USING_DSUPERLU
/**
 * Create a MGR direct solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRDirectSolverCreate( NALU_HYPRE_Solver *solver );

/**
 * Destroy a MGR direct solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRDirectSolverDestroy( NALU_HYPRE_Solver solver );

/**
 * Setup the MGR direct solver using DSUPERLU
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRDirectSolverSetup( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_ParCSRMatrix A,
                                      NALU_HYPRE_ParVector b,
                                      NALU_HYPRE_ParVector x      );

/**
* Solve the system using DSUPERLU.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved (Ignored by this function).
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
NALU_HYPRE_Int NALU_HYPRE_MGRDirectSolverSolve( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_ParCSRMatrix A,
                                      NALU_HYPRE_ParVector b,
                                      NALU_HYPRE_ParVector x      );
#endif

/**
 * Create a solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRCreate( NALU_HYPRE_Solver *solver );

/**
 * Destroy a solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRDestroy( NALU_HYPRE_Solver solver );

/**
 * Setup the MGR solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRSetup( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      );

/**
* Solve the system or apply MGR as a preconditioner.
* If used as a preconditioner, this function should be passed
* to the iterative solver \e SetPrecond function.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
NALU_HYPRE_Int NALU_HYPRE_MGRSolve( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      );

/**
 * Set the block data assuming that the physical variables are ordered contiguously,
 * i.e. p_1, p_2, ..., p_n, s_1, s_2, ..., s_n, ...
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param block_coarse_indexes [IN] index for each block coarse point per level
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRSetCpointsByContiguousBlock( NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Int  block_size,
                                                NALU_HYPRE_Int max_num_levels,
                                                NALU_HYPRE_BigInt *idx_array,
                                                NALU_HYPRE_Int *num_block_coarse_points,
                                                NALU_HYPRE_Int  **block_coarse_indexes);

/**
 * Set the block data (by grid points) and prescribe the coarse indexes per block
 * for each reduction level.
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param block_coarse_indexes [IN] index for each block coarse point per level
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRSetCpointsByBlock( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int  block_size,
                                      NALU_HYPRE_Int max_num_levels,
                                      NALU_HYPRE_Int *num_block_coarse_points,
                                      NALU_HYPRE_Int  **block_coarse_indexes);

/*--------------------------------------------------------------------------
 * NALU_HYPRE_Int NALU_HYPRE_MGRSetCpointsByPointMarkerArray
 *--------------------------------------------------------------------------*/
/**
 * Set the coarse indices for the levels using an array of tags for all the
 * local degrees of freedom.
 * TODO: Rename the function to make it more descriptive.
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param lvl_block_coarse_indexes [IN] indices for the coarse points per level
 * @param point_marker_array [IN] array of tags for the local degrees of freedom
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRSetCpointsByPointMarkerArray( NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Int  block_size,
                                                 NALU_HYPRE_Int  max_num_levels,
                                                 NALU_HYPRE_Int  *num_block_coarse_points,
                                                 NALU_HYPRE_Int  **lvl_block_coarse_indexes,
                                                 NALU_HYPRE_Int  *point_marker_array);

/**
 * (Optional) Set non C-points to F-points.
 * This routine determines how the coarse points are selected for the next level
 * reduction. Options for \e nonCptToFptFlag are:
 *
 *    - 0 : Allow points not prescribed as C points to be potentially set as C points
 *          using classical AMG coarsening strategies (currently uses CLJP-coarsening).
 *    - 1 : Fix points not prescribed as C points to be F points for the next reduction
 *
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetNonCpointsToFpoints( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int nonCptToFptFlag);

/**
 * (Optional) Set maximum number of coarsening (or reduction) levels.
 * The default is 10.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxCoarseLevels( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int maxlev );

/**
 * (Optional) Set the system block size.
 * This should match the block size set in the MGRSetCpointsByBlock function.
 * The default is 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetBlockSize( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int bsize );

/**
 * (Optional) Defines indexes of coarse nodes to be kept to the coarsest level.
 * These indexes are passed down through the MGR hierarchy to the coarsest grid
 * of the coarse grid (BoomerAMG) solver.
 *
 * @param solver [IN] solver or preconditioner object
 * @param reserved_coarse_size [IN] number of reserved coarse points
 * @param reserved_coarse_nodes [IN] (global) indexes of reserved coarse points
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetReservedCoarseNodes( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int reserved_coarse_size,
                                 NALU_HYPRE_BigInt *reserved_coarse_nodes );

/* (Optional) Set the level for reducing the reserved Cpoints before the coarse
 * grid solve. This is necessary for some applications, such as phase transitions.
 * The default is 0 (no reduction, i.e. keep the reserved cpoints in the coarse grid solve).
 * The default setup for the reduction is as follows:
 * interp_type = 2
 * restrict_type = 0
 * F-relax method = 99
 * Galerkin coarse grid
**/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetReservedCpointsLevelToKeep( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level);

/**
 * (Optional) Set the relaxation type for F-relaxation.
 * Currently supports the following flavors of relaxation types
 * as described in the \e BoomerAMGSetRelaxType:
 * \e relax_type 0 - 8, 13, 14, 18, 19, 98.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetRelaxType(NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int relax_type );

/**
 * (Optional) Set the strategy for F-relaxation.
 * Options for \e relax_method are:
 *
 *    - 0 : Single-level relaxation sweeps for F-relaxation as prescribed by \e MGRSetRelaxType
 *    - 1 : Multi-level relaxation strategy for F-relaxation (V(1,0) cycle currently supported).
 *
 *    NOTE: This function will be removed in favor of /e NALU_HYPRE_MGRSetFLevelRelaxType!!
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetFRelaxMethod(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int relax_method );

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxMethod(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *relax_method );

/**
 * (Optional) Set the relaxation type for F-relaxation at each level.
 * This function takes precedence over, and will replace \e NALU_HYPRE_MGRSetFRelaxMethod
 * and NALU_HYPRE_MGRSetRelaxType.
 * Options for \e relax_type entries are:
 *
 *    - 0, 3 - 8, 13, 14, 18: (as described in \e BoomerAMGSetRelaxType)
 *    - 1 : Multi-level relaxation strategy for F-relaxation (V(1,0) cycle currently supported).
 *    - 2 : AMG.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxType(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int *relax_type );

/**
 * (Optional) Set the strategy for coarse grid computation.
 * Options for \e cg_method are:
 *
 *    - 0 : Galerkin coarse grid computation using RAP.
 *    - 1 - 4 : Non-Galerkin coarse grid computation with dropping strategy.
 *         - 1: inv(A_FF) approximated by its (block) diagonal inverse
 *         - 2: CPR-like approximation with inv(A_FF) approximated by its diagonal inverse
 *         - 3: CPR-like approximation with inv(A_FF) approximated by its block diagonal inverse
 *         - 4: inv(A_FF) approximated by sparse approximate inverse
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetCoarseGridMethod(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cg_method );

/**
 * (Optional) Set the number of functions for F-relaxation V-cycle.
 * For problems like elasticity, one may want to perform coarsening and
 * interpolation for block matrices. The number of functions corresponds
 * to the number of scalar PDEs in the system.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxNumFunctions(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_functions);

/**
 * (Optional) Set the strategy for computing the MGR restriction operator.
 *
 * Options for \e restrict_type are:
 *
 *    - 0    : injection \f$[0  I]\f$
 *    - 1    : unscaled (not recommended)
 *    - 2    : diagonal scaling (Jacobi)
 *    - 3    : approximate inverse
 *    - 4    : pAIR distance 1
 *    - 5    : pAIR distance 2
 *    - 12   : Block Jacobi
 *    - 13   : CPR-like restriction operator
 *    - else : use classical modified interpolation
 *
 * The default is injection.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetRestrictType( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int restrict_type);

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelRestrictType( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int *restrict_type);

/**
 * (Optional) Set number of restriction sweeps.
 * This option is for \e restrict_type > 2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumRestrictSweeps( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int nsweeps );

/**
 * (Optional) Set the strategy for computing the MGR interpolation operator.
 * Options for \e interp_type are:
 *
 *    - 0    : injection \f$[0  I]^{T}\f$
 *    - 1    : L1-Jacobi
 *    - 2    : diagonal scaling (Jacobi)
 *    - 3    : classical modified interpolation
 *    - 4    : approximate inverse
 *    - 12   : Block Jacobi
 *    - else : classical modified interpolation
 *
 * The default is diagonal scaling.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetInterpType( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int interp_type );

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelInterpType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int *interp_type );

/**
 * (Optional) Set number of relaxation sweeps.
 * This option is for the "single level" F-relaxation (\e relax_method = 0).
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumRelaxSweeps( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int nsweeps );

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelNumRelaxSweeps( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int *nsweeps );

/**
 * (Optional) Set number of interpolation sweeps.
 * This option is for \e interp_type > 2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumInterpSweeps( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int nsweeps );

/**
 * (Optional) Set block size for block (global) smoother and interp/restriction.
 * This option is for \e interp_type/restrict_type == 12, and
 * \e smooth_type == 0 or 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetBlockJacobiBlockSize( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int blk_size );

NALU_HYPRE_Int NALU_HYPRE_MGRSetFSolver(NALU_HYPRE_Solver          solver,
                              NALU_HYPRE_PtrToParSolverFcn  fine_grid_solver_solve,
                              NALU_HYPRE_PtrToParSolverFcn  fine_grid_solver_setup,
                              NALU_HYPRE_Solver          fsolver );

NALU_HYPRE_Int NALU_HYPRE_MGRBuildAff(NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_Int *CF_marker,
                            NALU_HYPRE_Int debug_flag,
                            NALU_HYPRE_ParCSRMatrix *A_ff);

/**
 * (Optional) Set the coarse grid solver.
 * Currently uses BoomerAMG.
 * The default, if not set, is BoomerAMG with default options.
 *
 * @param solver [IN] solver or preconditioner object
 * @param coarse_grid_solver_solve [IN] solve routine for BoomerAMG
 * @param coarse_grid_solver_setup [IN] setup routine for BoomerAMG
 * @param coarse_grid_solver [IN] BoomerAMG solver
 **/
NALU_HYPRE_Int NALU_HYPRE_MGRSetCoarseSolver(NALU_HYPRE_Solver          solver,
                                   NALU_HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                                   NALU_HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                                   NALU_HYPRE_Solver          coarse_grid_solver );

/**
 * (Optional) Set the print level to print setup and solve information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 *    - 2 : print solve information
 *    - 3 : print both setup and solve information
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetPrintLevel( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int print_level );

NALU_HYPRE_Int
NALU_HYPRE_MGRSetFrelaxPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int print_level );

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCoarseGridPrintLevel( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int print_level );

/**
 * (Optional) Set the threshold to compress the coarse grid at each level
 * Use threshold = 0.0 if no truncation is applied. Otherwise, set the threshold
 * value for dropping entries for the coarse grid.
 * The default is 0.0.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetTruncateCoarseGridThreshold( NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Real threshold);


/**
 * (Optional) Requests logging of solver diagnostics.
 * Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default to 0 for do nothing.  The latest
 * residual will be available if logging > 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLogging( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int logging );

/**
 * (Optional) Set maximum number of iterations if used as a solver.
 * Set this to 1 if MGR is used as a preconditioner. The default is 20.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxIter( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int max_iter );

/**
 * (Optional) Set the convergence tolerance for the MGR solver.
 * Use tol = 0.0 if MGR is used as a preconditioner. The default is 1.e-6.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetTol( NALU_HYPRE_Solver solver,
                 NALU_HYPRE_Real tol );

/**
 * (Optional) Determines how many sweeps of global smoothing to do.
 * Default is 0 (no global smoothing).
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxGlobalSmoothIters( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int smooth_iter );

/**
 * (Optional) Determines how many sweeps of global smoothing to do on each level.
 * Default is 0 (no global smoothing).
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelSmoothIters( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int *smooth_iters );
/**
 * (Optional) Set the smoothing order for global smoothing at each level.
 * Options for \e level_smooth_order are:
 *    - 1 : Pre-smoothing - Down cycle (default)
 *    - 2 : Post-smoothing - Up cycle
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetGlobalSmoothCycle( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int global_smooth_cycle );

/**
 * (Optional) Determines type of global smoother.
 * Options for \e smooth_type are:
 *
 *    - 0 : block Jacobi (default)
 *    - 1 : block Gauss-Siedel
 *    - 2 : Jacobi
 *    - 3 : Gauss-Seidel, sequential (very slow!)
 *    - 4 : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    - 5 : hybrid Gauss-Seidel or SOR, forward solve
 *    - 6 : hybrid Gauss-Seidel or SOR, backward solve
 *    - 8 : Euclid (ILU)
 *    - 16 : NALU_HYPRE_ILU
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetGlobalSmoothType( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int smooth_type );

/**
 * (Optional) Determines type of global smoother for each level.
 * See \e NALU_HYPRE_MGRSetGlobalSmoothType for global smoother options.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelSmoothType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int *smooth_type );

/**
 * (Optional) Return the number of MGR iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRGetNumIterations( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int *num_iterations );

NALU_HYPRE_Int
NALU_HYPRE_MGRGetCoarseGridConvergenceFactor (NALU_HYPRE_Solver solver, NALU_HYPRE_Real *conv_factor );

/**
 * (Optional) Set the number of maximum points for interpolation operator.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetPMaxElmts( NALU_HYPRE_Solver solver, NALU_HYPRE_Int P_max_elmts);

/**
 * (Optional) Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_MGRGetFinalRelativeResidualNorm(  NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real *res_norm );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**
 * @name ParCSR ILU Solver
 *
 * (Parallel) ILU smoother
 *
 * @{
 **/

/**
 * Create a solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_ILUCreate( NALU_HYPRE_Solver *solver );

/**
 * Destroy a solver object
 **/
NALU_HYPRE_Int NALU_HYPRE_ILUDestroy( NALU_HYPRE_Solver solver );

/**
 * Setup the ILU solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
NALU_HYPRE_Int NALU_HYPRE_ILUSetup( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      );
/**
* Solve the system or apply ILU as a preconditioner.
* If used as a preconditioner, this function should be passed
* to the iterative solver \e SetPrecond function.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
NALU_HYPRE_Int NALU_HYPRE_ILUSolve( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      );

/**
 * (Optional) Set maximum number of iterations if used as a solver.
 * Set this to 1 if ILU is used as a preconditioner. The default is 20.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );

/**
 * (Optional) Set triangular solver type (0) direct (1) iterative
 * Set this to 1 Jacobi iterations. The default is 0 direct method.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetTriSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_Int tri_solve );

/**
 * (Optional) Set number of lower Jacobi iterations for the triangular L solves
 * Set this to integer > 0 when using iterative tri_solve (0). The default is 5 iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetLowerJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lower_jacobi_iterations );

/**
 * (Optional) Set number of upper Jacobi iterations for the triangular U solves
 * Set this to integer > 0 when using iterative tri_solve (0). The default is 5 iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetUpperJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int upper_jacobi_iterations );

/**
 * (Optional) Set the convergence tolerance for the ILU smoother.
 * Use tol = 0.0 if ILU is used as a preconditioner. The default is 1.e-7.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );

/**
 * (Optional) Set the level of fill k, for level-based ILU(k)
 * The default is 0 (for ILU(0)).
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetLevelOfFill( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lfil );

/**
 * (Optional) Set the max non-zeros per row in L and U factors (for ILUT)
 * The default is 1000.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxNnzPerRow( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nzmax );

/**
 * (Optional) Set the threshold for dropping in L and U factors (for ILUT).
 * Any fill-in less than this threshold is dropped in the factorization.
 * The default is 1.0e-2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold );

/**
 * (Optional) Set the array of thresholds for dropping in ILUT.
 * B, E, and F correspond to upper left, lower left and upper right
 * of 2 x 2 block decomposition respectively.
 * Any fill-in less than threshold is dropped in the factorization.
 *    - threshold[0] : threshold for matrix B.
 *    - threshold[1] : threshold for matrix E and F.
 *    - threshold[2] : threshold for matrix S (Schur Complement).
 * The default is 1.0e-2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold );

/**
 * (Optional) Set the threshold for dropping in Newton–Schulz–Hotelling iteration (NHS-ILU).
 * Any entries less than this threshold are dropped when forming the approximate inverse matrix.
 * The default is 1.0e-2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold );

/**
 * (Optional) Set the array of thresholds for dropping in Newton–Schulz–Hotelling
 * iteration (for NHS-ILU).  Any fill-in less than thresholds is dropped when
 * forming the approximate inverse matrix.
 *
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton–Schulz–Hotelling iteration.
 *
 * The default is 1.0e-2.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold );

/**
 * (Optional) Set maximum number of iterations for Schur System Solve.
 * For GMRES-ILU, this is the maximum number of iterations for GMRES.
 * The Krylov dimension for GMRES is set equal to this value to avoid restart.
 * For NSH-ILU, this is the maximum number of iterations for NSH solve.
 * The default is 5.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetSchurMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ss_max_iter );

/**
 * Set the type of ILU factorization.
 *
 * Options for \e ilu_type are:
 *    - 0 : BJ with ILU(k) (default, with k = 0)
 *    - 1 : BJ with ILUT
 *    - 10 : GMRES with ILU(k)
 *    - 11 : GMRES with ILUT
 *    - 20 : NSH with ILU(k)
 *    - 21 : NSH with ILUT
 *    - 30 : RAS with ILU(k)
 *    - 31 : RAS with ILUT
 *    - 40 : (nonsymmetric permutation) DDPQ-GMRES with ILU(k)
 *    - 41 : (nonsymmetric permutation) DDPQ-GMRES with ILUT
 *    - 50 : GMRES with RAP-ILU(0) using MILU(0) for P
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_type );

/**
 * Set the type of reordering for the local matrix.
 *
 * Options for \e reordering_type are:
 *    - 0 : No reordering
 *    - 1 : RCM (default)
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetLocalReordering( NALU_HYPRE_Solver solver, NALU_HYPRE_Int reordering_type );

/**
 * (Optional) Set the print level to print setup and solve information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 *    - 2 : print solve information
 *    - 3 : print both setup and solve information
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );

/**
 * (Optional) Requests logging of solver diagnostics.
 * Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default is 0, do nothing.  The latest
 * residual will be available if logging > 1.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUSetLogging( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );

/**
 * (Optional) Return the number of ILU iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUGetNumIterations( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );

/**
 * (Optional) Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ILUGetFinalRelativeResidualNorm(  NALU_HYPRE_Solver solver, NALU_HYPRE_Real *res_norm );

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParCSRMatrix GenerateLaplacian(MPI_Comm    comm,
                                     NALU_HYPRE_BigInt   nx,
                                     NALU_HYPRE_BigInt   ny,
                                     NALU_HYPRE_BigInt   nz,
                                     NALU_HYPRE_Int   P,
                                     NALU_HYPRE_Int   Q,
                                     NALU_HYPRE_Int   R,
                                     NALU_HYPRE_Int   p,
                                     NALU_HYPRE_Int   q,
                                     NALU_HYPRE_Int   r,
                                     NALU_HYPRE_Real *value);

NALU_HYPRE_ParCSRMatrix GenerateLaplacian27pt(MPI_Comm    comm,
                                         NALU_HYPRE_BigInt   nx,
                                         NALU_HYPRE_BigInt   ny,
                                         NALU_HYPRE_BigInt   nz,
                                         NALU_HYPRE_Int   P,
                                         NALU_HYPRE_Int   Q,
                                         NALU_HYPRE_Int   R,
                                         NALU_HYPRE_Int   p,
                                         NALU_HYPRE_Int   q,
                                         NALU_HYPRE_Int   r,
                                         NALU_HYPRE_Real *value);

NALU_HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm    comm,
                                        NALU_HYPRE_BigInt   nx,
                                        NALU_HYPRE_BigInt   ny,
                                        NALU_HYPRE_Int   P,
                                        NALU_HYPRE_Int   Q,
                                        NALU_HYPRE_Int   p,
                                        NALU_HYPRE_Int   q,
                                        NALU_HYPRE_Real *value);

NALU_HYPRE_ParCSRMatrix GenerateDifConv(MPI_Comm    comm,
                                   NALU_HYPRE_BigInt   nx,
                                   NALU_HYPRE_BigInt   ny,
                                   NALU_HYPRE_BigInt   nz,
                                   NALU_HYPRE_Int   P,
                                   NALU_HYPRE_Int   Q,
                                   NALU_HYPRE_Int   R,
                                   NALU_HYPRE_Int   p,
                                   NALU_HYPRE_Int   q,
                                   NALU_HYPRE_Int   r,
                                   NALU_HYPRE_Real *value);

NALU_HYPRE_ParCSRMatrix
GenerateRotate7pt(MPI_Comm   comm,
                  NALU_HYPRE_BigInt  nx,
                  NALU_HYPRE_BigInt  ny,
                  NALU_HYPRE_Int  P,
                  NALU_HYPRE_Int  Q,
                  NALU_HYPRE_Int  p,
                  NALU_HYPRE_Int  q,
                  NALU_HYPRE_Real alpha,
                  NALU_HYPRE_Real eps );

NALU_HYPRE_ParCSRMatrix
GenerateVarDifConv(MPI_Comm         comm,
                   NALU_HYPRE_BigInt        nx,
                   NALU_HYPRE_BigInt        ny,
                   NALU_HYPRE_BigInt        nz,
                   NALU_HYPRE_Int        P,
                   NALU_HYPRE_Int        Q,
                   NALU_HYPRE_Int        R,
                   NALU_HYPRE_Int        p,
                   NALU_HYPRE_Int        q,
                   NALU_HYPRE_Int        r,
                   NALU_HYPRE_Real       eps,
                   NALU_HYPRE_ParVector *rhs_ptr);

NALU_HYPRE_ParCSRMatrix
GenerateRSVarDifConv(MPI_Comm         comm,
                     NALU_HYPRE_BigInt        nx,
                     NALU_HYPRE_BigInt        ny,
                     NALU_HYPRE_BigInt        nz,
                     NALU_HYPRE_Int        P,
                     NALU_HYPRE_Int        Q,
                     NALU_HYPRE_Int        R,
                     NALU_HYPRE_Int        p,
                     NALU_HYPRE_Int        q,
                     NALU_HYPRE_Int        r,
                     NALU_HYPRE_Real       eps,
                     NALU_HYPRE_ParVector *rhs_ptr,
                     NALU_HYPRE_Int        type);

float*
GenerateCoordinates(MPI_Comm  comm,
                    NALU_HYPRE_BigInt nx,
                    NALU_HYPRE_BigInt ny,
                    NALU_HYPRE_BigInt nz,
                    NALU_HYPRE_Int P,
                    NALU_HYPRE_Int Q,
                    NALU_HYPRE_Int R,
                    NALU_HYPRE_Int p,
                    NALU_HYPRE_Int q,
                    NALU_HYPRE_Int r,
                    NALU_HYPRE_Int coorddim);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * (Optional) Switches on use of Jacobi interpolation after computing
 * an original interpolation
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetPostInterpType(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Int    post_interp_type);

/*
 * (Optional) Sets a truncation threshold for Jacobi interpolation.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Real   jacobi_trunc_threshold);

/*
 * (Optional) Defines the number of relaxation steps for CR
 * The default is 2.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(NALU_HYPRE_Solver solver,
                                            NALU_HYPRE_Int    num_CR_relax_steps);

/*
 * (Optional) Defines convergence rate for CR
 * The default is 0.7.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRRate(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   CR_rate);

/*
 * (Optional) Defines strong threshold for CR
 * The default is 0.0.
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRStrongTh(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Real   CR_strong_th);

/*
 * (Optional) Defines whether to use CG
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetCRUseCG(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    CR_use_CG);

/*
 * (Optional) Defines the Type of independent set algorithm used for CR
 **/
NALU_HYPRE_Int NALU_HYPRE_BoomerAMGSetISType(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    IS_type);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref Eigensolvers.
 *
 * @{
 **/

/**
 * Load interface interpreter.  Vector part loaded with hypre_ParKrylov
 * functions and multivector part loaded with mv_TempMultiVector functions.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRSetupInterpreter(mv_InterfaceInterpreter *i);

/**
 * Load Matvec interpreter with hypre_ParKrylov functions.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRSetupMatvec(NALU_HYPRE_MatvecFunctions *mv);

/*
 * Print multivector to file.
 **/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRMultiVectorPrint(void *x_,
                             const char *fileName);

/*
 * Read multivector from file.
 **/
void *
NALU_HYPRE_ParCSRMultiVectorRead(MPI_Comm comm,
                            void *ii_,
                            const char *fileName);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
