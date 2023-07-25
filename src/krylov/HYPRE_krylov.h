/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_KRYLOV_HEADER
#define NALU_HYPRE_KRYLOV_HEADER

#include "NALU_HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup KrylovSolvers Krylov Solvers
 *
 * A basic interface for Krylov solvers. These solvers support many of the
 * matrix/vector storage schemes in hypre.  They should be used in conjunction
 * with the storage-specific interfaces, particularly the specific Create() and
 * Destroy() functions.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Krylov Solvers
 *
 * @{
 **/

#ifndef NALU_HYPRE_SOLVER_STRUCT
#define NALU_HYPRE_SOLVER_STRUCT
struct nalu_hypre_Solver_struct;
/**
 * The solver object.
 **/
typedef struct nalu_hypre_Solver_struct *NALU_HYPRE_Solver;
#endif

#ifndef NALU_HYPRE_MATRIX_STRUCT
#define NALU_HYPRE_MATRIX_STRUCT
struct nalu_hypre_Matrix_struct;
/**
 * The matrix object.
 **/
typedef struct nalu_hypre_Matrix_struct *NALU_HYPRE_Matrix;
#endif

#ifndef NALU_HYPRE_VECTOR_STRUCT
#define NALU_HYPRE_VECTOR_STRUCT
struct nalu_hypre_Vector_struct;
/**
 * The vector object.
 **/
typedef struct nalu_hypre_Vector_struct *NALU_HYPRE_Vector;
#endif

typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToSolverFcn)(NALU_HYPRE_Solver,
                                          NALU_HYPRE_Matrix,
                                          NALU_HYPRE_Vector,
                                          NALU_HYPRE_Vector);

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
 * @name PCG Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetup(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Matrix A,
                         NALU_HYPRE_Vector b,
                         NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSolve(NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Matrix A,
                         NALU_HYPRE_Vector b,
                         NALU_HYPRE_Vector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetTol(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is
 * 0). If one desires the convergence test to check the absolute
 * convergence tolerance \e only, then set the relative convergence
 * tolerance to 0.0.  (The default convergence test is \f$ <C*r,r> \leq\f$
 * max(relative\f$\_\f$tolerance\f$^{2} \ast <C*b, b>\f$, absolute\f$\_\f$tolerance\f$^2\f$).)
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   a_tol);

/**
 * (Optional) Set a residual-based convergence tolerance which checks if
 * \f$\|r_{old}-r_{new}\| < rtol \|b\|\f$. This is useful when trying to converge to
 * very low relative and/or absolute tolerances, in order to bail-out before
 * roundoff errors affect the approximation.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetResidualTol(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   rtol);
/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetAbsoluteTolFactor(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real abstolf);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetStopCrit(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int stop_crit);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetMaxIter(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetTwoNorm(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    two_norm);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetRelChange(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    rel_change);

/**
 * (Optional) Recompute the residual at the end to double-check convergence.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetRecomputeResidual(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int    recompute_residual);

/**
 * (Optional) Periodically recompute the residual while iterating.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetRecomputeResidualP(NALU_HYPRE_Solver solver,
                                         NALU_HYPRE_Int    recompute_residual_p);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetPrecond(NALU_HYPRE_Solver         solver,
                              NALU_HYPRE_PtrToSolverFcn precond,
                              NALU_HYPRE_PtrToSolverFcn precond_setup,
                              NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetLogging(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGSetPrintLevel(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetNumIterations(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetResidual(NALU_HYPRE_Solver  solver,
                               void         *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetTol(NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Real   *tol);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetResidualTol(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Real   *rtol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetAbsoluteTolFactor(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real  *abstolf);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                           NALU_HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetStopCrit(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int   *stop_crit);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetMaxIter(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int    *max_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetTwoNorm(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int    *two_norm);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetRelChange(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *rel_change);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetSkipRealResidualCheck(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int   *skip_real_r_check);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetPrecond(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Solver *precond_data_ptr);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetLogging(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetPrintLevel(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_PCGGetConverged(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int          *converged);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetup(NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Matrix A,
                           NALU_HYPRE_Vector b,
                           NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSolve(NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Matrix A,
                           NALU_HYPRE_Vector b,
                           NALU_HYPRE_Vector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetTol(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetStopCrit(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetMinIter(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetKDim(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    k_dim);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetRelChange(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    rel_change);

/**
 * (Optional) By default, hypre checks for convergence by evaluating the actual
 * residual before returnig from GMRES (with restart if the true residual does
 * not indicate convergence). This option allows users to skip the evaluation
 * and the check of the actual residual for badly conditioned problems where
 * restart is not expected to be beneficial.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetSkipRealResidualCheck(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Int    skip_real_r_check);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetPrecond(NALU_HYPRE_Solver         solver,
                                NALU_HYPRE_PtrToSolverFcn precond,
                                NALU_HYPRE_PtrToSolverFcn precond_setup,
                                NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetLogging(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                  NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetResidual(NALU_HYPRE_Solver   solver,
                                 void          *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetTol(NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Real   *tol);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetAbsoluteTol(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                             NALU_HYPRE_Real  *cf_tol);

/*
 * OBSOLETE
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetStopCrit(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetMinIter(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int   *min_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetMaxIter(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *max_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetKDim(NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int    *k_dim);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetRelChange(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *rel_change);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Solver *precond_data_ptr);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetLogging(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetPrintLevel(NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_GMRESGetConverged(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *converged);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FlexGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetup(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Matrix A,
                               NALU_HYPRE_Vector b,
                               NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSolve(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Matrix A,
                               NALU_HYPRE_Vector b,
                               NALU_HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetTol(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetConvergenceFactorTol(NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetMinIter(NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetKDim(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    k_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetPrecond(NALU_HYPRE_Solver         solver,
                                    NALU_HYPRE_PtrToSolverFcn precond,
                                    NALU_HYPRE_PtrToSolverFcn precond_setup,
                                    NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetLogging(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                      NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetResidual(NALU_HYPRE_Solver   solver,
                                     void          *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetTol(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                                 NALU_HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetStopCrit(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetMinIter(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int   *min_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetMaxIter(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *max_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetKDim(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *k_dim);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Solver *precond_data_ptr);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetLogging(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetPrintLevel(NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetConverged(NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int    *converged);

/**
 * (Optional) Set a user-defined function to modify solve-time preconditioner
 * attributes.
 **/
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetModifyPC(NALU_HYPRE_Solver           solver,
                                     NALU_HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name LGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetup(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Matrix A,
                            NALU_HYPRE_Vector b,
                            NALU_HYPRE_Vector x);

/**
 * Solve the system. Details on LGMRES may be found in A. H. Baker,
 * E.R. Jessup, and T.A. Manteuffel, "A technique for accelerating the
 * convergence of restarted GMRES." SIAM Journal on Matrix Analysis and
 * Applications, 26 (2005), pp. 962-984. LGMRES(m,k) in the paper
 * corresponds to LGMRES(Kdim+AugDim, AugDim).
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSolve(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Matrix A,
                            NALU_HYPRE_Vector b,
                            NALU_HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetTol(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real   tol);
/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the approximation space
 * (includes the augmentation vectors).
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetKDim(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    k_dim);

/**
 * (Optional) Set the number of augmentation vectors  (default: 2).
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetAugDim(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    aug_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetPrecond(NALU_HYPRE_Solver         solver,
                       NALU_HYPRE_PtrToSolverFcn precond,
                       NALU_HYPRE_PtrToSolverFcn precond_setup,
                       NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetLogging(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                   NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetResidual(NALU_HYPRE_Solver   solver,
                                  void          *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetTol(NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                              NALU_HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetStopCrit(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetMinIter(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int   *min_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetMaxIter(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *max_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetKDim(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int    *k_dim);
/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetAugDim(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *k_dim);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Solver *precond_data_ptr);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetLogging(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetPrintLevel(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetConverged(NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int    *converged);

/**** added by KS ****** */
/**
 * @name COGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetup(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Matrix A,
                             NALU_HYPRE_Vector b,
                             NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSolve(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Matrix A,
                             NALU_HYPRE_Vector b,
                             NALU_HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetTol(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                               NALU_HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetMinIter(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetMaxIter(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetKDim(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    k_dim);

/**
 * (Optional) Set number of unrolling in mass funcyions in COGMRES
 * Can be 4 or 8. Default: no unrolling.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetUnroll(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    unroll);

/**
 * (Optional) Set the number of orthogonalizations in COGMRES (at most 2).
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetCGS(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    cgs);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetPrecond(NALU_HYPRE_Solver         solver,
                                  NALU_HYPRE_PtrToSolverFcn precond,
                                  NALU_HYPRE_PtrToSolverFcn precond_setup,
                                  NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetLogging(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetPrintLevel(NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetNumIterations(NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                    NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetResidual(NALU_HYPRE_Solver   solver,
                                   void          *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetTol(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                               NALU_HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
//NALU_HYPRE_Int NALU_HYPRE_COGMRESGetStopCrit(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *stop_crit);
//NALU_HYPRE_Int NALU_HYPRE_COGMRESSetStopCrit(NALU_HYPRE_Solver solver, NALU_HYPRE_Int *stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetMinIter(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int   *min_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetMaxIter(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *max_iter);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetKDim(NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int    *k_dim);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetUnroll(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *unroll);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetCGS(NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int    *cgs);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetPrecond(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Solver *precond_data_ptr);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetLogging(NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetPrintLevel(NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Int    *level);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetConverged(NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *converged);

/**
 * (Optional) Set a user-defined function to modify solve-time preconditioner
 * attributes.
 **/
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetModifyPC(NALU_HYPRE_Solver           solver,
                                   NALU_HYPRE_PtrToModifyPCFcn modify_pc);

/****** KS code ends here **************************************************/

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name BiCGSTAB Solver
 *
 * @{
 **/

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABDestroy(NALU_HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetup(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Matrix A,
                              NALU_HYPRE_Vector b,
                              NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSolve(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Matrix A,
                              NALU_HYPRE_Vector b,
                              NALU_HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetTol(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance \f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetAbsoluteTol(NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetConvergenceFactorTol(NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real   cf_tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetStopCrit(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetMinIter(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetMaxIter(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetPrecond(NALU_HYPRE_Solver         solver,
                                   NALU_HYPRE_PtrToSolverFcn precond,
                                   NALU_HYPRE_PtrToSolverFcn precond_setup,
                                   NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetLogging(NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetPrintLevel(NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetNumIterations(NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                     NALU_HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetResidual(NALU_HYPRE_Solver   solver,
                                    void          *residual);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetPrecond(NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Solver *precond_data_ptr);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name CGNR Solver
 *
 * @{
 **/

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRDestroy(NALU_HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetup(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Matrix A,
                          NALU_HYPRE_Vector b,
                          NALU_HYPRE_Vector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSolve(NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Matrix A,
                          NALU_HYPRE_Vector b,
                          NALU_HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetTol(NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Real   tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetStopCrit(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    stop_crit);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetMinIter(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetMaxIter(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    max_iter);

/**
 * (Optional) Set the preconditioner to use.
 * Note that the only preconditioner available in hypre for use with
 * CGNR is currently BoomerAMG. It requires to use Jacobi as
 * a smoother without CF smoothing, i.e. relax_type needs to be set to 0
 * or 7 and relax_order needs to be set to 0 by the user, since these
 * are not default values. It can be used with a relaxation weight for
 * Jacobi, which can significantly improve convergence.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetPrecond(NALU_HYPRE_Solver         solver,
                               NALU_HYPRE_PtrToSolverFcn precond,
                               NALU_HYPRE_PtrToSolverFcn precondT,
                               NALU_HYPRE_PtrToSolverFcn precond_setup,
                               NALU_HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetLogging(NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    logging);

#if 0 /* need to add */
/*
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRSetPrintLevel(NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    level);
#endif

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRGetNumIterations(NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRGetFinalRelativeResidualNorm(NALU_HYPRE_Solver  solver,
                                                 NALU_HYPRE_Real   *norm);

#if 0 /* need to add */
/*
 * Return the residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRGetResidual(NALU_HYPRE_Solver   solver,
                                void         **residual);
#endif

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_CGNRGetPrecond(NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Solver *precond_data_ptr);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
