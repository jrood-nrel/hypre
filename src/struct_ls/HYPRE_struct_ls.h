/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_STRUCT_LS_HEADER
#define NALU_HYPRE_STRUCT_LS_HEADER

#include "NALU_HYPRE_utilities.h"
#include "NALU_HYPRE_struct_mv.h"
#include "NALU_HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup StructSolvers Struct Solvers
 *
 * Linear solvers for structured grids. These solvers use matrix/vector storage
 * schemes that are tailored to structured grid problems.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Solvers
 *
 * @{
 **/

struct nalu_hypre_StructSolver_struct;
/**
 * The solver object.
 **/
typedef struct nalu_hypre_StructSolver_struct *NALU_HYPRE_StructSolver;

typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToStructSolverFcn)(NALU_HYPRE_StructSolver,
                                                NALU_HYPRE_StructMatrix,
                                                NALU_HYPRE_StructVector,
                                                NALU_HYPRE_StructVector);

#ifndef NALU_HYPRE_MODIFYPC
#define NALU_HYPRE_MODIFYPC
/* if pc not defined, then may need NALU_HYPRE_SOLVER also */

#ifndef NALU_HYPRE_SOLVER_STRUCT
#define NALU_HYPRE_SOLVER_STRUCT
struct nalu_hypre_Solver_struct;
typedef struct nalu_hypre_Solver_struct *NALU_HYPRE_Solver;
#endif

typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToModifyPCFcn)(NALU_HYPRE_Solver,
                                            NALU_HYPRE_Int,
                                            NALU_HYPRE_Real);
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Jacobi Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiCreate(MPI_Comm            comm,
                                   NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiDestroy(NALU_HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSetup(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSolve(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSetTol(NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Real         tol);


NALU_HYPRE_Int NALU_HYPRE_StructJacobiGetTol(NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Real *tol );

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSetMaxIter(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructJacobiGetMaxIter(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int *max_iter );

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSetZeroGuess(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructJacobiGetZeroGuess(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int *zeroguess );

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiSetNonZeroGuess(NALU_HYPRE_StructSolver solver);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                             NALU_HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                         NALU_HYPRE_Real         *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PFMG Solver
 *
 * PFMG is a semicoarsening multigrid solver that uses pointwise relaxation.
 * For periodic problems, users should try to set the grid size in periodic
 * dimensions to be as close to a power-of-two as possible.  That is, if the
 * grid size in a periodic dimension is given by \f$N = 2^m * M\f$ where \f$M\f$
 * is not a power-of-two, then \f$M\f$ should be as small as possible.  Large
 * values of \f$M\f$ will generally result in slower convergence rates.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGCreate(MPI_Comm            comm,
                                 NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGDestroy(NALU_HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetup(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_StructMatrix A,
                                NALU_HYPRE_StructVector b,
                                NALU_HYPRE_StructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSolve(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_StructMatrix A,
                                NALU_HYPRE_StructVector b,
                                NALU_HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetTol(NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetTol (NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_Real *tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetMaxIter(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetMaxIter(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int *max_iter);

/**
 * (Optional) Set maximum number of multigrid grid levels.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetMaxLevels(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          max_levels);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetMaxLevels (NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int *max_levels );

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetRelChange(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          rel_change);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetRelChange (NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int *rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetZeroGuess(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetZeroGuess(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int *zeroguess);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetNonZeroGuess(NALU_HYPRE_StructSolver solver);

/**
 * (Optional) Set relaxation type.
 *
 * Current relaxation methods set by \e relax_type are:
 *
 *    - 0 : Jacobi
 *    - 1 : Weighted Jacobi (default)
 *    - 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
 *    - 3 : Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetRelaxType(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          relax_type);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetRelaxType(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int *relax_type);

/*
 * (Optional) Set Jacobi weight (this is purposely not documented)
 */
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Real         weight);
NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Real        *weight);


/**
 * (Optional) Set type of coarse-grid operator to use.
 *
 * Current operators set by \e rap_type are:
 *
 *    - 0 : Galerkin (default)
 *    - 1 : non-Galerkin 5-pt or 7-pt stencils
 *
 * Both operators are constructed algebraically.  The non-Galerkin option
 * maintains a 5-pt stencil in 2D and a 7-pt stencil in 3D on all grid levels.
 * The stencil coefficients are computed by averaging techniques.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetRAPType(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int          rap_type);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetRAPType(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int *rap_type );

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetNumPreRelax(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          num_pre_relax);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetNumPreRelax(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int *num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetNumPostRelax(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          num_post_relax);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetNumPostRelax(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int *num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetSkipRelax(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          skip_relax);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetSkipRelax(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int *skip_relax);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetDxyz(NALU_HYPRE_StructSolver  solver,
                                  NALU_HYPRE_Real         *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetLogging(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetLogging(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int *logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int          print_level);

NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetPrintLevel(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int *print_level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                           NALU_HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                       NALU_HYPRE_Real         *norm);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetDeviceLevel( NALU_HYPRE_StructSolver  solver,
                                NALU_HYPRE_Int   device_level  );
#endif
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct SMG Solver
 *
 * SMG is a semicoarsening multigrid solver that uses plane smoothing (in 3D).
 * The plane smoother calls a 2D SMG algorithm with line smoothing, and the line
 * smoother is cyclic reduction (1D SMG).  For periodic problems, the grid size
 * in periodic dimensions currently must be a power-of-two.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGCreate(MPI_Comm            comm,
                                NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGDestroy(NALU_HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetup(NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_StructMatrix A,
                               NALU_HYPRE_StructVector b,
                               NALU_HYPRE_StructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSolve(NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_StructMatrix A,
                               NALU_HYPRE_StructVector b,
                               NALU_HYPRE_StructVector x);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetMemoryUse(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          memory_use);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetMemoryUse(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int *memory_use);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetTol(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetTol(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Real *tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetMaxIter(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetMaxIter(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int *max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetRelChange(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          rel_change);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetRelChange(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int *rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetZeroGuess(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetZeroGuess(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int *zeroguess);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetNonZeroGuess(NALU_HYPRE_StructSolver solver);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetNumPreRelax(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int          num_pre_relax);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetNumPreRelax(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int *num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetNumPostRelax(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          num_post_relax);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetNumPostRelax(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int *num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetLogging(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetLogging(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int *logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          print_level);

NALU_HYPRE_Int NALU_HYPRE_StructSMGGetPrintLevel(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int *print_level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                          NALU_HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                      NALU_HYPRE_Real         *norm);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetDeviceLevel( NALU_HYPRE_StructSolver  solver,
                               NALU_HYPRE_Int   device_level  );
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct CycRed Solver
 *
 * CycRed is a cyclic reduction solver that simultaneously solves a collection
 * of 1D tridiagonal systems embedded in a d-dimensional grid.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedCreate(MPI_Comm            comm,
                                   NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedDestroy(NALU_HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedSetup(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedSolve(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 *
 * (Optional) Set the dimension number for the embedded 1D tridiagonal systems.
 * The default is \e tdim = 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedSetTDim(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          tdim);

/**
 * (Optional) Set the base index and stride for the embedded 1D systems.  The
 * stride must be equal one in the dimension corresponding to the 1D systems
 * (see \ref NALU_HYPRE_StructCycRedSetTDim).
 **/
NALU_HYPRE_Int NALU_HYPRE_StructCycRedSetBase(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          ndim,
                                    NALU_HYPRE_Int         *base_index,
                                    NALU_HYPRE_Int         *base_stride);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPCGCreate(MPI_Comm            comm,
                                NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructPCGDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetup(NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_StructMatrix A,
                               NALU_HYPRE_StructVector b,
                               NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSolve(NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_StructMatrix A,
                               NALU_HYPRE_StructVector b,
                               NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetTol(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetAbsoluteTol(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetMaxIter(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetTwoNorm(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          two_norm);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetRelChange(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          rel_change);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetPrecond(NALU_HYPRE_StructSolver         solver,
                                    NALU_HYPRE_PtrToStructSolverFcn precond,
                                    NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                    NALU_HYPRE_StructSolver         precond_solver);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetLogging(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructPCGSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          level);

NALU_HYPRE_Int NALU_HYPRE_StructPCGGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                          NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                      NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructPCGGetResidual(NALU_HYPRE_StructSolver   solver,
                                     void               **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructDiagScaleSetup(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector y,
                                     NALU_HYPRE_StructVector x);

/**
 * Solve routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructDiagScale(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_StructMatrix HA,
                                NALU_HYPRE_StructVector Hy,
                                NALU_HYPRE_StructVector Hx);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGMRESCreate(MPI_Comm            comm,
                                  NALU_HYPRE_StructSolver *solver);


/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructGMRESDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetup(NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_StructMatrix A,
                                 NALU_HYPRE_StructVector b,
                                 NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSolve(NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_StructMatrix A,
                                 NALU_HYPRE_StructVector b,
                                 NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetTol(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetAbsoluteTol(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetMaxIter(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetKDim(NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Int          k_dim);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetPrecond(NALU_HYPRE_StructSolver         solver,
                                      NALU_HYPRE_PtrToStructSolverFcn precond,
                                      NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                      NALU_HYPRE_StructSolver         precond_solver);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetLogging(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          level);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                            NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                        NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructGMRESGetResidual(NALU_HYPRE_StructSolver   solver,
                                       void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESCreate(MPI_Comm            comm,
                                      NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetup(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector b,
                                     NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSolve(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector b,
                                     NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetTol(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetAbsoluteTol(NALU_HYPRE_StructSolver solver,
                                              NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetMaxIter(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetKDim(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          k_dim);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetPrecond(NALU_HYPRE_StructSolver         solver,
                                          NALU_HYPRE_PtrToStructSolverFcn precond,
                                          NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                          NALU_HYPRE_StructSolver         precond_solver);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetLogging(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                             NALU_HYPRE_Int          level);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                                NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                            NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESGetResidual(NALU_HYPRE_StructSolver   solver,
                                           void               **residual);

NALU_HYPRE_Int NALU_HYPRE_StructFlexGMRESSetModifyPC(NALU_HYPRE_StructSolver     solver,
                                           NALU_HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructLGMRESCreate(MPI_Comm            comm,
                                   NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructLGMRESDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetup(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSolve(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetTol(NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetAbsoluteTol(NALU_HYPRE_StructSolver solver,
                                           NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetMaxIter(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetKDim(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          k_dim);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetAugDim(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          aug_dim);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetPrecond(NALU_HYPRE_StructSolver         solver,
                                       NALU_HYPRE_PtrToStructSolverFcn precond,
                                       NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                       NALU_HYPRE_StructSolver         precond_solver);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetLogging(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          level);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                             NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                         NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructLGMRESGetResidual(NALU_HYPRE_StructSolver   solver,
                                        void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABCreate(MPI_Comm            comm,
                                     NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetup(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_StructMatrix A,
                                    NALU_HYPRE_StructVector b,
                                    NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSolve(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_StructMatrix A,
                                    NALU_HYPRE_StructVector b,
                                    NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetTol(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetAbsoluteTol(NALU_HYPRE_StructSolver solver,
                                             NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetMaxIter(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetPrecond(NALU_HYPRE_StructSolver         solver,
                                         NALU_HYPRE_PtrToStructSolverFcn precond,
                                         NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                         NALU_HYPRE_StructSolver         precond_solver);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetLogging(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                            NALU_HYPRE_Int          level);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                               NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                           NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructBiCGSTABGetResidual( NALU_HYPRE_StructSolver   solver,
                                           void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Hybrid Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridCreate(MPI_Comm            comm,
                                   NALU_HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridDestroy(NALU_HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetup(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSolve(NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_StructMatrix A,
                                  NALU_HYPRE_StructVector b,
                                  NALU_HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetTol(NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Real         tol);

/**
 * (Optional) Set an accepted convergence tolerance for diagonal scaling (DS).
 * The solver will switch preconditioners if the convergence of DS is slower
 * than \e cf_tol.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetConvergenceTol(NALU_HYPRE_StructSolver solver,
                                              NALU_HYPRE_Real         cf_tol);

/**
 * (Optional) Set maximum number of iterations for diagonal scaling (DS).  The
 * solver will switch preconditioners if DS reaches \e ds_max_its.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetDSCGMaxIter(NALU_HYPRE_StructSolver solver,
                                           NALU_HYPRE_Int          ds_max_its);

/**
 * (Optional) Set maximum number of iterations for general preconditioner (PRE).
 * The solver will stop if PRE reaches \e pre_max_its.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetPCGMaxIter(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          pre_max_its);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetTwoNorm(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          two_norm);

NALU_HYPRE_Int NALU_HYPRE_StructHybridSetStopCrit(NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int          stop_crit);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetRelChange(NALU_HYPRE_StructSolver solver,
                                         NALU_HYPRE_Int          rel_change);

/**
 * (Optional) Set the type of Krylov solver to use.
 *
 * Current krylov methods set by \e solver_type are:
 *
 *    - 0 : PCG (default)
 *    - 1 : GMRES
 *    - 2 : BiCGSTAB
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetSolverType(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          solver_type);

/**
 * (Optional) Set recompute residual (don't rely on 3-term recurrence).
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetRecomputeResidual( NALU_HYPRE_StructSolver  solver,
                                        NALU_HYPRE_Int           recompute_residual );

/**
 * (Optional) Get recompute residual option.
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetRecomputeResidual( NALU_HYPRE_StructSolver  solver,
                                        NALU_HYPRE_Int          *recompute_residual );

/**
 * (Optional) Set recompute residual period (don't rely on 3-term recurrence).
 *
 * Recomputes residual after every specified number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetRecomputeResidualP( NALU_HYPRE_StructSolver  solver,
                                         NALU_HYPRE_Int           recompute_residual_p );

/**
 * (Optional) Get recompute residual period option.
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetRecomputeResidualP( NALU_HYPRE_StructSolver  solver,
                                         NALU_HYPRE_Int          *recompute_residual_p );

/**
 * (Optional) Set the maximum size of the Krylov space when using GMRES.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetKDim(NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int          k_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetPrecond(NALU_HYPRE_StructSolver         solver,
                                       NALU_HYPRE_PtrToStructSolverFcn precond,
                                       NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                       NALU_HYPRE_StructSolver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetLogging(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          print_level);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                             NALU_HYPRE_Int          *num_its);

/**
 * Return the number of diagonal scaling iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridGetDSCGNumIterations(NALU_HYPRE_StructSolver  solver,
                                                 NALU_HYPRE_Int          *ds_num_its);

/**
 * Return the number of general preconditioning iterations taken.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridGetPCGNumIterations(NALU_HYPRE_StructSolver  solver,
                                                NALU_HYPRE_Int          *pre_num_its);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                         NALU_HYPRE_Real         *norm);

NALU_HYPRE_Int NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor(NALU_HYPRE_StructSolver solver,
                                                    NALU_HYPRE_Real pcg_atolf );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Struct SparseMSG Solver
 **/

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGCreate(MPI_Comm            comm,
                                      NALU_HYPRE_StructSolver *solver);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGDestroy(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetup(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector b,
                                     NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSolve(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_StructMatrix A,
                                     NALU_HYPRE_StructVector b,
                                     NALU_HYPRE_StructVector x);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetTol(NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Real         tol);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetMaxIter(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          max_iter);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetJump(NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int          jump);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetRelChange(NALU_HYPRE_StructSolver solver,
                                            NALU_HYPRE_Int          rel_change);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetZeroGuess(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetNonZeroGuess(NALU_HYPRE_StructSolver solver);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetRelaxType(NALU_HYPRE_StructSolver solver,
                                            NALU_HYPRE_Int          relax_type);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                               NALU_HYPRE_Real         weight);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetNumPreRelax(NALU_HYPRE_StructSolver solver,
                                              NALU_HYPRE_Int          num_pre_relax);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetNumPostRelax(NALU_HYPRE_StructSolver solver,
                                               NALU_HYPRE_Int          num_post_relax);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetNumFineRelax(NALU_HYPRE_StructSolver solver,
                                               NALU_HYPRE_Int          num_fine_relax);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetLogging(NALU_HYPRE_StructSolver solver,
                                          NALU_HYPRE_Int          logging);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGSetPrintLevel(NALU_HYPRE_StructSolver solver,
                                             NALU_HYPRE_Int   print_level);


NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGGetNumIterations(NALU_HYPRE_StructSolver  solver,
                                                NALU_HYPRE_Int          *num_iterations);

NALU_HYPRE_Int NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(NALU_HYPRE_StructSolver  solver,
                                                            NALU_HYPRE_Real         *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref Eigensolvers.
 *
 * @{
 **/

/**
 * Load interface interpreter. Vector part loaded with nalu_hypre_StructKrylov
 * functions and multivector part loaded with mv_TempMultiVector functions.
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructSetupInterpreter(mv_InterfaceInterpreter *i);

/**
 * Load Matvec interpreter with nalu_hypre_StructKrylov functions.
 **/
NALU_HYPRE_Int
NALU_HYPRE_StructSetupMatvec(NALU_HYPRE_MatvecFunctions *mv);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

