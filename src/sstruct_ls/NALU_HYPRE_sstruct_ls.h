/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_SSTRUCT_LS_HEADER
#define NALU_HYPRE_SSTRUCT_LS_HEADER

#include "NALU_HYPRE_config.h"
#include "NALU_HYPRE_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_sstruct_mv.h"
#include "NALU_HYPRE_struct_ls.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup SStructSolvers SStruct Solvers
 *
 * These solvers use matrix/vector storage schemes that are taylored
 * to semi-structured grid problems.
 *
 * @memo Linear solvers for semi-structured grids
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Solvers
 *
 * @{
 **/

struct nalu_hypre_SStructSolver_struct;
/**
 * The solver object.
 **/
typedef struct nalu_hypre_SStructSolver_struct *NALU_HYPRE_SStructSolver;

typedef NALU_HYPRE_Int (*NALU_HYPRE_PtrToSStructSolverFcn)(NALU_HYPRE_SStructSolver,
                                                 NALU_HYPRE_SStructMatrix,
                                                 NALU_HYPRE_SStructVector,
                                                 NALU_HYPRE_SStructVector);

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
 * @name SStruct SysPFMG Solver
 *
 * SysPFMG is a semicoarsening multigrid solver similar to PFMG, but for systems
 * of PDEs.  For periodic problems, users should try to set the grid size in
 * periodic dimensions to be as close to a power-of-two as possible (for more
 * details, see \ref Struct PFMG Solver).
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGCreate(MPI_Comm             comm,
                           NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGDestroy(NALU_HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetup(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSolve(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetTol(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Real          tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetMaxIter(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelChange(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNonZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.
 *
 * Current relaxation methods set by \e relax\_type are:
 *
 *    - 0 : Jacobi
 *    - 1 : Weighted Jacobi (default)
 *    - 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelaxType(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           relax_type);

/**
 * (Optional) Set Jacobi Weight.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetJacobiWeight(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Real          weight);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPreRelax(NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPostRelax(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetSkipRelax(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           skip_relax);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetDxyz(NALU_HYPRE_SStructSolver  solver,
                            NALU_HYPRE_Real          *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetLogging(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           print_level);


/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                     NALU_HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver solver,
                                                 NALU_HYPRE_Real         *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Split Solver
 *
 * @{
 **/

#define NALU_HYPRE_PFMG   10
#define NALU_HYPRE_SMG    11
#define NALU_HYPRE_Jacobi 17

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitCreate(MPI_Comm             comm,
                         NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitDestroy(NALU_HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetup(NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_SStructMatrix A,
                        NALU_HYPRE_SStructVector b,
                        NALU_HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSolve(NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_SStructMatrix A,
                        NALU_HYPRE_SStructVector b,
                        NALU_HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetTol(NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_Real          tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetMaxIter(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           max_iter);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetNonZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Set up the type of diagonal struct solver.  Either \e ssolver is
 * set to \e HYPRE\_SMG or \e HYPRE\_PFMG.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitSetStructSolver(NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           ssolver );

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                   NALU_HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                               NALU_HYPRE_Real          *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FAC Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACCreate(MPI_Comm             comm,
                       NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACDestroy2(NALU_HYPRE_SStructSolver solver);

/**
 * Re-distribute the composite matrix so that the amr hierachy is approximately
 * nested. Coarse underlying operators are also formed.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACAMR_RAP(NALU_HYPRE_SStructMatrix  A,
                        NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM],
                        NALU_HYPRE_SStructMatrix *fac_A);

/**
 * Set up the FAC solver structure .
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetup2(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector b,
                       NALU_HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSolve3(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector b,
                       NALU_HYPRE_SStructVector x);

/**
 * Set up amr structure
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPLevels(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           nparts,
                           NALU_HYPRE_Int          *plevels);
/**
 * Set up amr refinement factors
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetPRefinements(NALU_HYPRE_SStructSolver  solver,
                                NALU_HYPRE_Int            nparts,
                                NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] );

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the coarse level stencils reaching into a fine level grid.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroCFSten(NALU_HYPRE_SStructMatrix  A,
                           NALU_HYPRE_SStructGrid    grid,
                           NALU_HYPRE_Int            part,
                           NALU_HYPRE_Int            rfactors[NALU_HYPRE_MAXDIM]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the fine level stencils reaching into a coarse level grid.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroFCSten(NALU_HYPRE_SStructMatrix  A,
                           NALU_HYPRE_SStructGrid    grid,
                           NALU_HYPRE_Int            part);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places the identity in the coarse grid matrix underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroAMRMatrixData(NALU_HYPRE_SStructMatrix  A,
                                  NALU_HYPRE_Int            part_crse,
                                  NALU_HYPRE_Int            rfactors[NALU_HYPRE_MAXDIM]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places zeros in the coarse grid vector underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACZeroAMRVectorData(NALU_HYPRE_SStructVector  b,
                                  NALU_HYPRE_Int           *plevels,
                                  NALU_HYPRE_Int          (*rfactors)[NALU_HYPRE_MAXDIM] );

/**
 * (Optional) Set maximum number of FAC levels.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetMaxLevels( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           max_levels );
/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetTol(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_Real          tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetMaxIter(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelChange(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNonZeroGuess(NALU_HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.  See \ref NALU_HYPRE_SStructSysPFMGSetRelaxType
 * for appropriate values of \e relax\_type.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetRelaxType(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           relax_type);
/**
 * (Optional) Set Jacobi weight if weighted Jacobi is used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetJacobiWeight(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Real          weight);
/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPreRelax(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetNumPostRelax(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           num_post_relax);
/**
 * (Optional) Set coarsest solver type.
 *
 * Current solver types set by \e csolver\_type are:
 *
 *    - 1 : SysPFMG-PCG (default)
 *    - 2 : SysPFMG
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetCoarseSolverType(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           csolver_type);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACSetLogging(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           logging);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                 NALU_HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver solver,
                                             NALU_HYPRE_Real         *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**
 * @name SStruct Maxwell Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellCreate( MPI_Comm             comm,
                            NALU_HYPRE_SStructSolver *solver );
/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellDestroy( NALU_HYPRE_SStructSolver solver );

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetup(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * throughout the multigrid hierarchy.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSolve(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * only on the finest level, i.e., the node and edge multigrid
 * cycles are coupled only on the finest level.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSolve2(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x);

/**
 * Sets the gradient operator in the Maxwell solver.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetGrad(NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_ParCSRMatrix  T);

/**
 * Sets the coarsening factor.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetRfactors(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           rfactors[NALU_HYPRE_MAXDIM]);

/**
 * Finds the physical boundary row ranks on all levels.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellPhysBdy(NALU_HYPRE_SStructGrid  *grid_l,
                            NALU_HYPRE_Int           num_levels,
                            NALU_HYPRE_Int           rfactors[NALU_HYPRE_MAXDIM],
                            NALU_HYPRE_Int        ***BdryRanks_ptr,
                            NALU_HYPRE_Int         **BdryRanksCnt_ptr );

/**
 * Eliminates the rows and cols corresponding to the physical boundary in
 * a parcsr matrix.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellEliminateRowsCols(NALU_HYPRE_ParCSRMatrix  parA,
                                      NALU_HYPRE_Int           nrows,
                                      NALU_HYPRE_Int          *rows );

/**
 * Zeros the rows corresponding to the physical boundary in
 * a par vector.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellZeroVector(NALU_HYPRE_ParVector  b,
                               NALU_HYPRE_Int       *rows,
                               NALU_HYPRE_Int        nrows );

/**
 * (Optional) Set the constant coefficient flag- Nedelec interpolation
 * used.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetSetConstantCoef(NALU_HYPRE_SStructSolver solver,
                                       NALU_HYPRE_Int           flag);

/**
 * (Optional) Creates a gradient matrix from the grid. This presupposes
 * a particular orientation of the edge elements.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellGrad(NALU_HYPRE_SStructGrid    grid,
                         NALU_HYPRE_ParCSRMatrix  *T);

/**
 * (Optional) Set the convergence tolerance.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetTol(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Real          tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetMaxIter(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetRelChange(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           rel_change);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetNumPreRelax(NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetNumPostRelax(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellSetLogging(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           logging);

/**
 * Return the number of iterations taken.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                     NALU_HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver solver,
                                                 NALU_HYPRE_Real         *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructPCGCreate(MPI_Comm             comm,
                       NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructPCGDestroy(NALU_HYPRE_SStructSolver solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetup(NALU_HYPRE_SStructSolver solver,
                      NALU_HYPRE_SStructMatrix A,
                      NALU_HYPRE_SStructVector b,
                      NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSolve(NALU_HYPRE_SStructSolver solver,
                      NALU_HYPRE_SStructMatrix A,
                      NALU_HYPRE_SStructVector b,
                      NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetTol(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_Real          tol);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetAbsoluteTol(NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Real          tol);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetMaxIter(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           max_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetTwoNorm(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           two_norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetRelChange(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           rel_change);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetPrecond(NALU_HYPRE_SStructSolver          solver,
                           NALU_HYPRE_PtrToSStructSolverFcn  precond,
                           NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                           void                        *precond_solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetLogging(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           logging);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           level);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                 NALU_HYPRE_Int           *num_iterations);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                             NALU_HYPRE_Real          *norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetResidual(NALU_HYPRE_SStructSolver   solver,
                            void                **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructDiagScaleSetup(NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector y,
                            NALU_HYPRE_SStructVector x);

/**
 * Solve routine for diagonal preconditioning.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructDiagScale(NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector y,
                       NALU_HYPRE_SStructVector x);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESCreate(MPI_Comm             comm,
                         NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESDestroy(NALU_HYPRE_SStructSolver solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetup(NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_SStructMatrix A,
                        NALU_HYPRE_SStructVector b,
                        NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSolve(NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_SStructMatrix A,
                        NALU_HYPRE_SStructVector b,
                        NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetTol(NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_Real          tol);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetAbsoluteTol(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Real          tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetMinIter(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           min_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetMaxIter(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           max_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetKDim(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_Int           k_dim);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetStopCrit(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           stop_crit);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetPrecond(NALU_HYPRE_SStructSolver          solver,
                             NALU_HYPRE_PtrToSStructSolverFcn  precond,
                             NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                             void                        *precond_solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetLogging(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           logging);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           print_level);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                   NALU_HYPRE_Int           *num_iterations);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                               NALU_HYPRE_Real          *norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetResidual(NALU_HYPRE_SStructSolver   solver,
                              void                **residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESCreate(MPI_Comm             comm,
                             NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESDestroy(NALU_HYPRE_SStructSolver solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetup(NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector b,
                            NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSolve(NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector b,
                            NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetTol(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Real          tol);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetAbsoluteTol(NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Real          tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetMinIter(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           min_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetMaxIter(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           max_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetKDim(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           k_dim);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetPrecond(NALU_HYPRE_SStructSolver          solver,
                                 NALU_HYPRE_PtrToSStructSolverFcn  precond,
                                 NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetLogging(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           logging);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           print_level);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                       NALU_HYPRE_Int           *num_iterations);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                                   NALU_HYPRE_Real          *norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESGetResidual(NALU_HYPRE_SStructSolver   solver,
                                  void                **residual);

NALU_HYPRE_Int
NALU_HYPRE_SStructFlexGMRESSetModifyPC(NALU_HYPRE_SStructSolver    solver,
                                  NALU_HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESCreate(MPI_Comm             comm,
                          NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESDestroy(NALU_HYPRE_SStructSolver solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetup(NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSolve(NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetTol(NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_Real          tol);


NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetAbsoluteTol(NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Real          tol);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetMinIter(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           min_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetMaxIter(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           max_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetKDim(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           k_dim);
NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetAugDim(NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Int           aug_dim);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetPrecond(NALU_HYPRE_SStructSolver          solver,
                              NALU_HYPRE_PtrToSStructSolverFcn  precond,
                              NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void                        *precond_solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetLogging(NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           logging);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           print_level);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                    NALU_HYPRE_Int           *num_iterations);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                                NALU_HYPRE_Real          *norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetResidual(NALU_HYPRE_SStructSolver   solver,
                               void                **residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABCreate(MPI_Comm             comm,
                            NALU_HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABDestroy(NALU_HYPRE_SStructSolver solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetup(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSolve(NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetTol(NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Real          tol);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetAbsoluteTol(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Real          tol);
/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetMinIter(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           min_iter);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetMaxIter(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           max_iter);

/*
 * RE-VISIT
 **/
NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetStopCrit(NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           stop_crit);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetPrecond(NALU_HYPRE_SStructSolver          solver,
                                NALU_HYPRE_PtrToSStructSolverFcn  precond,
                                NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                                void                        *precond_solver);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetLogging(NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int           logging);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetPrintLevel(NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int           level);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetNumIterations(NALU_HYPRE_SStructSolver  solver,
                                      NALU_HYPRE_Int           *num_iterations);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(NALU_HYPRE_SStructSolver  solver,
                                                  NALU_HYPRE_Real          *norm);

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetResidual(NALU_HYPRE_SStructSolver   solver,
                                 void                **residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref Eigensolvers.
 *
 * @{
 **/

/**
  * Load interface interpreter.  Vector part loaded with nalu_hypre_SStructKrylov
  * functions and multivector part loaded with mv_TempMultiVector functions.
  **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSetupInterpreter(mv_InterfaceInterpreter *i);

/**
  * Load Matvec interpreter with nalu_hypre_SStructKrylov functions.
  **/
NALU_HYPRE_Int
NALU_HYPRE_SStructSetupMatvec(NALU_HYPRE_MatvecFunctions *mv);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**@}*/

#ifdef __cplusplus
}
#endif

#endif

