/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_LOBPCG_SOLVER
#define hypre_LOBPCG_SOLVER

#include "NALU_HYPRE_krylov.h"

#include "fortran_matrix.h"
#include "multivector.h"
#include "interpreter.h"
#include "temp_multivector.h"
#include "NALU_HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup Eigensolvers Eigensolvers
 *
 * These eigensolvers support many of the matrix/vector storage schemes in
 * hypre.  They should be used in conjunction with the storage-specific
 * interfaces.
 *
 * @memo A basic interface for eigensolvers
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name EigenSolvers
 *
 * @{
 **/

#ifndef NALU_HYPRE_SOLVER_STRUCT
#define NALU_HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
/**
 * The solver object.
 **/
typedef struct hypre_Solver_struct *NALU_HYPRE_Solver;
#endif

#ifndef NALU_HYPRE_MATRIX_STRUCT
#define NALU_HYPRE_MATRIX_STRUCT
struct hypre_Matrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_Matrix_struct *NALU_HYPRE_Matrix;
#endif

#ifndef NALU_HYPRE_VECTOR_STRUCT
#define NALU_HYPRE_VECTOR_STRUCT
struct hypre_Vector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_Vector_struct *NALU_HYPRE_Vector;
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name LOBPCG Eigensolver
 *
 * @{
 **/

/**
 * LOBPCG constructor.
 */
NALU_HYPRE_Int NALU_HYPRE_LOBPCGCreate(mv_InterfaceInterpreter *interpreter,
                             NALU_HYPRE_MatvecFunctions   *mvfunctions,
                             NALU_HYPRE_Solver            *solver);

/**
 * LOBPCG destructor.
 */
NALU_HYPRE_Int NALU_HYPRE_LOBPCGDestroy(NALU_HYPRE_Solver solver);

/**
 * (Optional) Set the preconditioner to use.  If not called, preconditioning is
 * not used.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetPrecond(NALU_HYPRE_Solver         solver,
                                 NALU_HYPRE_PtrToSolverFcn precond,
                                 NALU_HYPRE_PtrToSolverFcn precond_setup,
                                 NALU_HYPRE_Solver         precond_solver);

/**
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGGetPrecond(NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Solver *precond_data_ptr);

/**
 * Set up \e A and the preconditioner (if there is one).
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetup(NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Matrix A,
                            NALU_HYPRE_Vector b,
                            NALU_HYPRE_Vector x);

/**
 * (Optional) Set up \e B.  If not called, B = I.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetupB(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Matrix B,
                             NALU_HYPRE_Vector x);

/**
 * (Optional) Set the preconditioning to be applied to Tx = b, not Ax = b.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetupT(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Matrix T,
                             NALU_HYPRE_Vector x);

/**
 * Solve A x = lambda B x, y'x = 0.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSolve(NALU_HYPRE_Solver       solver,
                            mv_MultiVectorPtr  y,
                            mv_MultiVectorPtr  x,
                            NALU_HYPRE_Real        *lambda );

/**
 * (Optional) Set the absolute convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetTol(NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real   tol);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetRTol(NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetMaxIter(NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          max_iter);

/**
 * Define which initial guess for inner PCG iterations to use: \e mode = 0:
 * use zero initial guess, otherwise use RHS.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetPrecondUsageMode(NALU_HYPRE_Solver solver,
                                          NALU_HYPRE_Int          mode);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
NALU_HYPRE_Int NALU_HYPRE_LOBPCGSetPrintLevel(NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int          level);

/* Returns the pointer to residual norms matrix (blockSize x 1) */
utilities_FortranMatrix*
NALU_HYPRE_LOBPCGResidualNorms(NALU_HYPRE_Solver solver);

/* Returns the pointer to residual norms history matrix (blockSize x maxIter) */
utilities_FortranMatrix*
NALU_HYPRE_LOBPCGResidualNormsHistory(NALU_HYPRE_Solver solver);

/* Returns the pointer to eigenvalue history matrix (blockSize x maxIter) */
utilities_FortranMatrix*
NALU_HYPRE_LOBPCGEigenvaluesHistory(NALU_HYPRE_Solver solver);

/* Returns the number of iterations performed by LOBPCG */
NALU_HYPRE_Int NALU_HYPRE_LOBPCGIterations(NALU_HYPRE_Solver solver);

void hypre_LOBPCGMultiOperatorB(void *data,
                                void *x,
                                void *y);

void lobpcg_MultiVectorByMultiVector(mv_MultiVectorPtr        x,
                                     mv_MultiVectorPtr        y,
                                     utilities_FortranMatrix *xy);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
