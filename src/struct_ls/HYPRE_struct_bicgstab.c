/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   nalu_hypre_BiCGSTABFunctions * bicgstab_functions =
      nalu_hypre_BiCGSTABFunctionsCreate(
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );

   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_BiCGSTABCreate( bicgstab_functions ) );

   return nalu_hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetup( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_StructMatrix A,
                           NALU_HYPRE_StructVector b,
                           NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_BiCGSTABSetup( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSolve( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_StructMatrix A,
                           NALU_HYPRE_StructVector b,
                           NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_BiCGSTABSolve( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetTol( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_BiCGSTABSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetAbsoluteTol( NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_BiCGSTABSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetMaxIter( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_BiCGSTABSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}


/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetPrecond( NALU_HYPRE_StructSolver         solver,
                                NALU_HYPRE_PtrToStructSolverFcn precond,
                                NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                                NALU_HYPRE_StructSolver         precond_solver )
{
   return ( NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) precond,
                                      (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                      (NALU_HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetLogging( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          logging )
{
   return ( NALU_HYPRE_BiCGSTABSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Int level)
{
   return ( NALU_HYPRE_BiCGSTABSetPrintLevel( (NALU_HYPRE_Solver) solver, level ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                      NALU_HYPRE_Int          *num_iterations )
{
   return ( NALU_HYPRE_BiCGSTABGetNumIterations( (NALU_HYPRE_Solver) solver,
                                            num_iterations ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                                  NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver,
                                                        norm ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructBiCGSTABGetResidual( NALU_HYPRE_StructSolver  solver,
                                 void  **residual)
{
   return ( NALU_HYPRE_BiCGSTABGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}

