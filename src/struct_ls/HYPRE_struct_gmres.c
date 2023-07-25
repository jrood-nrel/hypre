/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   nalu_hypre_GMRESFunctions * gmres_functions =
      nalu_hypre_GMRESFunctionsCreate(
         nalu_hypre_StructKrylovCAlloc, nalu_hypre_StructKrylovFree,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovCreateVectorArray,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );

   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_GMRESCreate( gmres_functions ) );

   return nalu_hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_GMRESDestroy( (void *) solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetup( NALU_HYPRE_StructSolver solver,
                        NALU_HYPRE_StructMatrix A,
                        NALU_HYPRE_StructVector b,
                        NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_GMRESSetup( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSolve( NALU_HYPRE_StructSolver solver,
                        NALU_HYPRE_StructMatrix A,
                        NALU_HYPRE_StructVector b,
                        NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_GMRESSolve( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetTol( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_GMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetAbsoluteTol( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Real         atol    )
{
   return ( NALU_HYPRE_GMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, atol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetMaxIter( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_GMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetKDim( NALU_HYPRE_StructSolver solver,
                          NALU_HYPRE_Int          k_dim )
{
   return ( NALU_HYPRE_GMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetPrecond( NALU_HYPRE_StructSolver         solver,
                             NALU_HYPRE_PtrToStructSolverFcn precond,
                             NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                             NALU_HYPRE_StructSolver         precond_solver )
{
   return ( NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) precond,
                                   (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                   (NALU_HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetLogging( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          logging )
{
   return ( NALU_HYPRE_GMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          print_level )
{
   return ( NALU_HYPRE_GMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                   NALU_HYPRE_Int          *num_iterations )
{
   return ( NALU_HYPRE_GMRESGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                               NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, norm ) );
}


