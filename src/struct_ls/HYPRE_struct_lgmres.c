/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   nalu_hypre_LGMRESFunctions * lgmres_functions =
      nalu_hypre_LGMRESFunctionsCreate(
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

   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_LGMRESCreate( lgmres_functions ) );

   return nalu_hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_LGMRESDestroy( (void *) solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetup( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_LGMRESSetup( (NALU_HYPRE_Solver) solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSolve( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_LGMRESSolve( (NALU_HYPRE_Solver) solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetTol( NALU_HYPRE_StructSolver solver,
                          NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_LGMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}
/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetAbsoluteTol( NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_LGMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}
/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetMaxIter( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_LGMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetKDim( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          k_dim )
{
   return ( NALU_HYPRE_LGMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}



/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetAugDim( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          aug_dim )
{
   return ( NALU_HYPRE_LGMRESSetAugDim( (NALU_HYPRE_Solver) solver, aug_dim ) );
}


/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetPrecond( NALU_HYPRE_StructSolver         solver,
                              NALU_HYPRE_PtrToStructSolverFcn precond,
                              NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                              NALU_HYPRE_StructSolver         precond_solver )
{
   return ( NALU_HYPRE_LGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) precond,
                                    (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                    (NALU_HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetLogging( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          logging )
{
   return ( NALU_HYPRE_LGMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          print_level )
{
   return ( NALU_HYPRE_LGMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                    NALU_HYPRE_Int          *num_iterations )
{
   return ( NALU_HYPRE_LGMRESGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructLGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                                NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver,
                                                      norm ) );
}


