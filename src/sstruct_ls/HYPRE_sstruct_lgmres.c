/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESCreate( MPI_Comm             comm,
                           NALU_HYPRE_SStructSolver *solver )
{
   nalu_hypre_LGMRESFunctions * lgmres_functions =
      nalu_hypre_LGMRESFunctionsCreate(
         nalu_hypre_SStructKrylovCAlloc, nalu_hypre_SStructKrylovFree, nalu_hypre_SStructKrylovCommInfo,
         nalu_hypre_SStructKrylovCreateVector,
         nalu_hypre_SStructKrylovCreateVectorArray,
         nalu_hypre_SStructKrylovDestroyVector, nalu_hypre_SStructKrylovMatvecCreate,
         nalu_hypre_SStructKrylovMatvec, nalu_hypre_SStructKrylovMatvecDestroy,
         nalu_hypre_SStructKrylovInnerProd, nalu_hypre_SStructKrylovCopyVector,
         nalu_hypre_SStructKrylovClearVector,
         nalu_hypre_SStructKrylovScaleVector, nalu_hypre_SStructKrylovAxpy,
         nalu_hypre_SStructKrylovIdentitySetup, nalu_hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_LGMRESCreate( lgmres_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_LGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetup( NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_LGMRESSetup( (NALU_HYPRE_Solver) solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSolve( NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_SStructMatrix A,
                          NALU_HYPRE_SStructVector b,
                          NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_LGMRESSolve( (NALU_HYPRE_Solver) solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetKDim( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int           k_dim )
{
   return ( NALU_HYPRE_LGMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetAugDim( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           aug_dim )
{
   return ( NALU_HYPRE_LGMRESSetAugDim( (NALU_HYPRE_Solver) solver, aug_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetTol( NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_LGMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetAbsoluteTol( NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Real          atol )
{
   return ( NALU_HYPRE_LGMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, atol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetMinIter( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           min_iter )
{
   return ( NALU_HYPRE_LGMRESSetMinIter( (NALU_HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetMaxIter( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           max_iter )
{
   return ( NALU_HYPRE_LGMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetPrecond( NALU_HYPRE_SStructSolver          solver,
                               NALU_HYPRE_PtrToSStructSolverFcn  precond,
                               NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                               void *          precond_data )
{
   return ( NALU_HYPRE_LGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) precond,
                                    (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                    (NALU_HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetLogging( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           logging )
{
   return ( NALU_HYPRE_LGMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           level )
{
   return ( NALU_HYPRE_LGMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                     NALU_HYPRE_Int           *num_iterations )
{
   return ( NALU_HYPRE_LGMRESGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                 NALU_HYPRE_Real          *norm )
{
   return ( NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructLGMRESGetResidual( NALU_HYPRE_SStructSolver  solver,
                                void              **residual )
{
   return ( NALU_HYPRE_LGMRESGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}
