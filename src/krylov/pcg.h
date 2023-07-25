/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Preconditioned conjugate gradient (Omin) headers
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_PCG_HEADER
#define nalu_hypre_KRYLOV_PCG_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic PCG Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic PCG linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGData and nalu_hypre_PCGFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name PCG structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_PCGSFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size );
   NALU_HYPRE_Int    (*Free)          ( void *ptr );
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );

   NALU_HYPRE_Int    (*precond)();
   NALU_HYPRE_Int    (*precond_setup)();

} nalu_hypre_PCGFunctions;

/**
 * The {\tt hypre\_PCGData} object ...
 **/

/*
 Summary of Parameters to Control Stopping Test:
 - Standard (default) error tolerance: |delta-residual|/|right-hand-side|<tol
 where the norm is an energy norm wrt preconditioner, |r|=sqrt(<Cr,r>).
 - two_norm!=0 means: the norm is the L2 norm, |r|=sqrt(<r,r>)
 - rel_change!=0 means: if pass the other stopping criteria, also check the
 relative change in the solution x.  Pass iff this relative change is small.
 - tol = relative error tolerance, as above
 -a_tol = absolute convergence tolerance (default is 0.0)
   If one desires the convergence test to check the absolute
   convergence tolerance *only*, then set the relative convergence
   tolerance to 0.0.  (The default convergence test is  <C*r,r> <=
   max(relative_tolerance^2 * <C*b, b>, absolute_tolerance^2)
- cf_tol = convergence factor tolerance; if >0 used for special test
  for slow convergence
- stop_crit!=0 means (TO BE PHASED OUT):
  pure absolute error tolerance rather than a pure relative
  error tolerance on the residual.  Never applies if rel_change!=0 or atolf!=0.
 - atolf = absolute error tolerance factor to be used _together_ with the
 relative error tolerance, |delta-residual| / ( atolf + |right-hand-side| ) < tol
  (To BE PHASED OUT)
 - recompute_residual means: when the iteration seems to be converged, recompute the
 residual from scratch (r=b-Ax) and use this new residual to repeat the convergence test.
 This can be expensive, use this only if you have seen a problem with the regular
 residual computation.
 - recompute_residual_p means: recompute the residual from scratch (r=b-Ax)
 every "recompute_residual_p" iterations.  This can be expensive and degrade the
 convergence. Use it only if you have seen a problem with the regular residual
 computation.
*/

typedef struct
{
   NALU_HYPRE_Real   tol;
   NALU_HYPRE_Real   atolf;
   NALU_HYPRE_Real   cf_tol;
   NALU_HYPRE_Real   a_tol;
   NALU_HYPRE_Real   rtol;
   NALU_HYPRE_Int    max_iter;
   NALU_HYPRE_Int    two_norm;
   NALU_HYPRE_Int    rel_change;
   NALU_HYPRE_Int    recompute_residual;
   NALU_HYPRE_Int    recompute_residual_p;
   NALU_HYPRE_Int    stop_crit;
   NALU_HYPRE_Int    converged;
   NALU_HYPRE_Int    hybrid;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

   NALU_HYPRE_Int  owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void      *matvec_data;
   void      *precond_data;

   nalu_hypre_PCGFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int    num_iterations;
   NALU_HYPRE_Real   rel_residual_norm;

   NALU_HYPRE_Int    print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int    logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   NALU_HYPRE_Real  *rel_norms;

} nalu_hypre_PCGData;

#define nalu_hypre_PCGDataOwnsMatvecData(pcgdata)  ((pcgdata) -> owns_matvec_data)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic PCG Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

nalu_hypre_PCGFunctions *
nalu_hypre_PCGFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size ),
   NALU_HYPRE_Int    (*Free)          ( void *ptr ),
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y ),
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   NALU_HYPRE_Int    (*ClearVector)   ( void *x ),
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
   NALU_HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   NALU_HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
nalu_hypre_PCGCreate( nalu_hypre_PCGFunctions *pcg_functions );

#ifdef __cplusplus
}
#endif

#endif
