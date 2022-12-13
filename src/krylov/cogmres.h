/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * COGMRES gmres
 *
 *****************************************************************************/

#ifndef hypre_KRYLOV_COGMRES_HEADER
#define hypre_KRYLOV_COGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic COGMRES Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic COGMRES linear solver interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_COGMRESData and hypre_COGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name COGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_COGMRESFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size );
   NALU_HYPRE_Int    (*Free)          ( void *ptr );
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   void *       (*CreateVectorArray)  ( NALU_HYPRE_Int size, void *vectors );
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   NALU_HYPRE_Int    (*MassInnerProd) ( void *x, void **p, NALU_HYPRE_Int k, NALU_HYPRE_int unroll, void *result);
   NALU_HYPRE_Int    (*MassDotpTwo)( void *x, void *y, void **p, NALU_HYPRE_Int k, void *result_x,
                                NALU_HYPRE_int unroll, void *result_y);
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );
   NALU_HYPRE_Int    (*MassAxpy)      ( NALU_HYPRE_Complex *alpha, void **x, void *y, NALU_HYPRE_Int k,
                                   NALU_HYPRE_Int unroll);
   NALU_HYPRE_Int    (*precond)       ();
   NALU_HYPRE_Int    (*precond_setup) ();

   NALU_HYPRE_Int    (*modify_pc)(void *precond_data, NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm );

} hypre_COGMRESFunctions;

/**
 * The {\tt hypre\_COGMRESData} object ...
 **/

typedef struct
{
   NALU_HYPRE_Int      k_dim;
   NALU_HYPRE_Int      unroll;
   NALU_HYPRE_Int      cgs;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      rel_change;
   NALU_HYPRE_Int      skip_real_r_check;
   NALU_HYPRE_Int      converged;
   NALU_HYPRE_Real   tol;
   NALU_HYPRE_Real   cf_tol;
   NALU_HYPRE_Real   a_tol;
   NALU_HYPRE_Real   rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   hypre_COGMRESFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_COGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic COGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_COGMRESFunctions *
hypre_COGMRESFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size ),
   NALU_HYPRE_Int    (*Free)          ( void *ptr ),
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( NALU_HYPRE_Int size, void *vectors ),
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y ),
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   NALU_HYPRE_Int    (*MassInnerProd) ( void *x, void **p, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void *result),
   NALU_HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,
                                   void *result_x, void *result_y),
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   NALU_HYPRE_Int    (*ClearVector)   ( void *x ),
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
   NALU_HYPRE_Int    (*MassAxpy)      ( NALU_HYPRE_Complex *alpha, void **x, void *y, NALU_HYPRE_Int k,
                                   NALU_HYPRE_Int unroll),
   NALU_HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   NALU_HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_COGMRESCreate( hypre_COGMRESFunctions *gmres_functions );

#ifdef __cplusplus
}
#endif
#endif
