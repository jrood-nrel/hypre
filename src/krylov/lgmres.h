/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LGMRES lgmres
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_LGMRES_HEADER
#define nalu_hypre_KRYLOV_LGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic LGMRES Interface
 *
 * A general description of the interface goes here...
 *
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESData and nalu_hypre_LGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name LGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_LGMRESFunctions} object ...
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
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );

   NALU_HYPRE_Int    (*precond)       ();
   NALU_HYPRE_Int    (*precond_setup) ();

} nalu_hypre_LGMRESFunctions;

/**
 * The {\tt hypre\_LGMRESData} object ...
 **/

typedef struct
{
   NALU_HYPRE_Int      k_dim;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      rel_change;
   NALU_HYPRE_Int      stop_crit;
   NALU_HYPRE_Int      converged;
   NALU_HYPRE_Real   tol;
   NALU_HYPRE_Real   cf_tol;
   NALU_HYPRE_Real   a_tol;
   NALU_HYPRE_Real   rel_residual_norm;

   /*lgmres specific stuff */
   NALU_HYPRE_Int      aug_dim;
   NALU_HYPRE_Int      approx_constant;
   void   **aug_vecs;
   NALU_HYPRE_Int     *aug_order;
   void   **a_aug_vecs;
   /*---*/

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   nalu_hypre_LGMRESFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_LGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic LGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

nalu_hypre_LGMRESFunctions *
nalu_hypre_LGMRESFunctionsCreate(
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
nalu_hypre_LGMRESCreate( nalu_hypre_LGMRESFunctions *lgmres_functions );

#ifdef __cplusplus
}
#endif
#endif
