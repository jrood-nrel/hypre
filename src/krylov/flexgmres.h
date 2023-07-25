/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * FLEXGMRES flexible gmres
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_FLEXGMRES_HEADER
#define nalu_hypre_KRYLOV_FLEXGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic FlexGMRES Interface
 *
 * A general description of the interface goes here...
 *
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_FlexGMRESData and nalu_hypre_FlexGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name FlexGMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_FlexGMRESFunctions} object ...
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

   NALU_HYPRE_Int    (*precond)(void *vdata, void *A, void *b, void *x );
   NALU_HYPRE_Int    (*precond_setup)(void *vdata, void *A, void *b, void *x );

   NALU_HYPRE_Int    (*modify_pc)(void *precond_data, NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm );

} nalu_hypre_FlexGMRESFunctions;

/**
 * The {\tt hypre\_FlexGMRESData} object ...
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

   void   **pre_vecs;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   nalu_hypre_FlexGMRESFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_FlexGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic FlexGMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

nalu_hypre_FlexGMRESFunctions *
nalu_hypre_FlexGMRESFunctionsCreate(
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
nalu_hypre_FlexGMRESCreate( nalu_hypre_FlexGMRESFunctions *fgmres_functions );

#ifdef __cplusplus
}
#endif
#endif
