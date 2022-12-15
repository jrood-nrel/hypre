/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_BiCGSTAB_HEADER
#define nalu_hypre_KRYLOV_BiCGSTAB_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic BiCGSTAB Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic BiCGSTAB linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABData and nalu_hypre_BiCGSTABFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name BiCGSTAB structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_BiCGSTABSFunctions} object ...
 **/

/* functions in pcg_struct.c which aren't used here:
char *nalu_hypre_ParKrylovCAlloc( NALU_HYPRE_Int count , NALU_HYPRE_Int elt_size );
NALU_HYPRE_Int nalu_hypre_ParKrylovFree( char *ptr );
void *nalu_hypre_ParKrylovCreateVectorArray( NALU_HYPRE_Int n , void *vvector );
NALU_HYPRE_Int nalu_hypre_ParKrylovMatvecT( void *matvec_data , NALU_HYPRE_Real alpha , void *A , void *x , NALU_HYPRE_Real beta , void *y );
*/
/* functions in pcg_struct.c which are used here:
  void *nalu_hypre_ParKrylovCreateVector( void *vvector );
  NALU_HYPRE_Int nalu_hypre_ParKrylovDestroyVector( void *vvector );
  void *nalu_hypre_ParKrylovMatvecCreate( void *A , void *x );
  NALU_HYPRE_Int nalu_hypre_ParKrylovMatvec( void *matvec_data , NALU_HYPRE_Real alpha , void *A , void *x , NALU_HYPRE_Real beta , void *y );
  NALU_HYPRE_Int nalu_hypre_ParKrylovMatvecDestroy( void *matvec_data );
  NALU_HYPRE_Real nalu_hypre_ParKrylovInnerProd( void *x , void *y );
  NALU_HYPRE_Int nalu_hypre_ParKrylovCopyVector( void *x , void *y );
  NALU_HYPRE_Int nalu_hypre_ParKrylovClearVector( void *x );
  NALU_HYPRE_Int nalu_hypre_ParKrylovScaleVector( NALU_HYPRE_Real alpha , void *x );
  NALU_HYPRE_Int nalu_hypre_ParKrylovAxpy( NALU_HYPRE_Real alpha , void *x , void *y );
  NALU_HYPRE_Int nalu_hypre_ParKrylovCommInfo( void *A , NALU_HYPRE_Int *my_id , NALU_HYPRE_Int *num_procs );
  NALU_HYPRE_Int nalu_hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
  NALU_HYPRE_Int nalu_hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
*/

typedef struct
{
   void *     (*CreateVector)  ( void *vvector );
   NALU_HYPRE_Int  (*DestroyVector) ( void *vvector );
   void *     (*MatvecCreate)  ( void *A, void *x );
   NALU_HYPRE_Int  (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                 void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int  (*MatvecDestroy) ( void *matvec_data );
   NALU_HYPRE_Real (*InnerProd)     ( void *x, void *y );
   NALU_HYPRE_Int  (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int  (*ClearVector)   ( void *x );
   NALU_HYPRE_Int  (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int  (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );
   NALU_HYPRE_Int  (*CommInfo)      ( void *A, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs );
   NALU_HYPRE_Int  (*precond_setup) (void *vdata, void *A, void *b, void *x);
   NALU_HYPRE_Int  (*precond)       (void *vdata, void *A, void *b, void *x);

} nalu_hypre_BiCGSTABFunctions;

/**
 * The {\tt hypre\_BiCGSTABData} object ...
 **/

typedef struct
{
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      stop_crit;
   NALU_HYPRE_Int      converged;
   NALU_HYPRE_Int      hybrid;
   NALU_HYPRE_Real   tol;
   NALU_HYPRE_Real   cf_tol;
   NALU_HYPRE_Real   rel_residual_norm;
   NALU_HYPRE_Real   a_tol;


   void  *A;
   void  *r;
   void  *r0;
   void  *s;
   void  *v;
   void  *p;
   void  *q;

   void  *matvec_data;
   void    *precond_data;

   nalu_hypre_BiCGSTABFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int      logging;
   NALU_HYPRE_Int      print_level;
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_BiCGSTABData;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name generic BiCGSTAB Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

nalu_hypre_BiCGSTABFunctions *
nalu_hypre_BiCGSTABFunctionsCreate(
   void *     (*CreateVector)  ( void *vvector ),
   NALU_HYPRE_Int  (*DestroyVector) ( void *vvector ),
   void *     (*MatvecCreate)  ( void *A, void *x ),
   NALU_HYPRE_Int  (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                 void *x, NALU_HYPRE_Complex beta, void *y ),
   NALU_HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
   NALU_HYPRE_Real (*InnerProd)     ( void *x, void *y ),
   NALU_HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
   NALU_HYPRE_Int  (*ClearVector)   ( void *x ),
   NALU_HYPRE_Int  (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
   NALU_HYPRE_Int  (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
   NALU_HYPRE_Int  (*CommInfo)      ( void *A, NALU_HYPRE_Int *my_id,
                                 NALU_HYPRE_Int *num_procs ),
   NALU_HYPRE_Int  (*PrecondSetup)  (void *vdata, void *A, void *b, void *x ),
   NALU_HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
);

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
nalu_hypre_BiCGSTABCreate( nalu_hypre_BiCGSTABFunctions * bicgstab_functions );


#ifdef __cplusplus
}
#endif

#endif
