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

#ifndef hypre_KRYLOV_BiCGSTAB_HEADER
#define hypre_KRYLOV_BiCGSTAB_HEADER

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
 * hypre_BiCGSTABData and hypre_BiCGSTABFunctions
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
char *hypre_ParKrylovCAlloc( NALU_HYPRE_Int count , NALU_HYPRE_Int elt_size );
NALU_HYPRE_Int hypre_ParKrylovFree( char *ptr );
void *hypre_ParKrylovCreateVectorArray( NALU_HYPRE_Int n , void *vvector );
NALU_HYPRE_Int hypre_ParKrylovMatvecT( void *matvec_data , NALU_HYPRE_Real alpha , void *A , void *x , NALU_HYPRE_Real beta , void *y );
*/
/* functions in pcg_struct.c which are used here:
  void *hypre_ParKrylovCreateVector( void *vvector );
  NALU_HYPRE_Int hypre_ParKrylovDestroyVector( void *vvector );
  void *hypre_ParKrylovMatvecCreate( void *A , void *x );
  NALU_HYPRE_Int hypre_ParKrylovMatvec( void *matvec_data , NALU_HYPRE_Real alpha , void *A , void *x , NALU_HYPRE_Real beta , void *y );
  NALU_HYPRE_Int hypre_ParKrylovMatvecDestroy( void *matvec_data );
  NALU_HYPRE_Real hypre_ParKrylovInnerProd( void *x , void *y );
  NALU_HYPRE_Int hypre_ParKrylovCopyVector( void *x , void *y );
  NALU_HYPRE_Int hypre_ParKrylovClearVector( void *x );
  NALU_HYPRE_Int hypre_ParKrylovScaleVector( NALU_HYPRE_Real alpha , void *x );
  NALU_HYPRE_Int hypre_ParKrylovAxpy( NALU_HYPRE_Real alpha , void *x , void *y );
  NALU_HYPRE_Int hypre_ParKrylovCommInfo( void *A , NALU_HYPRE_Int *my_id , NALU_HYPRE_Int *num_procs );
  NALU_HYPRE_Int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
  NALU_HYPRE_Int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );
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

} hypre_BiCGSTABFunctions;

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

   hypre_BiCGSTABFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int      logging;
   NALU_HYPRE_Int      print_level;
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} hypre_BiCGSTABData;

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

hypre_BiCGSTABFunctions *
hypre_BiCGSTABFunctionsCreate(
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
hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions * bicgstab_functions );


#ifdef __cplusplus
}
#endif

#endif
