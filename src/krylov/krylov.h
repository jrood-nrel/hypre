/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "NALU_HYPRE_krylov.h"

#ifndef nalu_hypre_KRYLOV_HEADER
#define nalu_hypre_KRYLOV_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"

#define nalu_hypre_CTAllocF(type, count, funcs, location) \
  ( (type *)(*(funcs->CAlloc))((size_t)(count), (size_t)sizeof(type), location) )

#define nalu_hypre_TFreeF( ptr, funcs ) ( (*(funcs->Free))((void *)ptr), ptr = NULL )

#ifdef __cplusplus
extern "C" {
#endif

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
   void *nalu_hypre_ParKrylovCAlloc( NALU_HYPRE_Int count , NALU_HYPRE_Int elt_size );
   NALU_HYPRE_Int nalu_hypre_ParKrylovFree( void *ptr );
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

#define nalu_hypre_BiCGSTABDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

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
      NALU_HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
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

/******************************************************************************
 *
 * cgnr (conjugate gradient on the normal equations A^TAx = A^Tb) functions
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_CGNR_HEADER
#define nalu_hypre_KRYLOV_CGNR_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic CGNR Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic CGNR linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_CGNRData and nalu_hypre_CGNRFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name CGNR structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_CGNRSFunctions} object ...
 **/

typedef struct
{
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int    (*MatvecT)       ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y );
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data );
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );
   NALU_HYPRE_Int    (*precond_setup) ( void *vdata, void *A, void *b, void *x );
   NALU_HYPRE_Int    (*precond)       ( void *vdata, void *A, void *b, void *x );
   NALU_HYPRE_Int    (*precondT)      ( void *vdata, void *A, void *b, void *x );

} nalu_hypre_CGNRFunctions;

/**
 * The {\tt hypre\_CGNRData} object ...
 **/

typedef struct
{
   NALU_HYPRE_Real   tol;
   NALU_HYPRE_Real   rel_residual_norm;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      stop_crit;

   void    *A;
   void    *p;
   void    *q;
   void    *r;
   void    *t;

   void    *matvec_data;
   void    *precond_data;

   nalu_hypre_CGNRFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   /* additional log info (logged when `logging' > 0) */
   NALU_HYPRE_Int      logging;
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_CGNRData;


#ifdef __cplusplus
extern "C" {
#endif

   /**
    * @name generic CGNR Solver
    *
    * Description...
    **/
   /*@{*/

   /**
    * Description...
    *
    * @param param [IN] ...
    **/
   nalu_hypre_CGNRFunctions *
   nalu_hypre_CGNRFunctionsCreate(
      NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                      NALU_HYPRE_Int   *num_procs ),
      void *       (*CreateVector)  ( void *vector ),
      NALU_HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                      void *x, NALU_HYPRE_Complex beta, void *y ),
      NALU_HYPRE_Int    (*MatvecT)       ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                      void *x, NALU_HYPRE_Complex beta, void *y ),
      NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
      NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      NALU_HYPRE_Int    (*ClearVector)   ( void *x ),
      NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
      NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
      NALU_HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      NALU_HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
      NALU_HYPRE_Int    (*PrecondT)      ( void *vdata, void *A, void *b, void *x )
   );

   /**
    * Description...
    *
    * @param param [IN] ...
    **/

   void *
   nalu_hypre_CGNRCreate( nalu_hypre_CGNRFunctions *cgnr_functions );

#ifdef __cplusplus
}
#endif

#endif

/******************************************************************************
 *
 * GMRES gmres
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_GMRES_HEADER
#define nalu_hypre_KRYLOV_GMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic GMRES Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic GMRES linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_GMRESData and nalu_hypre_GMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_GMRESFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
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

   NALU_HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   NALU_HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

} nalu_hypre_GMRESFunctions;

/**
 * The {\tt hypre\_GMRESData} object ...
 **/

typedef struct
{
   NALU_HYPRE_Int      k_dim;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      rel_change;
   NALU_HYPRE_Int      skip_real_r_check;
   NALU_HYPRE_Int      stop_crit;
   NALU_HYPRE_Int      converged;
   NALU_HYPRE_Int      hybrid;
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

   nalu_hypre_GMRESFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_GMRESData;

#define nalu_hypre_GMRESDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

#ifdef __cplusplus
extern "C" {
#endif

   /**
    * @name generic GMRES Solver
    *
    * Description...
    **/
   /*@{*/

   /**
    * Description...
    *
    * @param param [IN] ...
    **/

   nalu_hypre_GMRESFunctions *
   nalu_hypre_GMRESFunctionsCreate(
      void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
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
   nalu_hypre_GMRESCreate( nalu_hypre_GMRESFunctions *gmres_functions );

#ifdef __cplusplus
}
#endif
#endif

/***********KS code ****************/
/******************************************************************************
 *
 * COGMRES cogmres
 *
 *****************************************************************************/

#ifndef nalu_hypre_KRYLOV_COGMRES_HEADER
#define nalu_hypre_KRYLOV_COGMRES_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic GMRES Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic GMRES linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_COGMRESData and nalu_hypre_COGMRESFunctions
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_GMRESFunctions} object ...
 **/

typedef struct
{
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
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
   NALU_HYPRE_Int    (*MassInnerProd) ( void *x, void **p, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void *result);
   NALU_HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll,
                                   void *result_x, void *result_y);
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );
   NALU_HYPRE_Int    (*MassAxpy)      ( NALU_HYPRE_Complex * alpha, void **x, void *y, NALU_HYPRE_Int k,
                                   NALU_HYPRE_Int unroll);
   NALU_HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   NALU_HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

   NALU_HYPRE_Int    (*modify_pc)( void *precond_data, NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm);


} nalu_hypre_COGMRESFunctions;

/**
 * The {\tt hypre\_GMRESData} object ...
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
   NALU_HYPRE_Int      stop_crit;
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

   nalu_hypre_COGMRESFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   char    *log_file_name;

} nalu_hypre_COGMRESData;

#ifdef __cplusplus
extern "C" {
#endif

   /**
    * @name generic GMRES Solver
    *
    * Description...
    **/
   /*@{*/

   /**
    * Description...
    *
    * @param param [IN] ...
    **/

   nalu_hypre_COGMRESFunctions *
   nalu_hypre_COGMRESFunctionsCreate(
      void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
      NALU_HYPRE_Int    (*Free)          ( void *ptr ),
      NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                      NALU_HYPRE_Int   *num_procs ),
      void *       (*CreateVector)  ( void *vector ),
      void *       (*CreateVectorArray)  ( NALU_HYPRE_Int size, void *vectors ),
      NALU_HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A, void *x,
                                      NALU_HYPRE_Complex beta, void *y ),
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
   nalu_hypre_COGMRESCreate( nalu_hypre_COGMRESFunctions *gmres_functions );

#ifdef __cplusplus
}
#endif
#endif



/***********end of KS code *********/



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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
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

   NALU_HYPRE_Int    (*precond)       (void *vdata, void *A, void *b, void *x);
   NALU_HYPRE_Int    (*precond_setup) (void *vdata, void *A, void *b, void *x);

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location),
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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
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

   NALU_HYPRE_Int    (*modify_pc)( void *precond_data, NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm);

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
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
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location );
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

   NALU_HYPRE_Int    (*precond)(void *vdata, void *A, void *b, void *x);
   NALU_HYPRE_Int    (*precond_setup)(void *vdata, void *A, void *b, void *x);

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
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      two_norm;
   NALU_HYPRE_Int      rel_change;
   NALU_HYPRE_Int      recompute_residual;
   NALU_HYPRE_Int      recompute_residual_p;
   NALU_HYPRE_Int      stop_crit;
   NALU_HYPRE_Int      converged;
   NALU_HYPRE_Int      hybrid;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                   If that is ever changed, it still must be kept if logging>1 */

   NALU_HYPRE_Int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   nalu_hypre_PCGFunctions * functions;

   /* log info (always logged) */
   NALU_HYPRE_Int      num_iterations;
   NALU_HYPRE_Real   rel_residual_norm;

   NALU_HYPRE_Int     print_level; /* printing when print_level>0 */
   NALU_HYPRE_Int     logging;  /* extra computations for logging when logging>0 */
   NALU_HYPRE_Real  *norms;
   NALU_HYPRE_Real  *rel_norms;

} nalu_hypre_PCGData;

#define nalu_hypre_PCGDataOwnsMatvecData(pcgdata)  ((pcgdata) -> owns_matvec_data)
#define nalu_hypre_PCGDataHybrid(pcgdata)  ((pcgdata) -> hybrid)

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
      void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
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

/* bicgstab.c */
void *nalu_hypre_BiCGSTABCreate ( nalu_hypre_BiCGSTABFunctions *bicgstab_functions );
NALU_HYPRE_Int nalu_hypre_BiCGSTABDestroy ( void *bicgstab_vdata );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetup ( void *bicgstab_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSolve ( void *bicgstab_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetTol ( void *bicgstab_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetAbsoluteTol ( void *bicgstab_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetConvergenceFactorTol ( void *bicgstab_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetMinIter ( void *bicgstab_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetMaxIter ( void *bicgstab_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetStopCrit ( void *bicgstab_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata, NALU_HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_BiCGSTABGetPrecond ( void *bicgstab_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetLogging ( void *bicgstab_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetHybrid ( void *bicgstab_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_BiCGSTABSetPrintLevel ( void *bicgstab_vdata, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int nalu_hypre_BiCGSTABGetConverged ( void *bicgstab_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_BiCGSTABGetNumIterations ( void *bicgstab_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm ( void *bicgstab_vdata,
                                                       NALU_HYPRE_Real *relative_residual_norm );
NALU_HYPRE_Int nalu_hypre_BiCGSTABGetResidual ( void *bicgstab_vdata, void **residual );

/* cgnr.c */
void *nalu_hypre_CGNRCreate ( nalu_hypre_CGNRFunctions *cgnr_functions );
NALU_HYPRE_Int nalu_hypre_CGNRDestroy ( void *cgnr_vdata );
NALU_HYPRE_Int nalu_hypre_CGNRSetup ( void *cgnr_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_CGNRSolve ( void *cgnr_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_CGNRSetTol ( void *cgnr_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_CGNRSetMinIter ( void *cgnr_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_CGNRSetMaxIter ( void *cgnr_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_CGNRSetStopCrit ( void *cgnr_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_CGNRSetPrecond ( void *cgnr_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 NALU_HYPRE_Int (*precondT )(void*, void*, void*, void*), NALU_HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_CGNRGetPrecond ( void *cgnr_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_CGNRSetLogging ( void *cgnr_vdata, NALU_HYPRE_Int logging );
NALU_HYPRE_Int nalu_hypre_CGNRGetNumIterations ( void *cgnr_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_CGNRGetFinalRelativeResidualNorm ( void *cgnr_vdata,
                                                   NALU_HYPRE_Real *relative_residual_norm );

/* gmres.c */
void *nalu_hypre_GMRESCreate ( nalu_hypre_GMRESFunctions *gmres_functions );
NALU_HYPRE_Int nalu_hypre_GMRESDestroy ( void *gmres_vdata );
NALU_HYPRE_Int nalu_hypre_GMRESGetResidual ( void *gmres_vdata, void **residual );
NALU_HYPRE_Int nalu_hypre_GMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_GMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_GMRESSetKDim ( void *gmres_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int nalu_hypre_GMRESGetKDim ( void *gmres_vdata, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int nalu_hypre_GMRESSetTol ( void *gmres_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_GMRESGetTol ( void *gmres_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_GMRESSetAbsoluteTol ( void *gmres_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_GMRESGetAbsoluteTol ( void *gmres_vdata, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int nalu_hypre_GMRESSetConvergenceFactorTol ( void *gmres_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_GMRESGetConvergenceFactorTol ( void *gmres_vdata, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int nalu_hypre_GMRESSetMinIter ( void *gmres_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_GMRESGetMinIter ( void *gmres_vdata, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int nalu_hypre_GMRESSetMaxIter ( void *gmres_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_GMRESGetMaxIter ( void *gmres_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_GMRESSetRelChange ( void *gmres_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_GMRESGetRelChange ( void *gmres_vdata, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int nalu_hypre_GMRESSetSkipRealResidualCheck ( void *gmres_vdata, NALU_HYPRE_Int skip_real_r_check );
NALU_HYPRE_Int nalu_hypre_GMRESGetSkipRealResidualCheck ( void *gmres_vdata, NALU_HYPRE_Int *skip_real_r_check );
NALU_HYPRE_Int nalu_hypre_GMRESSetStopCrit ( void *gmres_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_GMRESGetStopCrit ( void *gmres_vdata, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int nalu_hypre_GMRESSetPrecond ( void *gmres_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
                                  NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_GMRESGetPrecond ( void *gmres_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_GMRESSetPrintLevel ( void *gmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_GMRESGetPrintLevel ( void *gmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_GMRESSetLogging ( void *gmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_GMRESGetLogging ( void *gmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_GMRESSetHybrid ( void *gmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_GMRESGetNumIterations ( void *gmres_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_GMRESGetConverged ( void *gmres_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_GMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                    NALU_HYPRE_Real *relative_residual_norm );

/* cogmres.c */
void *nalu_hypre_COGMRESCreate ( nalu_hypre_COGMRESFunctions *gmres_functions );
NALU_HYPRE_Int nalu_hypre_COGMRESDestroy ( void *gmres_vdata );
NALU_HYPRE_Int nalu_hypre_COGMRESGetResidual ( void *gmres_vdata, void **residual );
NALU_HYPRE_Int nalu_hypre_COGMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_COGMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_COGMRESSetKDim ( void *gmres_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int nalu_hypre_COGMRESGetKDim ( void *gmres_vdata, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int nalu_hypre_COGMRESSetUnroll ( void *gmres_vdata, NALU_HYPRE_Int unroll );
NALU_HYPRE_Int nalu_hypre_COGMRESGetUnroll ( void *gmres_vdata, NALU_HYPRE_Int *unroll );
NALU_HYPRE_Int nalu_hypre_COGMRESSetCGS ( void *gmres_vdata, NALU_HYPRE_Int cgs );
NALU_HYPRE_Int nalu_hypre_COGMRESGetCGS ( void *gmres_vdata, NALU_HYPRE_Int *cgs );
NALU_HYPRE_Int nalu_hypre_COGMRESSetTol ( void *gmres_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_COGMRESGetTol ( void *gmres_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_COGMRESSetAbsoluteTol ( void *gmres_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_COGMRESGetAbsoluteTol ( void *gmres_vdata, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int nalu_hypre_COGMRESSetConvergenceFactorTol ( void *gmres_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_COGMRESGetConvergenceFactorTol ( void *gmres_vdata, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int nalu_hypre_COGMRESSetMinIter ( void *gmres_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_COGMRESGetMinIter ( void *gmres_vdata, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int nalu_hypre_COGMRESSetMaxIter ( void *gmres_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_COGMRESGetMaxIter ( void *gmres_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_COGMRESSetRelChange ( void *gmres_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_COGMRESGetRelChange ( void *gmres_vdata, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int nalu_hypre_COGMRESSetSkipRealResidualCheck ( void *gmres_vdata, NALU_HYPRE_Int skip_real_r_check );
NALU_HYPRE_Int nalu_hypre_COGMRESGetSkipRealResidualCheck ( void *gmres_vdata, NALU_HYPRE_Int *skip_real_r_check );
NALU_HYPRE_Int nalu_hypre_COGMRESSetPrecond ( void *gmres_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_COGMRESGetPrecond ( void *gmres_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_COGMRESSetPrintLevel ( void *gmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_COGMRESGetPrintLevel ( void *gmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_COGMRESSetLogging ( void *gmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_COGMRESGetLogging ( void *gmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_COGMRESGetNumIterations ( void *gmres_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_COGMRESGetConverged ( void *gmres_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_COGMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                      NALU_HYPRE_Real *relative_residual_norm );
NALU_HYPRE_Int nalu_hypre_COGMRESSetModifyPC ( void *fgmres_vdata, NALU_HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm));



/* flexgmres.c */
void *nalu_hypre_FlexGMRESCreate ( nalu_hypre_FlexGMRESFunctions *fgmres_functions );
NALU_HYPRE_Int nalu_hypre_FlexGMRESDestroy ( void *fgmres_vdata );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetResidual ( void *fgmres_vdata, void **residual );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetup ( void *fgmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSolve ( void *fgmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetKDim ( void *fgmres_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetKDim ( void *fgmres_vdata, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetTol ( void *fgmres_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetTol ( void *fgmres_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetAbsoluteTol ( void *fgmres_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetAbsoluteTol ( void *fgmres_vdata, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetConvergenceFactorTol ( void *fgmres_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetConvergenceFactorTol ( void *fgmres_vdata, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetMinIter ( void *fgmres_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetMinIter ( void *fgmres_vdata, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetMaxIter ( void *fgmres_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetMaxIter ( void *fgmres_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetStopCrit ( void *fgmres_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetStopCrit ( void *fgmres_vdata, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetPrecond ( void *fgmres_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetPrecond ( void *fgmres_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetLogging ( void *fgmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetLogging ( void *fgmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetConverged ( void *fgmres_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata,
                                                        NALU_HYPRE_Real *relative_residual_norm );
NALU_HYPRE_Int nalu_hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata,
                                       NALU_HYPRE_Int (*modify_pc )(void *precond_data, NALU_HYPRE_Int iteration, NALU_HYPRE_Real rel_residual_norm));
NALU_HYPRE_Int nalu_hypre_FlexGMRESModifyPCDefault ( void *precond_data, NALU_HYPRE_Int iteration,
                                           NALU_HYPRE_Real rel_residual_norm );

/* lgmres.c */
void *nalu_hypre_LGMRESCreate ( nalu_hypre_LGMRESFunctions *lgmres_functions );
NALU_HYPRE_Int nalu_hypre_LGMRESDestroy ( void *lgmres_vdata );
NALU_HYPRE_Int nalu_hypre_LGMRESGetResidual ( void *lgmres_vdata, void **residual );
NALU_HYPRE_Int nalu_hypre_LGMRESSetup ( void *lgmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_LGMRESSolve ( void *lgmres_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_LGMRESSetKDim ( void *lgmres_vdata, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int nalu_hypre_LGMRESGetKDim ( void *lgmres_vdata, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int nalu_hypre_LGMRESSetAugDim ( void *lgmres_vdata, NALU_HYPRE_Int aug_dim );
NALU_HYPRE_Int nalu_hypre_LGMRESGetAugDim ( void *lgmres_vdata, NALU_HYPRE_Int *aug_dim );
NALU_HYPRE_Int nalu_hypre_LGMRESSetTol ( void *lgmres_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_LGMRESGetTol ( void *lgmres_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_LGMRESSetAbsoluteTol ( void *lgmres_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_LGMRESGetAbsoluteTol ( void *lgmres_vdata, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int nalu_hypre_LGMRESSetConvergenceFactorTol ( void *lgmres_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_LGMRESGetConvergenceFactorTol ( void *lgmres_vdata, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int nalu_hypre_LGMRESSetMinIter ( void *lgmres_vdata, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int nalu_hypre_LGMRESGetMinIter ( void *lgmres_vdata, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int nalu_hypre_LGMRESSetMaxIter ( void *lgmres_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_LGMRESGetMaxIter ( void *lgmres_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_LGMRESSetStopCrit ( void *lgmres_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_LGMRESGetStopCrit ( void *lgmres_vdata, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int nalu_hypre_LGMRESSetPrecond ( void *lgmres_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_LGMRESGetPrecond ( void *lgmres_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_LGMRESSetPrintLevel ( void *lgmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_LGMRESGetPrintLevel ( void *lgmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_LGMRESSetLogging ( void *lgmres_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_LGMRESGetLogging ( void *lgmres_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_LGMRESGetNumIterations ( void *lgmres_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_LGMRESGetConverged ( void *lgmres_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata,
                                                     NALU_HYPRE_Real *relative_residual_norm );

/* NALU_HYPRE_bicgstab.c */
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                                NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                                NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                     NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_BiCGSTABGetResidual ( NALU_HYPRE_Solver solver, void *residual );

/* NALU_HYPRE_cgnr.c */
NALU_HYPRE_Int NALU_HYPRE_CGNRDestroy ( NALU_HYPRE_Solver solver );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_CGNRSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                 NALU_HYPRE_PtrToSolverFcn precondT, NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_CGNRGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_CGNRSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging );
NALU_HYPRE_Int NALU_HYPRE_CGNRGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_CGNRGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );

/* NALU_HYPRE_gmres.c */
NALU_HYPRE_Int NALU_HYPRE_GMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_GMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetSkipRealResidualCheck ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int skip_real_r_check );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetSkipRealResidualCheck ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *skip_real_r_check );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                  NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_GMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetConverged ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_GMRESGetResidual ( NALU_HYPRE_Solver solver, void *residual );

/* NALU_HYPRE_cogmres.c */
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                               NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                               NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetUnroll ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int unroll );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetUnroll ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *unroll );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetCGS ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int cgs );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetCGS ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cgs );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetSkipRealResidualCheck ( NALU_HYPRE_Solver solver,
                                                  NALU_HYPRE_Int skip_real_r_check );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetSkipRealResidualCheck ( NALU_HYPRE_Solver solver,
                                                  NALU_HYPRE_Int *skip_real_r_check );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                    NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_COGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetConverged ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_COGMRESGetResidual ( NALU_HYPRE_Solver solver, void *residual );

/* NALU_HYPRE_flexgmres.c */
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                                 NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b,
                                 NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                      NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetConverged ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetResidual ( NALU_HYPRE_Solver solver, void *residual );
NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetModifyPC ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int (*modify_pc )(NALU_HYPRE_Solver,
                                                                                    NALU_HYPRE_Int, NALU_HYPRE_Real ));

/* NALU_HYPRE_lgmres.c */
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int k_dim );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetKDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *k_dim );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetAugDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int aug_dim );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetAugDim ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *aug_dim );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int min_iter );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetMinIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *min_iter );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                   NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_LGMRESSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetConverged ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_LGMRESGetResidual ( NALU_HYPRE_Solver solver, void *residual );

/* NALU_HYPRE_pcg.c */
NALU_HYPRE_Int NALU_HYPRE_PCGSetup ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_PCGSolve ( NALU_HYPRE_Solver solver, NALU_HYPRE_Matrix A, NALU_HYPRE_Vector b, NALU_HYPRE_Vector x );
NALU_HYPRE_Int NALU_HYPRE_PCGSetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol );
NALU_HYPRE_Int NALU_HYPRE_PCGGetTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int NALU_HYPRE_PCGSetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int NALU_HYPRE_PCGGetAbsoluteTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int NALU_HYPRE_PCGSetAbsoluteTolFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real abstolf );
NALU_HYPRE_Int NALU_HYPRE_PCGGetAbsoluteTolFactor ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *abstolf );
NALU_HYPRE_Int NALU_HYPRE_PCGSetResidualTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real rtol );
NALU_HYPRE_Int NALU_HYPRE_PCGGetResidualTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *rtol );
NALU_HYPRE_Int NALU_HYPRE_PCGSetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int NALU_HYPRE_PCGGetConvergenceFactorTol ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int NALU_HYPRE_PCGSetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int NALU_HYPRE_PCGGetMaxIter ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int NALU_HYPRE_PCGSetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int NALU_HYPRE_PCGGetStopCrit ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int NALU_HYPRE_PCGSetTwoNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int NALU_HYPRE_PCGGetTwoNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *two_norm );
NALU_HYPRE_Int NALU_HYPRE_PCGSetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int NALU_HYPRE_PCGGetRelChange ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int NALU_HYPRE_PCGSetRecomputeResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int recompute_residual );
NALU_HYPRE_Int NALU_HYPRE_PCGGetRecomputeResidual ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *recompute_residual );
NALU_HYPRE_Int NALU_HYPRE_PCGSetRecomputeResidualP ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int recompute_residual_p );
NALU_HYPRE_Int NALU_HYPRE_PCGGetRecomputeResidualP ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *recompute_residual_p );
NALU_HYPRE_Int NALU_HYPRE_PCGSetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_PtrToSolverFcn precond,
                                NALU_HYPRE_PtrToSolverFcn precond_setup, NALU_HYPRE_Solver precond_solver );
NALU_HYPRE_Int NALU_HYPRE_PCGGetPrecond ( NALU_HYPRE_Solver solver, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int NALU_HYPRE_PCGSetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_PCGGetLogging ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_PCGSetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level );
NALU_HYPRE_Int NALU_HYPRE_PCGGetPrintLevel ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *level );
NALU_HYPRE_Int NALU_HYPRE_PCGGetNumIterations ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int NALU_HYPRE_PCGGetConverged ( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int NALU_HYPRE_PCGGetFinalRelativeResidualNorm ( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *norm );
NALU_HYPRE_Int NALU_HYPRE_PCGGetResidual ( NALU_HYPRE_Solver solver, void *residual );

/* pcg.c */
void *nalu_hypre_PCGCreate ( nalu_hypre_PCGFunctions *pcg_functions );
NALU_HYPRE_Int nalu_hypre_PCGDestroy ( void *pcg_vdata );
NALU_HYPRE_Int nalu_hypre_PCGGetResidual ( void *pcg_vdata, void **residual );
NALU_HYPRE_Int nalu_hypre_PCGSetup ( void *pcg_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_PCGSolve ( void *pcg_vdata, void *A, void *b, void *x );
NALU_HYPRE_Int nalu_hypre_PCGSetTol ( void *pcg_vdata, NALU_HYPRE_Real tol );
NALU_HYPRE_Int nalu_hypre_PCGGetTol ( void *pcg_vdata, NALU_HYPRE_Real *tol );
NALU_HYPRE_Int nalu_hypre_PCGSetAbsoluteTol ( void *pcg_vdata, NALU_HYPRE_Real a_tol );
NALU_HYPRE_Int nalu_hypre_PCGGetAbsoluteTol ( void *pcg_vdata, NALU_HYPRE_Real *a_tol );
NALU_HYPRE_Int nalu_hypre_PCGSetAbsoluteTolFactor ( void *pcg_vdata, NALU_HYPRE_Real atolf );
NALU_HYPRE_Int nalu_hypre_PCGGetAbsoluteTolFactor ( void *pcg_vdata, NALU_HYPRE_Real *atolf );
NALU_HYPRE_Int nalu_hypre_PCGSetResidualTol ( void *pcg_vdata, NALU_HYPRE_Real rtol );
NALU_HYPRE_Int nalu_hypre_PCGGetResidualTol ( void *pcg_vdata, NALU_HYPRE_Real *rtol );
NALU_HYPRE_Int nalu_hypre_PCGSetConvergenceFactorTol ( void *pcg_vdata, NALU_HYPRE_Real cf_tol );
NALU_HYPRE_Int nalu_hypre_PCGGetConvergenceFactorTol ( void *pcg_vdata, NALU_HYPRE_Real *cf_tol );
NALU_HYPRE_Int nalu_hypre_PCGSetMaxIter ( void *pcg_vdata, NALU_HYPRE_Int max_iter );
NALU_HYPRE_Int nalu_hypre_PCGGetMaxIter ( void *pcg_vdata, NALU_HYPRE_Int *max_iter );
NALU_HYPRE_Int nalu_hypre_PCGSetTwoNorm ( void *pcg_vdata, NALU_HYPRE_Int two_norm );
NALU_HYPRE_Int nalu_hypre_PCGGetTwoNorm ( void *pcg_vdata, NALU_HYPRE_Int *two_norm );
NALU_HYPRE_Int nalu_hypre_PCGSetRelChange ( void *pcg_vdata, NALU_HYPRE_Int rel_change );
NALU_HYPRE_Int nalu_hypre_PCGGetRelChange ( void *pcg_vdata, NALU_HYPRE_Int *rel_change );
NALU_HYPRE_Int nalu_hypre_PCGSetRecomputeResidual ( void *pcg_vdata, NALU_HYPRE_Int recompute_residual );
NALU_HYPRE_Int nalu_hypre_PCGGetRecomputeResidual ( void *pcg_vdata, NALU_HYPRE_Int *recompute_residual );
NALU_HYPRE_Int nalu_hypre_PCGSetRecomputeResidualP ( void *pcg_vdata, NALU_HYPRE_Int recompute_residual_p );
NALU_HYPRE_Int nalu_hypre_PCGGetRecomputeResidualP ( void *pcg_vdata, NALU_HYPRE_Int *recompute_residual_p );
NALU_HYPRE_Int nalu_hypre_PCGSetStopCrit ( void *pcg_vdata, NALU_HYPRE_Int stop_crit );
NALU_HYPRE_Int nalu_hypre_PCGGetStopCrit ( void *pcg_vdata, NALU_HYPRE_Int *stop_crit );
NALU_HYPRE_Int nalu_hypre_PCGGetPrecond ( void *pcg_vdata, NALU_HYPRE_Solver *precond_data_ptr );
NALU_HYPRE_Int nalu_hypre_PCGSetPrecond ( void *pcg_vdata, NALU_HYPRE_Int (*precond )(void*, void*, void*, void*),
                                NALU_HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
NALU_HYPRE_Int nalu_hypre_PCGSetPrintLevel ( void *pcg_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_PCGGetPrintLevel ( void *pcg_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_PCGSetLogging ( void *pcg_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_PCGGetLogging ( void *pcg_vdata, NALU_HYPRE_Int *level );
NALU_HYPRE_Int nalu_hypre_PCGSetHybrid ( void *pcg_vdata, NALU_HYPRE_Int level );
NALU_HYPRE_Int nalu_hypre_PCGGetNumIterations ( void *pcg_vdata, NALU_HYPRE_Int *num_iterations );
NALU_HYPRE_Int nalu_hypre_PCGGetConverged ( void *pcg_vdata, NALU_HYPRE_Int *converged );
NALU_HYPRE_Int nalu_hypre_PCGPrintLogging ( void *pcg_vdata, NALU_HYPRE_Int myid );
NALU_HYPRE_Int nalu_hypre_PCGGetFinalRelativeResidualNorm ( void *pcg_vdata,
                                                  NALU_HYPRE_Real *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif

