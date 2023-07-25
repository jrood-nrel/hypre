/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovCAlloc
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParKrylovCAlloc( size_t               count,
                       size_t               elt_size,
                       NALU_HYPRE_MemoryLocation location )
{
   return ( (void*) nalu_hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovFree
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovFree( void *ptr )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_TFree( ptr, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParKrylovCreateVector( void *vvector )
{
   nalu_hypre_ParVector *vector = (nalu_hypre_ParVector *) vvector;
   nalu_hypre_ParVector *new_vector;

   new_vector = nalu_hypre_ParMultiVectorCreate( nalu_hypre_ParVectorComm(vector),
                                            nalu_hypre_ParVectorGlobalSize(vector),
                                            nalu_hypre_ParVectorPartitioning(vector),
                                            nalu_hypre_ParVectorNumVectors(vector) );

   nalu_hypre_ParVectorInitialize_v2(new_vector, nalu_hypre_ParVectorMemoryLocation(vector));

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovCreateVectorArray
 * Note: one array will be allocated for all vectors, with vector 0 owning
 * the data, vector i will have data[i*size] assigned, not owning data
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParKrylovCreateVectorArray(NALU_HYPRE_Int n, void *vvector )
{
   nalu_hypre_ParVector *vector = (nalu_hypre_ParVector *) vvector;

   nalu_hypre_ParVector **new_vector;
   NALU_HYPRE_Int i, size, num_vectors;
   NALU_HYPRE_Complex *array_data;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParVectorMemoryLocation(vector);

   size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(vector));
   num_vectors = nalu_hypre_VectorNumVectors(nalu_hypre_ParVectorLocalVector(vector));
   array_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, (n * size * num_vectors), memory_location);
   new_vector = nalu_hypre_CTAlloc(nalu_hypre_ParVector*, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      new_vector[i] = nalu_hypre_ParMultiVectorCreate( nalu_hypre_ParVectorComm(vector),
                                                  nalu_hypre_ParVectorGlobalSize(vector),
                                                  nalu_hypre_ParVectorPartitioning(vector),
                                                  nalu_hypre_ParVectorNumVectors(vector) );
      nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(new_vector[i])) = &array_data[i * size * num_vectors];
      nalu_hypre_ParVectorInitialize_v2(new_vector[i], memory_location);
      if (i)
      {
         nalu_hypre_VectorOwnsData(nalu_hypre_ParVectorLocalVector(new_vector[i])) = 0;
      }
      nalu_hypre_ParVectorActualLocalSize(new_vector[i]) = size;
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovDestroyVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovDestroyVector( void *vvector )
{
   nalu_hypre_ParVector *vector = (nalu_hypre_ParVector *) vvector;

   return ( nalu_hypre_ParVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_ParKrylovMatvecCreate( void   *A,
                             void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovMatvec( void   *matvec_data,
                       NALU_HYPRE_Complex  alpha,
                       void   *A,
                       void   *x,
                       NALU_HYPRE_Complex  beta,
                       void   *y           )
{
   return ( nalu_hypre_ParCSRMatrixMatvec ( alpha,
                                       (nalu_hypre_ParCSRMatrix *) A,
                                       (nalu_hypre_ParVector *) x,
                                       beta,
                                       (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMatvecT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovMatvecT(void   *matvec_data,
                       NALU_HYPRE_Complex  alpha,
                       void   *A,
                       void   *x,
                       NALU_HYPRE_Complex  beta,
                       void   *y           )
{
   return ( nalu_hypre_ParCSRMatrixMatvecT( alpha,
                                       (nalu_hypre_ParCSRMatrix *) A,
                                       (nalu_hypre_ParVector *) x,
                                       beta,
                                       (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_ParKrylovInnerProd( void *x,
                          void *y )
{
   return ( nalu_hypre_ParVectorInnerProd( (nalu_hypre_ParVector *) x,
                                      (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMassInnerProd
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParKrylovMassInnerProd( void *x,
                              void **y, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void  * result )
{
   return ( nalu_hypre_ParVectorMassInnerProd( (nalu_hypre_ParVector *) x, (nalu_hypre_ParVector **) y, k, unroll,
                                          (NALU_HYPRE_Real*)result ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMassDotpTwo
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParKrylovMassDotpTwo( void *x, void *y,
                            void **z, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void  *result_x, void *result_y )
{
   return ( nalu_hypre_ParVectorMassDotpTwo( (nalu_hypre_ParVector *) x, (nalu_hypre_ParVector *) y,
                                        (nalu_hypre_ParVector **) z, k,
                                        unroll, (NALU_HYPRE_Real *)result_x, (NALU_HYPRE_Real *)result_y ) );
}



/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovCopyVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovCopyVector( void *x,
                           void *y )
{
   return ( nalu_hypre_ParVectorCopy( (nalu_hypre_ParVector *) x,
                                 (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovClearVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovClearVector( void *x )
{
   return ( nalu_hypre_ParVectorSetZeros( (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovScaleVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovScaleVector( NALU_HYPRE_Complex  alpha,
                            void   *x     )
{
   return ( nalu_hypre_ParVectorScale( alpha, (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovAxpy( NALU_HYPRE_Complex alpha,
                     void   *x,
                     void   *y )
{
   return ( nalu_hypre_ParVectorAxpy( alpha, (nalu_hypre_ParVector *) x,
                                 (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovMassAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovMassAxpy( NALU_HYPRE_Complex *alpha,
                         void   **x,
                         void   *y,
                         NALU_HYPRE_Int k,
                         NALU_HYPRE_Int unroll )
{
   return ( nalu_hypre_ParVectorMassAxpy( alpha, (nalu_hypre_ParVector **) x,
                                     (nalu_hypre_ParVector *) y, k, unroll));
}



/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovCommInfo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovCommInfo( void   *A, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs)
{
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm ( (nalu_hypre_ParCSRMatrix *) A);
   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovIdentitySetup( void *vdata,
                              void *A,
                              void *b,
                              void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParKrylovIdentity
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParKrylovIdentity( void *vdata,
                         void *A,
                         void *b,
                         void *x     )

{
   return ( nalu_hypre_ParKrylovCopyVector( b, x ) );
}
