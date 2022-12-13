/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCAlloc
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCAlloc( size_t               count,
                       size_t               elt_size,
                       NALU_HYPRE_MemoryLocation location )
{
   return ( (void*) hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovFree
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovFree( void *ptr )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_TFree( ptr, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVector
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVector( void *vvector )
{
   hypre_ParVector *vector = (hypre_ParVector *) vvector;
   hypre_ParVector *new_vector;

   new_vector = hypre_ParMultiVectorCreate( hypre_ParVectorComm(vector),
                                            hypre_ParVectorGlobalSize(vector),
                                            hypre_ParVectorPartitioning(vector),
                                            hypre_ParVectorNumVectors(vector) );

   hypre_ParVectorInitialize_v2(new_vector, hypre_ParVectorMemoryLocation(vector));

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVectorArray
 * Note: one array will be allocated for all vectors, with vector 0 owning
 * the data, vector i will have data[i*size] assigned, not owning data
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovCreateVectorArray(NALU_HYPRE_Int n, void *vvector )
{
   hypre_ParVector *vector = (hypre_ParVector *) vvector;

   hypre_ParVector **new_vector;
   NALU_HYPRE_Int i, size, num_vectors;
   NALU_HYPRE_Complex *array_data;

   NALU_HYPRE_MemoryLocation memory_location = hypre_ParVectorMemoryLocation(vector);

   size = hypre_VectorSize(hypre_ParVectorLocalVector(vector));
   num_vectors = hypre_VectorNumVectors(hypre_ParVectorLocalVector(vector));
   array_data = hypre_CTAlloc(NALU_HYPRE_Complex, (n * size * num_vectors), memory_location);
   new_vector = hypre_CTAlloc(hypre_ParVector*, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      new_vector[i] = hypre_ParMultiVectorCreate( hypre_ParVectorComm(vector),
                                                  hypre_ParVectorGlobalSize(vector),
                                                  hypre_ParVectorPartitioning(vector),
                                                  hypre_ParVectorNumVectors(vector) );
      hypre_VectorData(hypre_ParVectorLocalVector(new_vector[i])) = &array_data[i * size * num_vectors];
      hypre_ParVectorInitialize_v2(new_vector[i], memory_location);
      if (i)
      {
         hypre_VectorOwnsData(hypre_ParVectorLocalVector(new_vector[i])) = 0;
      }
      hypre_ParVectorActualLocalSize(new_vector[i]) = size;
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovDestroyVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovDestroyVector( void *vvector )
{
   hypre_ParVector *vector = (hypre_ParVector *) vvector;

   return ( hypre_ParVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParKrylovMatvecCreate( void   *A,
                             void   *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovMatvec( void   *matvec_data,
                       NALU_HYPRE_Complex  alpha,
                       void   *A,
                       void   *x,
                       NALU_HYPRE_Complex  beta,
                       void   *y           )
{
   return ( hypre_ParCSRMatrixMatvec ( alpha,
                                       (hypre_ParCSRMatrix *) A,
                                       (hypre_ParVector *) x,
                                       beta,
                                       (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovMatvecT(void   *matvec_data,
                       NALU_HYPRE_Complex  alpha,
                       void   *A,
                       void   *x,
                       NALU_HYPRE_Complex  beta,
                       void   *y           )
{
   return ( hypre_ParCSRMatrixMatvecT( alpha,
                                       (hypre_ParCSRMatrix *) A,
                                       (hypre_ParVector *) x,
                                       beta,
                                       (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovMatvecDestroy( void *matvec_data )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
hypre_ParKrylovInnerProd( void *x,
                          void *y )
{
   return ( hypre_ParVectorInnerProd( (hypre_ParVector *) x,
                                      (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProd
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_ParKrylovMassInnerProd( void *x,
                              void **y, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void  * result )
{
   return ( hypre_ParVectorMassInnerProd( (hypre_ParVector *) x, (hypre_ParVector **) y, k, unroll,
                                          (NALU_HYPRE_Real*)result ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassDotpTwo
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_ParKrylovMassDotpTwo( void *x, void *y,
                            void **z, NALU_HYPRE_Int k, NALU_HYPRE_Int unroll, void  *result_x, void *result_y )
{
   return ( hypre_ParVectorMassDotpTwo( (hypre_ParVector *) x, (hypre_ParVector *) y,
                                        (hypre_ParVector **) z, k,
                                        unroll, (NALU_HYPRE_Real *)result_x, (NALU_HYPRE_Real *)result_y ) );
}



/*--------------------------------------------------------------------------
 * hypre_ParKrylovCopyVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovCopyVector( void *x,
                           void *y )
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovClearVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovClearVector( void *x )
{
   return ( hypre_ParVectorSetZeros( (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovScaleVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovScaleVector( NALU_HYPRE_Complex  alpha,
                            void   *x     )
{
   return ( hypre_ParVectorScale( alpha, (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovAxpy( NALU_HYPRE_Complex alpha,
                     void   *x,
                     void   *y )
{
   return ( hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovMassAxpy( NALU_HYPRE_Complex *alpha,
                         void   **x,
                         void   *y,
                         NALU_HYPRE_Int k,
                         NALU_HYPRE_Int unroll )
{
   return ( hypre_ParVectorMassAxpy( alpha, (hypre_ParVector **) x,
                                     (hypre_ParVector *) y, k, unroll));
}



/*--------------------------------------------------------------------------
 * hypre_ParKrylovCommInfo
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovCommInfo( void   *A, NALU_HYPRE_Int *my_id, NALU_HYPRE_Int *num_procs)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm ( (hypre_ParCSRMatrix *) A);
   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovIdentitySetup( void *vdata,
                              void *A,
                              void *b,
                              void *x     )

{
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentity
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_ParKrylovIdentity( void *vdata,
                         void *A,
                         void *b,
                         void *x     )

{
   return ( hypre_ParKrylovCopyVector( b, x ) );
}
