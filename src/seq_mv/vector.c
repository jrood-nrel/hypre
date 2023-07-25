/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_Vector class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector *
nalu_hypre_SeqVectorCreate( NALU_HYPRE_Int size )
{
   nalu_hypre_Vector  *vector;

   vector = nalu_hypre_CTAlloc(nalu_hypre_Vector, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_VectorData(vector) = NULL;
   nalu_hypre_VectorSize(vector) = size;

   nalu_hypre_VectorNumVectors(vector) = 1;
   nalu_hypre_VectorMultiVecStorageMethod(vector) = 0;

   /* set defaults */
   nalu_hypre_VectorOwnsData(vector) = 1;

   nalu_hypre_VectorMemoryLocation(vector) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector *
nalu_hypre_SeqMultiVectorCreate( NALU_HYPRE_Int size, NALU_HYPRE_Int num_vectors )
{
   nalu_hypre_Vector *vector = nalu_hypre_SeqVectorCreate(size);
   nalu_hypre_VectorNumVectors(vector) = num_vectors;

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorDestroy( nalu_hypre_Vector *vector )
{
   if (vector)
   {
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_VectorMemoryLocation(vector);

      if (nalu_hypre_VectorOwnsData(vector))
      {
         nalu_hypre_TFree(nalu_hypre_VectorData(vector), memory_location);
      }

      nalu_hypre_TFree(vector, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorInitialize_v2
 *
 * Initialize a vector at a given memory location
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorInitialize_v2( nalu_hypre_Vector *vector, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int  size = nalu_hypre_VectorSize(vector);
   NALU_HYPRE_Int  num_vectors = nalu_hypre_VectorNumVectors(vector);
   NALU_HYPRE_Int  multivec_storage_method = nalu_hypre_VectorMultiVecStorageMethod(vector);

   nalu_hypre_VectorMemoryLocation(vector) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if (!nalu_hypre_VectorData(vector))
   {
      nalu_hypre_VectorData(vector) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_vectors * size, memory_location);
   }

   if (multivec_storage_method == 0)
   {
      nalu_hypre_VectorVectorStride(vector) = size;
      nalu_hypre_VectorIndexStride(vector)  = 1;
   }
   else if (multivec_storage_method == 1)
   {
      nalu_hypre_VectorVectorStride(vector) = 1;
      nalu_hypre_VectorIndexStride(vector)  = num_vectors;
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Invalid multivec storage method!\n");
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorInitialize( nalu_hypre_Vector *vector )
{
   return nalu_hypre_SeqVectorInitialize_v2(vector, nalu_hypre_VectorMemoryLocation(vector));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetDataOwner( nalu_hypre_Vector *vector,
                             NALU_HYPRE_Int     owns_data   )
{
   nalu_hypre_VectorOwnsData(vector) = owns_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetSize( nalu_hypre_Vector *vector,
                        NALU_HYPRE_Int     size   )
{
   NALU_HYPRE_Int  multivec_storage_method = nalu_hypre_VectorMultiVecStorageMethod(vector);

   nalu_hypre_VectorSize(vector) = size;
   if (multivec_storage_method == 0)
   {
      nalu_hypre_VectorVectorStride(vector) = size;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorResize
 *
 * Resize a sequential vector when changing its number of components.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorResize( nalu_hypre_Vector *vector,
                       NALU_HYPRE_Int     num_vectors_in )
{
   NALU_HYPRE_Int  method        = nalu_hypre_VectorMultiVecStorageMethod(vector);
   NALU_HYPRE_Int  size          = nalu_hypre_VectorSize(vector);
   NALU_HYPRE_Int  num_vectors   = nalu_hypre_VectorNumVectors(vector);
   NALU_HYPRE_Int  total_size    = num_vectors * size;
   NALU_HYPRE_Int  total_size_in = num_vectors_in * size;

   /* Reallocate data array */
   if (total_size_in > total_size)
   {
      nalu_hypre_VectorData(vector) = nalu_hypre_TReAlloc_v2(nalu_hypre_VectorData(vector),
                                                   NALU_HYPRE_Complex,
                                                   total_size,
                                                   NALU_HYPRE_Complex,
                                                   total_size_in,
                                                   nalu_hypre_VectorMemoryLocation(vector));
   }

   /* Update vector info */
   nalu_hypre_VectorNumVectors(vector) = num_vectors_in;
   if (method == 0)
   {
      nalu_hypre_VectorVectorStride(vector) = size;
      nalu_hypre_VectorIndexStride(vector)  = 1;
   }
   else if (method == 1)
   {
      nalu_hypre_VectorVectorStride(vector) = 1;
      nalu_hypre_VectorIndexStride(vector)  = num_vectors;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorRead
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector *
nalu_hypre_SeqVectorRead( char *file_name )
{
   nalu_hypre_Vector  *vector;

   FILE    *fp;

   NALU_HYPRE_Complex *data;
   NALU_HYPRE_Int      size;

   NALU_HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   nalu_hypre_fscanf(fp, "%d", &size);

   vector = nalu_hypre_SeqVectorCreate(size);

   nalu_hypre_VectorMemoryLocation(vector) = NALU_HYPRE_MEMORY_HOST;

   nalu_hypre_SeqVectorInitialize(vector);

   data = nalu_hypre_VectorData(vector);
   for (j = 0; j < size; j++)
   {
      nalu_hypre_fscanf(fp, "%le", &data[j]);
   }

   fclose(fp);

   /* multivector code not written yet */
   nalu_hypre_assert( nalu_hypre_VectorNumVectors(vector) == 1 );

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorPrint( nalu_hypre_Vector *vector,
                      char         *file_name )
{
   FILE          *fp;

   NALU_HYPRE_Complex *data;
   NALU_HYPRE_Int      size, num_vectors, vecstride, idxstride;

   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Complex  value;

   num_vectors = nalu_hypre_VectorNumVectors(vector);
   vecstride = nalu_hypre_VectorVectorStride(vector);
   idxstride = nalu_hypre_VectorIndexStride(vector);

   /*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/

   data = nalu_hypre_VectorData(vector);
   size = nalu_hypre_VectorSize(vector);

   fp = fopen(file_name, "w");

   if ( nalu_hypre_VectorNumVectors(vector) == 1 )
   {
      nalu_hypre_fprintf(fp, "%d\n", size);
   }
   else
   {
      nalu_hypre_fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
   }

   if ( num_vectors > 1 )
   {
      for ( j = 0; j < num_vectors; ++j )
      {
         nalu_hypre_fprintf(fp, "vector %d\n", j );
         for (i = 0; i < size; i++)
         {
            value = data[ j * vecstride + i * idxstride ];
#ifdef NALU_HYPRE_COMPLEX
            nalu_hypre_fprintf(fp, "%.14e , %.14e\n",
                          nalu_hypre_creal(value), nalu_hypre_cimag(value));
#else
            nalu_hypre_fprintf(fp, "%.14e\n", value);
#endif
         }
      }
   }
   else
   {
      for (i = 0; i < size; i++)
      {
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf(fp, "%.14e , %.14e\n",
                       nalu_hypre_creal(data[i]), nalu_hypre_cimag(data[i]));
#else
         nalu_hypre_fprintf(fp, "%.14e\n", data[i]);
#endif
      }
   }

   fclose(fp);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetConstantValuesHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetConstantValuesHost( nalu_hypre_Vector *v,
                                      NALU_HYPRE_Complex value )
{
   NALU_HYPRE_Complex *vector_data = nalu_hypre_VectorData(v);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(v);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(v);
   NALU_HYPRE_Int      total_size  = size * num_vectors;
   NALU_HYPRE_Int      i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      vector_data[i] = value;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetConstantValues( nalu_hypre_Vector *v,
                                  NALU_HYPRE_Complex value )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int   num_vectors = nalu_hypre_VectorNumVectors(v);
   NALU_HYPRE_Int   size        = nalu_hypre_VectorSize(v);
   NALU_HYPRE_Int   total_size  = size * num_vectors;

   /* Trivial case */
   if (total_size <= 0)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_VectorMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_SeqVectorSetConstantValuesDevice(v, value);
   }
   else
#endif
   {
      nalu_hypre_SeqVectorSetConstantValuesHost(v, value);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSetRandomValues
 *
 * returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorSetRandomValues( nalu_hypre_Vector *v,
                                NALU_HYPRE_Int     seed )
{
   NALU_HYPRE_Complex *vector_data = nalu_hypre_VectorData(v);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(v);
   NALU_HYPRE_Int      i;

   nalu_hypre_SeedRand(seed);
   size *= nalu_hypre_VectorNumVectors(v);

   if (nalu_hypre_GetActualMemLocation(nalu_hypre_VectorMemoryLocation(v)) == nalu_hypre_MEMORY_HOST)
   {
      /* RDF: threading this loop may cause problems because of nalu_hypre_Rand() */
      for (i = 0; i < size; i++)
      {
         vector_data[i] = 2.0 * nalu_hypre_Rand() - 1.0;
      }
   }
   else
   {
      NALU_HYPRE_Complex *h_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < size; i++)
      {
         h_data[i] = 2.0 * nalu_hypre_Rand() - 1.0;
      }
      nalu_hypre_TMemcpy(vector_data, h_data, NALU_HYPRE_Complex, size, nalu_hypre_VectorMemoryLocation(v),
                    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(h_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorCopy( nalu_hypre_Vector *x,
                     nalu_hypre_Vector *y )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_GpuProfilingPushRange("SeqVectorCopy");

   size_t size = nalu_hypre_min(nalu_hypre_VectorSize(x), nalu_hypre_VectorSize(y)) * nalu_hypre_VectorNumVectors(x);

   nalu_hypre_TMemcpy( nalu_hypre_VectorData(y),
                  nalu_hypre_VectorData(x),
                  NALU_HYPRE_Complex,
                  size,
                  nalu_hypre_VectorMemoryLocation(y),
                  nalu_hypre_VectorMemoryLocation(x) );

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif
   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorCloneDeep_v2
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector*
nalu_hypre_SeqVectorCloneDeep_v2( nalu_hypre_Vector *x, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int      size          = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      num_vectors   = nalu_hypre_VectorNumVectors(x);

   nalu_hypre_Vector *y = nalu_hypre_SeqMultiVectorCreate( size, num_vectors );

   nalu_hypre_VectorMultiVecStorageMethod(y) = nalu_hypre_VectorMultiVecStorageMethod(x);
   nalu_hypre_VectorVectorStride(y) = nalu_hypre_VectorVectorStride(x);
   nalu_hypre_VectorIndexStride(y) = nalu_hypre_VectorIndexStride(x);

   nalu_hypre_SeqVectorInitialize_v2(y, memory_location);
   nalu_hypre_SeqVectorCopy( x, y );

   return y;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorCloneDeep
 *
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector*
nalu_hypre_SeqVectorCloneDeep( nalu_hypre_Vector *x )
{
   return nalu_hypre_SeqVectorCloneDeep_v2(x, nalu_hypre_VectorMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorCloneShallow
 *
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector *
nalu_hypre_SeqVectorCloneShallow( nalu_hypre_Vector *x )
{
   NALU_HYPRE_Int     size         = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int     num_vectors  = nalu_hypre_VectorNumVectors(x);
   nalu_hypre_Vector *y            = nalu_hypre_SeqMultiVectorCreate(size, num_vectors);

   nalu_hypre_VectorMultiVecStorageMethod(y) = nalu_hypre_VectorMultiVecStorageMethod(x);
   nalu_hypre_VectorVectorStride(y) = nalu_hypre_VectorVectorStride(x);
   nalu_hypre_VectorIndexStride(y) = nalu_hypre_VectorIndexStride(x);

   nalu_hypre_VectorMemoryLocation(y) = nalu_hypre_VectorMemoryLocation(x);

   nalu_hypre_VectorData(y) = nalu_hypre_VectorData(x);
   nalu_hypre_SeqVectorSetDataOwner(y, 0);
   nalu_hypre_SeqVectorInitialize(y);

   return y;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorScaleHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorScaleHost( NALU_HYPRE_Complex alpha,
                          nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(y);
   NALU_HYPRE_Int      total_size  = size * num_vectors;
   NALU_HYPRE_Int      i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      y_data[i] *= alpha;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorScale( NALU_HYPRE_Complex alpha,
                      nalu_hypre_Vector *y )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   /* special cases */
   if (alpha == 1.0)
   {
      return nalu_hypre_error_flag;
   }

   if (alpha == 0.0)
   {
      return nalu_hypre_SeqVectorSetConstantValues(y, 0.0);
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_VectorMemoryLocation(y));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_SeqVectorScaleDevice(alpha, y);
   }
   else
#endif
   {
      nalu_hypre_SeqVectorScaleHost(alpha, y);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpyHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpyHost( NALU_HYPRE_Complex alpha,
                         nalu_hypre_Vector *x,
                         nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;
   NALU_HYPRE_Int      i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      y_data[i] += alpha * x_data[i];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpy( NALU_HYPRE_Complex alpha,
                     nalu_hypre_Vector *x,
                     nalu_hypre_Vector *y )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_VectorMemoryLocation(x),
                                                      nalu_hypre_VectorMemoryLocation(y) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_SeqVectorAxpyDevice(alpha, x, y);
   }
   else
#endif
   {
      nalu_hypre_SeqVectorAxpyHost(alpha, x, y);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpyzHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpyzHost( NALU_HYPRE_Complex alpha,
                          nalu_hypre_Vector *x,
                          NALU_HYPRE_Complex beta,
                          nalu_hypre_Vector *y,
                          nalu_hypre_Vector *z )
{
   NALU_HYPRE_Complex *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Complex *z_data      = nalu_hypre_VectorData(z);

   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;
   NALU_HYPRE_Int      i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      z_data[i] = alpha * x_data[i] + beta * y_data[i];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorAxpyz
 *
 * Computes z = a*x + b*y
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorAxpyz( NALU_HYPRE_Complex alpha,
                      nalu_hypre_Vector *x,
                      NALU_HYPRE_Complex beta,
                      nalu_hypre_Vector *y,
                      nalu_hypre_Vector *z )
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_VectorMemoryLocation(x),
                                                      nalu_hypre_VectorMemoryLocation(y));
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_SeqVectorAxpyzDevice(alpha, x, beta, y, z);
   }
   else
#endif
   {
      nalu_hypre_SeqVectorAxpyzHost(alpha, x, beta, y, z);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorElmdivpyHost
 *
 * if marker != NULL: only for marker[i] == marker_val
 *
 * TODO:
 *        1) Change to nalu_hypre_SeqVectorElmdivpyMarkedHost?
 *        2) Add vecstride/idxstride variables
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorElmdivpyHost( nalu_hypre_Vector *x,
                             nalu_hypre_Vector *b,
                             nalu_hypre_Vector *y,
                             NALU_HYPRE_Int    *marker,
                             NALU_HYPRE_Int     marker_val )
{
   NALU_HYPRE_Complex   *x_data        = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex   *b_data        = nalu_hypre_VectorData(b);
   NALU_HYPRE_Complex   *y_data        = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int        num_vectors_x = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int        num_vectors_y = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int        num_vectors_b = nalu_hypre_VectorNumVectors(b);
   NALU_HYPRE_Int        size          = nalu_hypre_VectorSize(b);
   NALU_HYPRE_Int        i, j;
   NALU_HYPRE_Complex    val;

   if (num_vectors_b == 1)
   {
      if (num_vectors_x == 1 &&
          num_vectors_y == 1)
      {
         if (marker)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               if (marker[i] == marker_val)
               {
                  y_data[i] += x_data[i] / b_data[i];
               }
            }
         }
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               y_data[i] += x_data[i] / b_data[i];
            }
         } /* if (marker) */
      }
      else if (num_vectors_x == 2 &&
               num_vectors_y == 2)
      {
         if (marker)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               if (marker[i] == marker_val)
               {
                  y_data[i]        += x_data[i]        * val;
                  y_data[i + size] += x_data[i + size] * val;
               }
            }
         }
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];

               y_data[i]        += x_data[i]        * val;
               y_data[i + size] += x_data[i + size] * val;
            }
         } /* if (marker) */
      }
      else if (num_vectors_x == num_vectors_y)
      {
         if (marker)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i, j) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               if (marker[i] == marker_val)
               {
                  for (j = 0; j < num_vectors_x; j++)
                  {
                     y_data[i + size * j] += x_data[i + size * j] * val;
                  }
               }
            }
         }
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i, j) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < size; i++)
            {
               val = 1.0 / b_data[i];
               for (j = 0; j < num_vectors_x; j++)
               {
                  y_data[i + size * j] += x_data[i + size * j] * val;
               }
            }
         } /* if (marker) */
      }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Unsupported combination of num_vectors!\n");
      }
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "num_vectors_b != 1 not supported!\n");
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorElmdivpyMarked
 *
 * Computes: y[i] = y[i] + x[i] / b[i] for marker[i] = marker_val
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorElmdivpyMarked( nalu_hypre_Vector *x,
                               nalu_hypre_Vector *b,
                               nalu_hypre_Vector *y,
                               NALU_HYPRE_Int    *marker,
                               NALU_HYPRE_Int     marker_val)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   /* Sanity checks */
   if (nalu_hypre_VectorSize(y) != nalu_hypre_VectorSize(b))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "sizes of y and b do not match!\n");
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_VectorSize(x) < nalu_hypre_VectorSize(y))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "x_size is smaller than y_size!\n");
      return nalu_hypre_error_flag;
   }

   if (!nalu_hypre_VectorSize(x))
   {
      /* VPM: Do not throw an error message here since this can happen for idle processors */
      return nalu_hypre_error_flag;
   }

   if (!nalu_hypre_VectorData(x))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "x_data is not present!\n");
      return nalu_hypre_error_flag;
   }

   if (!nalu_hypre_VectorData(b))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "b_data is not present!\n");
      return nalu_hypre_error_flag;
   }

   if (!nalu_hypre_VectorData(y))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "y_data is not present!\n");
      return nalu_hypre_error_flag;
   }

   /* row-wise multivec is not supported */
   nalu_hypre_assert(nalu_hypre_VectorMultiVecStorageMethod(x) == 0);
   nalu_hypre_assert(nalu_hypre_VectorMultiVecStorageMethod(b) == 0);
   nalu_hypre_assert(nalu_hypre_VectorMultiVecStorageMethod(y) == 0);

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_VectorMemoryLocation(x),
                                                      nalu_hypre_VectorMemoryLocation(b) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_SeqVectorElmdivpyDevice(x, b, y, marker, marker_val);
   }
   else
#endif
   {
      nalu_hypre_SeqVectorElmdivpyHost(x, b, y, marker, marker_val);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorElmdivpy
 *
 * Computes: y = y + x ./ b
 *
 * Notes:
 *    1) y and b must have the same sizes
 *    2) x_size can be larger than y_size
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SeqVectorElmdivpy( nalu_hypre_Vector *x,
                         nalu_hypre_Vector *b,
                         nalu_hypre_Vector *y )
{
   return nalu_hypre_SeqVectorElmdivpyMarked(x, b, y, NULL, -1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorInnerProdHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_SeqVectorInnerProdHost( nalu_hypre_Vector *x,
                              nalu_hypre_Vector *y )
{
   NALU_HYPRE_Complex *x_data      = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data      = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int      size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int      total_size  = size * num_vectors;

   NALU_HYPRE_Real     result      = 0.0;
   NALU_HYPRE_Int      i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) reduction(+:result) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      result += nalu_hypre_conj(y_data[i]) * x_data[i];
   }

   return result;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_SeqVectorInnerProd( nalu_hypre_Vector *x,
                          nalu_hypre_Vector *y )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Real result;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_VectorMemoryLocation(x),
                                                      nalu_hypre_VectorMemoryLocation(y) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      result = nalu_hypre_SeqVectorInnerProdDevice(x, y);
   }
   else
#endif
   {
      result = nalu_hypre_SeqVectorInnerProdHost(x, y);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSumEltsHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex
nalu_hypre_SeqVectorSumEltsHost( nalu_hypre_Vector *vector )
{
   NALU_HYPRE_Complex  *data        = nalu_hypre_VectorData( vector );
   NALU_HYPRE_Int       num_vectors = nalu_hypre_VectorNumVectors(vector);
   NALU_HYPRE_Int       size        = nalu_hypre_VectorSize(vector);
   NALU_HYPRE_Int       total_size  = size * num_vectors;

   NALU_HYPRE_Complex   sum  = 0;
   NALU_HYPRE_Int       i;

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < total_size; i++)
   {
      sum += data[i];
   }

   return sum;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SeqVectorSumElts:
 *
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex
nalu_hypre_SeqVectorSumElts( nalu_hypre_Vector *v )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Complex sum;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_VectorMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      sum = nalu_hypre_SeqVectorSumEltsDevice(v);
   }
   else
#endif
   {
      sum = nalu_hypre_SeqVectorSumEltsHost(v);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return sum;
}



















#if 0
/* y[i] = max(alpha*x[i], beta*y[i]) */
NALU_HYPRE_Int
nalu_hypre_SeqVectorMax( NALU_HYPRE_Complex alpha,
                    nalu_hypre_Vector *x,
                    NALU_HYPRE_Complex beta,
                    nalu_hypre_Vector *y     )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Complex *x_data = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      size   = nalu_hypre_VectorSize(x);

   size *= nalu_hypre_VectorNumVectors(x);

   //nalu_hypre_SeqVectorPrefetch(x, NALU_HYPRE_MEMORY_DEVICE);
   //nalu_hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);

   thrust::maximum<NALU_HYPRE_Complex> mx;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_THRUST_CALL( transform,
                      thrust::make_transform_iterator(x_data,        alpha * _1),
                      thrust::make_transform_iterator(x_data + size, alpha * _1),
                      thrust::make_transform_iterator(y_data,        beta  * _1),
                      y_data,
                      mx );
#else
   NALU_HYPRE_Int i;
#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, x_data)
#elif defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      y_data[i] += nalu_hypre_max(alpha * x_data[i], beta * y_data[i]);
   }

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_BLAS1] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}
#endif
