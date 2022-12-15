/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"

/******************************************************************************
 *
 * Routines for nalu_hypre_IntArray struct for holding an array of integers
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_IntArray *
nalu_hypre_IntArrayCreate( NALU_HYPRE_Int size )
{
   nalu_hypre_IntArray  *array;

   array = nalu_hypre_CTAlloc(nalu_hypre_IntArray, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_IntArrayData(array) = NULL;
   nalu_hypre_IntArraySize(array) = size;

   nalu_hypre_IntArrayMemoryLocation(array) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return array;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayDestroy( nalu_hypre_IntArray *array )
{
   NALU_HYPRE_Int ierr = 0;

   if (array)
   {
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_IntArrayMemoryLocation(array);

      nalu_hypre_TFree(nalu_hypre_IntArrayData(array), memory_location);

      nalu_hypre_TFree(array, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInitialize_v2( nalu_hypre_IntArray *array, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int  size = nalu_hypre_IntArraySize(array);
   NALU_HYPRE_Int  ierr = 0;

   nalu_hypre_IntArrayMemoryLocation(array) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if ( !nalu_hypre_IntArrayData(array) )
   {
      nalu_hypre_IntArrayData(array) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, size, memory_location);
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_IntArrayInitialize( nalu_hypre_IntArray *array )
{
   NALU_HYPRE_Int ierr;

   ierr = nalu_hypre_IntArrayInitialize_v2( array, nalu_hypre_IntArrayMemoryLocation(array) );

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_IntArrayCopy( nalu_hypre_IntArray *x,
                    nalu_hypre_IntArray *y )
{
   NALU_HYPRE_Int ierr = 0;

   size_t size = nalu_hypre_min( nalu_hypre_IntArraySize(x), nalu_hypre_IntArraySize(y) );

   nalu_hypre_TMemcpy( nalu_hypre_IntArrayData(y),
                  nalu_hypre_IntArrayData(x),
                  NALU_HYPRE_Int,
                  size,
                  nalu_hypre_IntArrayMemoryLocation(y),
                  nalu_hypre_IntArrayMemoryLocation(x) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

nalu_hypre_IntArray *
nalu_hypre_IntArrayCloneDeep_v2( nalu_hypre_IntArray *x, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int    size = nalu_hypre_IntArraySize(x);

   nalu_hypre_IntArray *y = nalu_hypre_IntArrayCreate( size );

   nalu_hypre_IntArrayInitialize_v2(y, memory_location);
   nalu_hypre_IntArrayCopy( x, y );

   return y;
}

nalu_hypre_IntArray *
nalu_hypre_IntArrayCloneDeep( nalu_hypre_IntArray *x )
{
   return nalu_hypre_IntArrayCloneDeep_v2(x, nalu_hypre_IntArrayMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArraySetConstantValues
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int
nalu_hypre_IntArraySetConstantValuesDevice( nalu_hypre_IntArray *v,
                                       NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = nalu_hypre_IntArraySize(v);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   hypreDevice_IntFilln( array_data, size, value );

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_Int i;
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(array_data)
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }
#endif

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}
#endif

NALU_HYPRE_Int
nalu_hypre_IntArraySetConstantValues( nalu_hypre_IntArray *v,
                                 NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = nalu_hypre_IntArraySize(v);

   if (size <= 0)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_IntArrayMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IntArraySetConstantValuesDevice(v, value);
   }
   else
#endif
   {
      NALU_HYPRE_Int i;
#if defined(NALU_HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         array_data[i] = value;
      }
   }

   return nalu_hypre_error_flag;
}

