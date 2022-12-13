/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

/******************************************************************************
 *
 * Routines for hypre_IntArray struct for holding an array of integers
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_IntArrayCreate
 *--------------------------------------------------------------------------*/

hypre_IntArray *
hypre_IntArrayCreate( NALU_HYPRE_Int size )
{
   hypre_IntArray  *array;

   array = hypre_CTAlloc(hypre_IntArray, 1, NALU_HYPRE_MEMORY_HOST);

   hypre_IntArrayData(array) = NULL;
   hypre_IntArraySize(array) = size;

   hypre_IntArrayMemoryLocation(array) = hypre_HandleMemoryLocation(hypre_handle());

   return array;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_IntArrayDestroy( hypre_IntArray *array )
{
   NALU_HYPRE_Int ierr = 0;

   if (array)
   {
      NALU_HYPRE_MemoryLocation memory_location = hypre_IntArrayMemoryLocation(array);

      hypre_TFree(hypre_IntArrayData(array), memory_location);

      hypre_TFree(array, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_IntArrayInitialize_v2( hypre_IntArray *array, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int  size = hypre_IntArraySize(array);
   NALU_HYPRE_Int  ierr = 0;

   hypre_IntArrayMemoryLocation(array) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if ( !hypre_IntArrayData(array) )
   {
      hypre_IntArrayData(array) = hypre_CTAlloc(NALU_HYPRE_Int, size, memory_location);
   }

   return ierr;
}

NALU_HYPRE_Int
hypre_IntArrayInitialize( hypre_IntArray *array )
{
   NALU_HYPRE_Int ierr;

   ierr = hypre_IntArrayInitialize_v2( array, hypre_IntArrayMemoryLocation(array) );

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_IntArrayCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_IntArrayCopy( hypre_IntArray *x,
                    hypre_IntArray *y )
{
   NALU_HYPRE_Int ierr = 0;

   size_t size = hypre_min( hypre_IntArraySize(x), hypre_IntArraySize(y) );

   hypre_TMemcpy( hypre_IntArrayData(y),
                  hypre_IntArrayData(x),
                  NALU_HYPRE_Int,
                  size,
                  hypre_IntArrayMemoryLocation(y),
                  hypre_IntArrayMemoryLocation(x) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

hypre_IntArray *
hypre_IntArrayCloneDeep_v2( hypre_IntArray *x, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int    size = hypre_IntArraySize(x);

   hypre_IntArray *y = hypre_IntArrayCreate( size );

   hypre_IntArrayInitialize_v2(y, memory_location);
   hypre_IntArrayCopy( x, y );

   return y;
}

hypre_IntArray *
hypre_IntArrayCloneDeep( hypre_IntArray *x )
{
   return hypre_IntArrayCloneDeep_v2(x, hypre_IntArrayMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * hypre_IntArraySetConstantValues
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
NALU_HYPRE_Int
hypre_IntArraySetConstantValuesDevice( hypre_IntArray *v,
                                       NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = hypre_IntArraySize(v);

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

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}
#endif

NALU_HYPRE_Int
hypre_IntArraySetConstantValues( hypre_IntArray *v,
                                 NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = hypre_IntArraySize(v);

   if (size <= 0)
   {
      return hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_IntArrayMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      hypre_IntArraySetConstantValuesDevice(v, value);
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

   return hypre_error_flag;
}

