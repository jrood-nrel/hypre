/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

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
   if (array)
   {
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_IntArrayMemoryLocation(array);

      nalu_hypre_TFree(nalu_hypre_IntArrayData(array), memory_location);

      nalu_hypre_TFree(array, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInitialize_v2( nalu_hypre_IntArray *array, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int  size = nalu_hypre_IntArraySize(array);

   nalu_hypre_IntArrayMemoryLocation(array) = memory_location;

   /* Caveat: for pre-existing data, the memory location must be guaranteed
    * to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered
    * when being used, and freed */
   if (!nalu_hypre_IntArrayData(array))
   {
      nalu_hypre_IntArrayData(array) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, size, memory_location);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInitialize( nalu_hypre_IntArray *array )
{
   nalu_hypre_IntArrayInitialize_v2( array, nalu_hypre_IntArrayMemoryLocation(array) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCopy
 *
 * Copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayCopy( nalu_hypre_IntArray *x,
                    nalu_hypre_IntArray *y )
{
   size_t size = nalu_hypre_min( nalu_hypre_IntArraySize(x), nalu_hypre_IntArraySize(y) );

   nalu_hypre_TMemcpy( nalu_hypre_IntArrayData(y),
                  nalu_hypre_IntArrayData(x),
                  NALU_HYPRE_Int,
                  size,
                  nalu_hypre_IntArrayMemoryLocation(y),
                  nalu_hypre_IntArrayMemoryLocation(x) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCloneDeep_v2
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

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

nalu_hypre_IntArray *
nalu_hypre_IntArrayCloneDeep( nalu_hypre_IntArray *x )
{
   return nalu_hypre_IntArrayCloneDeep_v2(x, nalu_hypre_IntArrayMemoryLocation(x));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayMigrate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayMigrate( nalu_hypre_IntArray      *v,
                       NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int            size                = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int           *v_data              = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_MemoryLocation old_memory_location = nalu_hypre_IntArrayMemoryLocation(v);

   NALU_HYPRE_Int           *w_data;

   /* Update v's memory location */
   nalu_hypre_IntArrayMemoryLocation(v) = memory_location;

   if ( nalu_hypre_GetActualMemLocation(memory_location) !=
        nalu_hypre_GetActualMemLocation(old_memory_location) )
   {
      w_data = nalu_hypre_TAlloc(NALU_HYPRE_Int, size, memory_location);
      nalu_hypre_TMemcpy(w_data, v_data, NALU_HYPRE_Int, size,
                    memory_location, old_memory_location);
      nalu_hypre_TFree(v_data, old_memory_location);
      nalu_hypre_IntArrayData(v) = w_data;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayPrint( MPI_Comm        comm,
                     nalu_hypre_IntArray *array,
                     const char     *filename )
{
   NALU_HYPRE_Int             size            = nalu_hypre_IntArraySize(array);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_IntArrayMemoryLocation(array);

   nalu_hypre_IntArray       *h_array;
   NALU_HYPRE_Int            *data;

   FILE                 *file;
   NALU_HYPRE_Int             i, myid;
   char                  new_filename[1024];

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /* Move data to host if needed*/
   h_array = (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE) ?
             nalu_hypre_IntArrayCloneDeep_v2(array, NALU_HYPRE_MEMORY_HOST) : array;
   data = nalu_hypre_IntArrayData(h_array);

   /* Open file */
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file\n");
      return nalu_hypre_error_flag;
   }

   /* Print to file */
   nalu_hypre_fprintf(file, "%d\n", size);
   for (i = 0; i < size; i++)
   {
      nalu_hypre_fprintf(file, "%d\n", data[i]);
   }
   fclose(file);

   /* Free memory */
   if (h_array != array)
   {
      nalu_hypre_IntArrayDestroy(h_array);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayRead( MPI_Comm         comm,
                    const char      *filename,
                    nalu_hypre_IntArray **array_ptr )
{
   nalu_hypre_IntArray       *array;
   NALU_HYPRE_Int             size;
   FILE                 *file;
   NALU_HYPRE_Int             i, myid;
   char                  new_filename[1024];

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /* Open file */
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open input file\n");
      return nalu_hypre_error_flag;
   }

   /* Read array size from file */
   nalu_hypre_fscanf(file, "%d\n", &size);

   /* Create IntArray on the host */
   array = nalu_hypre_IntArrayCreate(size);
   nalu_hypre_IntArrayInitialize_v2(array, NALU_HYPRE_MEMORY_HOST);

   /* Read array values from file */
   for (i = 0; i < size; i++)
   {
      nalu_hypre_fscanf(file, "%d\n", &nalu_hypre_IntArrayData(array)[i]);
   }
   fclose(file);

   /* Migrate to final memory location */
   nalu_hypre_IntArrayMigrate(array, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));

   /* Set output pointer */
   *array_ptr = array;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArraySetConstantValuesHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArraySetConstantValuesHost( nalu_hypre_IntArray *v,
                                     NALU_HYPRE_Int       value )
{
   NALU_HYPRE_Int *array_data = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  size       = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int  i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArraySetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArraySetConstantValues( nalu_hypre_IntArray *v,
                                 NALU_HYPRE_Int       value )
{
   if (nalu_hypre_IntArraySize(v) <= 0)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_IntArrayMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IntArraySetConstantValuesDevice(v, value);
   }
   else
#endif
   {
      nalu_hypre_IntArraySetConstantValuesHost(v, value);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCountHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayCountHost( nalu_hypre_IntArray *v,
                         NALU_HYPRE_Int       value,
                         NALU_HYPRE_Int      *num_values_ptr )
{
   NALU_HYPRE_Int  *array_data  = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int   size        = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int   num_values  = 0;
   NALU_HYPRE_Int   i;

#if !defined(_MSC_VER) && defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) reduction(+:num_values) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      num_values += (array_data[i] == value) ? 1 : 0;
   }

   *num_values_ptr = num_values;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayCount
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayCount( nalu_hypre_IntArray *v,
                     NALU_HYPRE_Int       value,
                     NALU_HYPRE_Int      *num_values_ptr )
{
   if (nalu_hypre_IntArraySize(v) <= 0)
   {
      *num_values_ptr = 0;
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_IntArrayMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IntArrayCountDevice(v, value, num_values_ptr);
   }
   else
#endif
   {
      nalu_hypre_IntArrayCountHost(v, value, num_values_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInverseMappingHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInverseMappingHost( nalu_hypre_IntArray  *v,
                                  nalu_hypre_IntArray  *w )
{
   NALU_HYPRE_Int   size    = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int  *v_data  = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int  *w_data  = nalu_hypre_IntArrayData(w);

   NALU_HYPRE_Int   i;

#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < size; i++)
   {
      w_data[v_data[i]] = i;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayInverseMapping
 *
 * Compute the reverse mapping (w) given an input array (v)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayInverseMapping( nalu_hypre_IntArray  *v,
                              nalu_hypre_IntArray **w_ptr )
{
   NALU_HYPRE_Int             size = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_IntArrayMemoryLocation(v);
   nalu_hypre_IntArray       *w;

   /* Create and initialize output array */
   w = nalu_hypre_IntArrayCreate(size);
   nalu_hypre_IntArrayInitialize_v2(w, memory_location);

   /* Exit if array has no elements */
   if (nalu_hypre_IntArraySize(w) <= 0)
   {
      *w_ptr = w;

      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(memory_location);

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IntArrayInverseMappingDevice(v, w);
   }
   else
#endif
   {
      nalu_hypre_IntArrayInverseMappingHost(v, w);
   }

   /* Set output pointer */
   *w_ptr = w;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IntArrayNegate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntArrayNegate( nalu_hypre_IntArray *v )
{
   NALU_HYPRE_Int  *array_data  = nalu_hypre_IntArrayData(v);
   NALU_HYPRE_Int   size        = nalu_hypre_IntArraySize(v);
   NALU_HYPRE_Int   i;

   if (size <= 0)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(nalu_hypre_IntArrayMemoryLocation(v));

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IntArrayNegateDevice(v);
   }
   else
#endif
   {
#if defined(NALU_HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         array_data[i] = - array_data[i];
      }
   }

   return nalu_hypre_error_flag;
}
