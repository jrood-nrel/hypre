/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_Vector class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

NALU_HYPRE_Int nalu_hypre_FillResponseParToVectorAll(void*, NALU_HYPRE_Int, NALU_HYPRE_Int, void*, MPI_Comm, void**,
                                           NALU_HYPRE_Int*);

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCreate
 *
 * If create is called and partitioning is NOT null, then it is assumed that it
 * is array of length 2 containing the start row of the calling processor
 * followed by the start row of the next processor - AHB 6/05
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_ParVectorCreate( MPI_Comm      comm,
                       NALU_HYPRE_BigInt  global_size,
                       NALU_HYPRE_BigInt *partitioning_in )
{
   nalu_hypre_ParVector *vector;
   NALU_HYPRE_Int        num_procs, my_id, local_size;
   NALU_HYPRE_BigInt     partitioning[2];

   if (global_size < 0)
   {
      nalu_hypre_error_in_arg(2);
      return NULL;
   }
   vector = nalu_hypre_CTAlloc(nalu_hypre_ParVector, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (!partitioning_in)
   {
      nalu_hypre_MPI_Comm_size(comm, &num_procs);
      nalu_hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, partitioning);
   }
   else
   {
      partitioning[0] = partitioning_in[0];
      partitioning[1] = partitioning_in[1];
   }
   local_size = (NALU_HYPRE_Int) (partitioning[1] - partitioning[0]);

   nalu_hypre_ParVectorAssumedPartition(vector) = NULL;

   nalu_hypre_ParVectorComm(vector)            = comm;
   nalu_hypre_ParVectorGlobalSize(vector)      = global_size;
   nalu_hypre_ParVectorPartitioning(vector)[0] = partitioning[0];
   nalu_hypre_ParVectorPartitioning(vector)[1] = partitioning[1];
   nalu_hypre_ParVectorFirstIndex(vector)      = nalu_hypre_ParVectorPartitioning(vector)[0];
   nalu_hypre_ParVectorLastIndex(vector)       = nalu_hypre_ParVectorPartitioning(vector)[1] - 1;
   nalu_hypre_ParVectorLocalVector(vector)     = nalu_hypre_SeqVectorCreate(local_size);

   /* set defaults */
   nalu_hypre_ParVectorOwnsData(vector)         = 1;
   nalu_hypre_ParVectorActualLocalSize(vector)  = 0;

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_ParMultiVectorCreate( MPI_Comm      comm,
                            NALU_HYPRE_BigInt  global_size,
                            NALU_HYPRE_BigInt *partitioning,
                            NALU_HYPRE_Int     num_vectors )
{
   /* note that global_size is the global length of a single vector */
   nalu_hypre_ParVector *vector = nalu_hypre_ParVectorCreate( comm, global_size, partitioning );
   nalu_hypre_ParVectorNumVectors(vector) = num_vectors;
   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorDestroy( nalu_hypre_ParVector *vector )
{
   if (vector)
   {
      if ( nalu_hypre_ParVectorOwnsData(vector) )
      {
         nalu_hypre_SeqVectorDestroy(nalu_hypre_ParVectorLocalVector(vector));
      }

      if (nalu_hypre_ParVectorAssumedPartition(vector))
      {
         nalu_hypre_AssumedPartitionDestroy(nalu_hypre_ParVectorAssumedPartition(vector));
      }

      nalu_hypre_TFree(vector, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInitialize_v2
 *
 * Initialize a nalu_hypre_ParVector at a given memory location
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorInitialize_v2( nalu_hypre_ParVector *vector, NALU_HYPRE_MemoryLocation memory_location )
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_SeqVectorInitialize_v2(nalu_hypre_ParVectorLocalVector(vector), memory_location);

   nalu_hypre_ParVectorActualLocalSize(vector) = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(vector));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorInitialize( nalu_hypre_ParVector *vector )
{
   return nalu_hypre_ParVectorInitialize_v2(vector, nalu_hypre_ParVectorMemoryLocation(vector));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetComponent
 *
 * Set the identifier of the active component of a nalu_hypre_ParVector for the
 * purpose of Set/AddTo/Get values functions.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetComponent( nalu_hypre_ParVector *vector,
                             NALU_HYPRE_Int        component )
{
   nalu_hypre_Vector *local_vector = nalu_hypre_ParVectorLocalVector(vector);

   nalu_hypre_VectorComponent(local_vector) = component;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetDataOwner( nalu_hypre_ParVector *vector,
                             NALU_HYPRE_Int        owns_data )
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParVectorOwnsData(vector) = owns_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetLocalSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetLocalSize( nalu_hypre_ParVector *vector,
                             NALU_HYPRE_Int        local_size )
{
   nalu_hypre_Vector *local_vector = nalu_hypre_ParVectorLocalVector(vector);

   nalu_hypre_SeqVectorSetSize(local_vector, local_size);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetNumVectors
 * call before calling nalu_hypre_ParVectorInitialize
 * probably this will do more harm than good, use nalu_hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/
#if 0
NALU_HYPRE_Int
nalu_hypre_ParVectorSetNumVectors( nalu_hypre_ParVector *vector,
                              NALU_HYPRE_Int        num_vectors )
{
   NALU_HYPRE_Int    ierr = 0;
   nalu_hypre_Vector *local_vector = nalu_hypre_ParVectorLocalVector(v);

   nalu_hypre_SeqVectorSetNumVectors( local_vector, num_vectors );

   return ierr;
}
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorResize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorResize( nalu_hypre_ParVector *vector,
                       NALU_HYPRE_Int        num_vectors )
{
   if (vector)
   {
      nalu_hypre_SeqVectorResize(nalu_hypre_ParVectorLocalVector(vector), num_vectors);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorRead
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector*
nalu_hypre_ParVectorRead( MPI_Comm    comm,
                     const char *file_name )
{
   char             new_file_name[256];
   nalu_hypre_ParVector *par_vector;
   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_BigInt     partitioning[2];
   NALU_HYPRE_BigInt     global_size;
   FILE            *fp;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   nalu_hypre_sprintf(new_file_name, "%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_name, "r");
   nalu_hypre_fscanf(fp, "%b\n", &global_size);
   nalu_hypre_fscanf(fp, "%b\n", &partitioning[0]);
   nalu_hypre_fscanf(fp, "%b\n", &partitioning[1]);
   fclose (fp);
   par_vector = nalu_hypre_CTAlloc(nalu_hypre_ParVector, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParVectorComm(par_vector) = comm;
   nalu_hypre_ParVectorGlobalSize(par_vector) = global_size;

   nalu_hypre_ParVectorFirstIndex(par_vector) = partitioning[0];
   nalu_hypre_ParVectorLastIndex(par_vector) = partitioning[1] - 1;

   nalu_hypre_ParVectorPartitioning(par_vector)[0] = partitioning[0];
   nalu_hypre_ParVectorPartitioning(par_vector)[1] = partitioning[1];

   nalu_hypre_ParVectorOwnsData(par_vector) = 1;

   nalu_hypre_sprintf(new_file_name, "%s.%d", file_name, my_id);
   nalu_hypre_ParVectorLocalVector(par_vector) = nalu_hypre_SeqVectorRead(new_file_name);

   /* multivector code not written yet */
   nalu_hypre_assert( nalu_hypre_ParVectorNumVectors(par_vector) == 1 );

   return par_vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorPrint( nalu_hypre_ParVector  *vector,
                      const char       *file_name )
{
   char          new_file_name[256];
   nalu_hypre_Vector *local_vector;
   MPI_Comm      comm;
   NALU_HYPRE_Int     my_id;
   NALU_HYPRE_BigInt *partitioning;
   NALU_HYPRE_BigInt  global_size;
   FILE         *fp;

   if (!vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   local_vector = nalu_hypre_ParVectorLocalVector(vector);
   comm = nalu_hypre_ParVectorComm(vector);
   partitioning = nalu_hypre_ParVectorPartitioning(vector);
   global_size = nalu_hypre_ParVectorGlobalSize(vector);

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_sprintf(new_file_name, "%s.%d", file_name, my_id);
   nalu_hypre_SeqVectorPrint(local_vector, new_file_name);
   nalu_hypre_sprintf(new_file_name, "%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_name, "w");
   nalu_hypre_fprintf(fp, "%b\n", global_size);
   nalu_hypre_fprintf(fp, "%b\n", partitioning[0]);
   nalu_hypre_fprintf(fp, "%b\n", partitioning[1]);

   fclose(fp);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetConstantValues( nalu_hypre_ParVector *v,
                                  NALU_HYPRE_Complex    value )
{
   nalu_hypre_Vector *v_local = nalu_hypre_ParVectorLocalVector(v);

   return nalu_hypre_SeqVectorSetConstantValues(v_local, value);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetZeros
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetZeros( nalu_hypre_ParVector *v )
{
   nalu_hypre_ParVectorAllZeros(v) = 1;

   return nalu_hypre_ParVectorSetConstantValues(v, 0.0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorSetRandomValues( nalu_hypre_ParVector *v,
                                NALU_HYPRE_Int        seed )
{
   NALU_HYPRE_Int     my_id;
   nalu_hypre_Vector *v_local = nalu_hypre_ParVectorLocalVector(v);

   MPI_Comm     comm = nalu_hypre_ParVectorComm(v);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   seed *= (my_id + 1);

   return nalu_hypre_SeqVectorSetRandomValues(v_local, seed);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorCopy( nalu_hypre_ParVector *x,
                     nalu_hypre_ParVector *y )
{
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   return nalu_hypre_SeqVectorCopy(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCloneShallow
 *
 * Returns a complete copy of a nalu_hypre_ParVector x - a shallow copy, re-using
 * the partitioning and data arrays of x
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_ParVectorCloneShallow( nalu_hypre_ParVector *x )
{
   nalu_hypre_ParVector * y =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParVectorComm(x), nalu_hypre_ParVectorGlobalSize(x),
                            nalu_hypre_ParVectorPartitioning(x));

   nalu_hypre_ParVectorOwnsData(y) = 1;
   /* ...This vector owns its local vector, although the local vector doesn't
    * own _its_ data */
   nalu_hypre_SeqVectorDestroy( nalu_hypre_ParVectorLocalVector(y) );
   nalu_hypre_ParVectorLocalVector(y) = nalu_hypre_SeqVectorCloneShallow(nalu_hypre_ParVectorLocalVector(x) );
   nalu_hypre_ParVectorFirstIndex(y) = nalu_hypre_ParVectorFirstIndex(x);

   return y;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCloneDeep_v2
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_ParVectorCloneDeep_v2( nalu_hypre_ParVector *x, NALU_HYPRE_MemoryLocation memory_location )
{
   nalu_hypre_ParVector *y =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParVectorComm(x), nalu_hypre_ParVectorGlobalSize(x),
                            nalu_hypre_ParVectorPartitioning(x));

   nalu_hypre_ParVectorOwnsData(y) = 1;
   nalu_hypre_SeqVectorDestroy( nalu_hypre_ParVectorLocalVector(y) );
   nalu_hypre_ParVectorLocalVector(y) = nalu_hypre_SeqVectorCloneDeep_v2( nalu_hypre_ParVectorLocalVector(x),
                                                                memory_location );
   nalu_hypre_ParVectorFirstIndex(y) = nalu_hypre_ParVectorFirstIndex(x); //RL: WHY HERE?

   return y;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorMigrate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorMigrate(nalu_hypre_ParVector *x, NALU_HYPRE_MemoryLocation memory_location)
{
   if (!x)
   {
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_GetActualMemLocation(memory_location) !=
        nalu_hypre_GetActualMemLocation(nalu_hypre_ParVectorMemoryLocation(x)) )
   {
      nalu_hypre_Vector *x_local = nalu_hypre_SeqVectorCloneDeep_v2(nalu_hypre_ParVectorLocalVector(x), memory_location);
      nalu_hypre_SeqVectorDestroy(nalu_hypre_ParVectorLocalVector(x));
      nalu_hypre_ParVectorLocalVector(x) = x_local;
   }
   else
   {
      nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(x)) = memory_location;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorScale( NALU_HYPRE_Complex    alpha,
                      nalu_hypre_ParVector *y )
{
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   return nalu_hypre_SeqVectorScale(alpha, y_local);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorAxpy( NALU_HYPRE_Complex    alpha,
                     nalu_hypre_ParVector *x,
                     nalu_hypre_ParVector *y )
{
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   return nalu_hypre_SeqVectorAxpy(alpha, x_local, y_local);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorAxpyz
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorAxpyz( NALU_HYPRE_Complex    alpha,
                      nalu_hypre_ParVector *x,
                      NALU_HYPRE_Complex    beta,
                      nalu_hypre_ParVector *y,
                      nalu_hypre_ParVector *z )
{
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);
   nalu_hypre_Vector *z_local = nalu_hypre_ParVectorLocalVector(z);

   return nalu_hypre_SeqVectorAxpyz(alpha, x_local, beta, y_local, z_local);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_ParVectorInnerProd( nalu_hypre_ParVector *x,
                          nalu_hypre_ParVector *y )
{
   MPI_Comm      comm    = nalu_hypre_ParVectorComm(x);
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   NALU_HYPRE_Real result = 0.0;
   NALU_HYPRE_Real local_result = nalu_hypre_SeqVectorInnerProd(x_local, y_local);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] -= nalu_hypre_MPI_Wtime();
#endif
   nalu_hypre_MPI_Allreduce(&local_result, &result, 1, NALU_HYPRE_MPI_REAL,
                       nalu_hypre_MPI_SUM, comm);
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_ALL_REDUCE] += nalu_hypre_MPI_Wtime();
#endif

   return result;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorElmdivpy
 *
 * y = y + x ./ b [MATLAB Notation]
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorElmdivpy( nalu_hypre_ParVector *x,
                         nalu_hypre_ParVector *b,
                         nalu_hypre_ParVector *y )
{
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *b_local = nalu_hypre_ParVectorLocalVector(b);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   return nalu_hypre_SeqVectorElmdivpy(x_local, b_local, y_local);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorElmdivpyMarked
 *
 * y[i] += x[i] / b[i] where marker[i] == marker_val
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorElmdivpyMarked( nalu_hypre_ParVector *x,
                               nalu_hypre_ParVector *b,
                               nalu_hypre_ParVector *y,
                               NALU_HYPRE_Int       *marker,
                               NALU_HYPRE_Int        marker_val )
{
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *b_local = nalu_hypre_ParVectorLocalVector(b);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);

   return nalu_hypre_SeqVectorElmdivpyMarked(x_local, b_local, y_local, marker, marker_val);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_VectorToParVector
 *
 * Generates a ParVector from a Vector on proc 0 and distributes the pieces
 * to the other procs in comm
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_VectorToParVector ( MPI_Comm      comm,
                          nalu_hypre_Vector *v,
                          NALU_HYPRE_BigInt *vec_starts )
{
   NALU_HYPRE_BigInt        global_size;
   NALU_HYPRE_BigInt       *global_vec_starts = NULL;
   NALU_HYPRE_BigInt        first_index;
   NALU_HYPRE_BigInt        last_index;
   NALU_HYPRE_Int           local_size;
   NALU_HYPRE_Int           num_vectors;
   NALU_HYPRE_Int           num_procs, my_id;
   NALU_HYPRE_Int           global_vecstride, vecstride, idxstride;
   nalu_hypre_ParVector    *par_vector;
   nalu_hypre_Vector       *local_vector;
   NALU_HYPRE_Complex      *v_data;
   NALU_HYPRE_Complex      *local_data;
   nalu_hypre_MPI_Request  *requests;
   nalu_hypre_MPI_Status   *status, status0;
   NALU_HYPRE_Int           i, j, k, p;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == 0)
   {
      global_size = (NALU_HYPRE_BigInt)nalu_hypre_VectorSize(v);
      v_data = nalu_hypre_VectorData(v);
      num_vectors = nalu_hypre_VectorNumVectors(v); /* for multivectors */
      global_vecstride = nalu_hypre_VectorVectorStride(v);
   }

   nalu_hypre_MPI_Bcast(&global_size, 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   nalu_hypre_MPI_Bcast(&num_vectors, 1, NALU_HYPRE_MPI_INT, 0, comm);
   nalu_hypre_MPI_Bcast(&global_vecstride, 1, NALU_HYPRE_MPI_INT, 0, comm);

   if (num_vectors == 1)
   {
      par_vector = nalu_hypre_ParVectorCreate(comm, global_size, vec_starts);
   }
   else
   {
      par_vector = nalu_hypre_ParMultiVectorCreate(comm, global_size, vec_starts, num_vectors);
   }

   vec_starts  = nalu_hypre_ParVectorPartitioning(par_vector);
   first_index = nalu_hypre_ParVectorFirstIndex(par_vector);
   last_index  = nalu_hypre_ParVectorLastIndex(par_vector);
   local_size  = (NALU_HYPRE_Int)(last_index - first_index) + 1;

   if (my_id == 0)
   {
      global_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_MPI_Gather(&first_index, 1, NALU_HYPRE_MPI_BIG_INT, global_vec_starts,
                    1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   if (my_id == 0)
   {
      global_vec_starts[num_procs] = nalu_hypre_ParVectorGlobalSize(par_vector);
   }

   nalu_hypre_ParVectorInitialize(par_vector);
   local_vector = nalu_hypre_ParVectorLocalVector(par_vector);
   local_data = nalu_hypre_VectorData(local_vector);
   vecstride = nalu_hypre_VectorVectorStride(local_vector);
   idxstride = nalu_hypre_VectorIndexStride(local_vector);
   /* so far the only implemented multivector StorageMethod is 0 */
   nalu_hypre_assert( idxstride == 1 );

   if (my_id == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_vectors * (num_procs - 1), NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_vectors * (num_procs - 1), NALU_HYPRE_MEMORY_HOST);
      k = 0;
      for (p = 1; p < num_procs; p++)
         for (j = 0; j < num_vectors; ++j)
         {
            nalu_hypre_MPI_Isend( &v_data[(NALU_HYPRE_Int) global_vec_starts[p]] + j * global_vecstride,
                             (NALU_HYPRE_Int)(global_vec_starts[p + 1] - global_vec_starts[p]),
                             NALU_HYPRE_MPI_COMPLEX, p, 0, comm, &requests[k++] );
         }
      if (num_vectors == 1)
      {
         for (i = 0; i < local_size; i++)
         {
            local_data[i] = v_data[i];
         }
      }
      else
      {
         for (j = 0; j < num_vectors; ++j)
         {
            for (i = 0; i < local_size; i++)
            {
               local_data[i + j * vecstride] = v_data[i + j * global_vecstride];
            }
         }
      }
      nalu_hypre_MPI_Waitall(num_procs - 1, requests, status);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      for ( j = 0; j < num_vectors; ++j )
         nalu_hypre_MPI_Recv( local_data + j * vecstride, local_size, NALU_HYPRE_MPI_COMPLEX,
                         0, 0, comm, &status0 );
   }

   if (global_vec_starts)
   {
      nalu_hypre_TFree(global_vec_starts, NALU_HYPRE_MEMORY_HOST);
   }

   return par_vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorToVectorAll
 *
 * Generates a Vector on every proc which has a piece of the data
 * from a ParVector on several procs in comm,
 * vec_starts needs to contain the partitioning across all procs in comm
 *--------------------------------------------------------------------------*/

nalu_hypre_Vector *
nalu_hypre_ParVectorToVectorAll( nalu_hypre_ParVector *par_v )
{
   MPI_Comm             comm = nalu_hypre_ParVectorComm(par_v);
   NALU_HYPRE_BigInt         global_size = nalu_hypre_ParVectorGlobalSize(par_v);
   nalu_hypre_Vector        *local_vector = nalu_hypre_ParVectorLocalVector(par_v);
   NALU_HYPRE_Int            num_procs, my_id;
   NALU_HYPRE_Int            num_vectors = nalu_hypre_ParVectorNumVectors(par_v);
   nalu_hypre_Vector        *vector;
   NALU_HYPRE_Complex       *vector_data;
   NALU_HYPRE_Complex       *local_data;
   NALU_HYPRE_Int            local_size;
   nalu_hypre_MPI_Request   *requests;
   nalu_hypre_MPI_Status    *status;
   NALU_HYPRE_Int            i, j;
   NALU_HYPRE_Int           *used_procs;
   NALU_HYPRE_Int            num_types, num_requests;
   NALU_HYPRE_Int            vec_len, proc_id;

   NALU_HYPRE_Int *new_vec_starts;

   NALU_HYPRE_Int num_contacts;
   NALU_HYPRE_Int contact_proc_list[1];
   NALU_HYPRE_Int contact_send_buf[1];
   NALU_HYPRE_Int contact_send_buf_starts[2];
   NALU_HYPRE_Int max_response_size;
   NALU_HYPRE_Int *response_recv_buf = NULL;
   NALU_HYPRE_Int *response_recv_buf_starts = NULL;
   nalu_hypre_DataExchangeResponse response_obj;
   nalu_hypre_ProcListElements send_proc_obj;

   NALU_HYPRE_Int *send_info = NULL;
   nalu_hypre_MPI_Status  status1;
   NALU_HYPRE_Int count, tag1 = 112, tag2 = 223;
   NALU_HYPRE_Int start;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   local_size = (NALU_HYPRE_Int)(nalu_hypre_ParVectorLastIndex(par_v) -
                            nalu_hypre_ParVectorFirstIndex(par_v) + 1);

   /* determine procs which hold data of par_v and store ids in used_procs */
   /* we need to do an exchange data for this.  If I own row then I will contact
      processor 0 with the endpoint of my local range */

   if (local_size > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0] =  nalu_hypre_ParVectorLastIndex(par_v);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }

   /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_proc_obj.storage_length, NALU_HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_proc_obj.storage_length + 1, NALU_HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements =
      nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  send_proc_obj.element_storage_length, NALU_HYPRE_MEMORY_HOST);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = nalu_hypre_FillResponseParToVectorAll;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/


   nalu_hypre_DataExchangeList(num_contacts,
                          contact_proc_list, contact_send_buf,
                          contact_send_buf_starts, sizeof(NALU_HYPRE_Int),
                          //0, &response_obj,
                          sizeof(NALU_HYPRE_Int), &response_obj,
                          max_response_size, 1,
                          comm, (void**) &response_recv_buf,
                          &response_recv_buf_starts);

   /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row*/
   if (my_id)
   {
      if (local_size)
      {
         /* look for a message from processor 0 */
         nalu_hypre_MPI_Probe(0, tag1, comm, &status1);
         nalu_hypre_MPI_Get_count(&status1, NALU_HYPRE_MPI_INT, &count);

         send_info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  count, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_MPI_Recv(send_info, count, NALU_HYPRE_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */
         num_types = send_info[0];
         used_procs =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types, NALU_HYPRE_MEMORY_HOST);
         new_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types + 1, NALU_HYPRE_MEMORY_HOST);

         for (i = 1; i <= num_types; i++)
         {
            used_procs[i - 1] = (NALU_HYPRE_Int)send_info[i];
         }
         for (i = num_types + 1; i < count; i++)
         {
            new_vec_starts[i - num_types - 1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         nalu_hypre_TFree(send_proc_obj.vec_starts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_proc_obj.id, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_proc_obj.elements, NALU_HYPRE_MEMORY_HOST);
         if (response_recv_buf) { nalu_hypre_TFree(response_recv_buf, NALU_HYPRE_MEMORY_HOST); }
         if (response_recv_buf_starts) { nalu_hypre_TFree(response_recv_buf_starts, NALU_HYPRE_MEMORY_HOST); }
         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types, NALU_HYPRE_MEMORY_HOST);
      new_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types + 1, NALU_HYPRE_MEMORY_HOST);

      new_vec_starts[0] = 0;
      for (i = 0; i < num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i + 1] = send_proc_obj.elements[i] + 1;
      }
      nalu_hypre_qsort0(used_procs, 0, num_types - 1);
      nalu_hypre_qsort0(new_vec_starts, 0, num_types);
      /*now we need to put into an array to send */
      count =  2 * num_types + 2;
      send_info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  count, NALU_HYPRE_MEMORY_HOST);
      send_info[0] = num_types;
      for (i = 1; i <= num_types; i++)
      {
         send_info[i] = (NALU_HYPRE_Int)used_procs[i - 1];
      }
      for (i = num_types + 1; i < count; i++)
      {
         send_info[i] = new_vec_starts[i - num_types - 1];
      }
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_types, NALU_HYPRE_MEMORY_HOST);
      status =  nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_types, NALU_HYPRE_MEMORY_HOST);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }


      for (i = start; i < num_types; i++)
      {
         nalu_hypre_MPI_Isend(send_info, count, NALU_HYPRE_MPI_INT, used_procs[i],
                         tag1, comm, &requests[i - start]);
      }
      nalu_hypre_MPI_Waitall(num_types - start, requests, status);

      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   }

   /* clean up */
   nalu_hypre_TFree(send_proc_obj.vec_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_proc_obj.id, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_proc_obj.elements, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_info, NALU_HYPRE_MEMORY_HOST);
   if (response_recv_buf) { nalu_hypre_TFree(response_recv_buf, NALU_HYPRE_MEMORY_HOST); }
   if (response_recv_buf_starts) { nalu_hypre_TFree(response_recv_buf_starts, NALU_HYPRE_MEMORY_HOST); }

   /* now proc 0 can exit if it has no rows */
   if (!local_size)
   {
      nalu_hypre_TFree(used_procs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(new_vec_starts, NALU_HYPRE_MEMORY_HOST);
      return NULL;
   }

   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this vector should be rather small */

   local_data = nalu_hypre_VectorData(local_vector);
   vector = nalu_hypre_SeqVectorCreate((NALU_HYPRE_Int)global_size);
   nalu_hypre_VectorNumVectors(vector) = num_vectors;
   nalu_hypre_SeqVectorInitialize(vector);
   vector_data = nalu_hypre_VectorData(vector);

   num_requests = 2 * num_types;

   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_requests, NALU_HYPRE_MEMORY_HOST);

   /* initialize data exchange among used_procs and generate vector  - here we
      send to ourself also*/

   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      vec_len = (NALU_HYPRE_Int)(new_vec_starts[i + 1] - new_vec_starts[i]);
      nalu_hypre_MPI_Irecv(&vector_data[(NALU_HYPRE_Int)new_vec_starts[i]], num_vectors * vec_len,
                      NALU_HYPRE_MPI_COMPLEX, proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
      nalu_hypre_MPI_Isend(local_data, num_vectors * local_size, NALU_HYPRE_MPI_COMPLEX,
                      used_procs[i], tag2, comm, &requests[j++]);
   }

   nalu_hypre_MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(used_procs, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(new_vec_starts, NALU_HYPRE_MEMORY_HOST);

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorPrintIJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorPrintIJ( nalu_hypre_ParVector *vector,
                        NALU_HYPRE_Int        base_j,
                        const char      *filename )
{
   MPI_Comm          comm;
   NALU_HYPRE_BigInt      global_size, j;
   NALU_HYPRE_BigInt     *partitioning;
   NALU_HYPRE_Complex    *local_data;
   NALU_HYPRE_Int         myid, num_procs, i, part0;
   char              new_filename[255];
   FILE             *file;
   if (!vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   comm         = nalu_hypre_ParVectorComm(vector);
   global_size  = nalu_hypre_ParVectorGlobalSize(vector);
   partitioning = nalu_hypre_ParVectorPartitioning(vector);

   /* multivector code not written yet */
   nalu_hypre_assert( nalu_hypre_ParVectorNumVectors(vector) == 1 );
   if ( nalu_hypre_ParVectorNumVectors(vector) != 1 ) { nalu_hypre_error_in_arg(1); }

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   local_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(vector));

   nalu_hypre_fprintf(file, "%b \n", global_size);
   for (i = 0; i < 2; i++)
   {
      nalu_hypre_fprintf(file, "%b ", partitioning[i] + base_j);
   }
   nalu_hypre_fprintf(file, "\n");

   part0 = partitioning[0];
   for (j = part0; j < partitioning[1]; j++)
   {
      nalu_hypre_fprintf(file, "%b %.14e\n", j + base_j, local_data[(NALU_HYPRE_Int)(j - part0)]);
   }

   fclose(file);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorReadIJ
 * Warning: wrong base for assumed partition if base > 0
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorReadIJ( MPI_Comm          comm,
                       const char       *filename,
                       NALU_HYPRE_Int        *base_j_ptr,
                       nalu_hypre_ParVector **vector_ptr )
{
   NALU_HYPRE_BigInt      global_size, J;
   nalu_hypre_ParVector  *vector;
   nalu_hypre_Vector     *local_vector;
   NALU_HYPRE_Complex    *local_data;
   NALU_HYPRE_BigInt      partitioning[2];
   NALU_HYPRE_Int         base_j;

   NALU_HYPRE_Int         myid, num_procs, i, j;
   char              new_filename[255];
   FILE             *file;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_fscanf(file, "%b", &global_size);
   /* this may need to be changed so that the base is available in the file! */
   nalu_hypre_fscanf(file, "%b", partitioning);
   for (i = 0; i < 2; i++)
   {
      nalu_hypre_fscanf(file, "%b", partitioning + i);
   }
   /* This is not yet implemented correctly! */
   base_j = 0;
   vector = nalu_hypre_ParVectorCreate(comm, global_size,
                                  partitioning);

   nalu_hypre_ParVectorInitialize(vector);

   local_vector = nalu_hypre_ParVectorLocalVector(vector);
   local_data   = nalu_hypre_VectorData(local_vector);

   for (j = 0; j < (NALU_HYPRE_Int)(partitioning[1] - partitioning[0]); j++)
   {
      nalu_hypre_fscanf(file, "%b %le", &J, local_data + j);
   }

   fclose(file);

   *base_j_ptr = base_j;
   *vector_ptr = vector;

   /* multivector code not written yet */
   nalu_hypre_assert( nalu_hypre_ParVectorNumVectors(vector) == 1 );
   if ( nalu_hypre_ParVectorNumVectors(vector) != 1 ) { nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC); }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_FillResponseParToVectorAll
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FillResponseParToVectorAll( void       *p_recv_contact_buf,
                                  NALU_HYPRE_Int   contact_size,
                                  NALU_HYPRE_Int   contact_proc,
                                  void       *ro,
                                  MPI_Comm    comm,
                                  void      **p_send_response_buf,
                                  NALU_HYPRE_Int  *response_message_size )
{
   NALU_HYPRE_Int     myid;
   NALU_HYPRE_Int     i, index, count, elength;

   NALU_HYPRE_BigInt    *recv_contact_buf = (NALU_HYPRE_BigInt * ) p_recv_contact_buf;

   nalu_hypre_DataExchangeResponse  *response_obj = (nalu_hypre_DataExchangeResponse*)ro;

   nalu_hypre_ProcListElements      *send_proc_obj = (nalu_hypre_ProcListElements*)response_obj->data2;
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 10; /*add space for 10 more processors*/
      send_proc_obj->id = nalu_hypre_TReAlloc(send_proc_obj->id, NALU_HYPRE_Int,
                                         send_proc_obj->storage_length, NALU_HYPRE_MEMORY_HOST);
      send_proc_obj->vec_starts =
         nalu_hypre_TReAlloc(send_proc_obj->vec_starts, NALU_HYPRE_Int,
                        send_proc_obj->storage_length + 1, NALU_HYPRE_MEMORY_HOST);
   }

   /*initialize*/
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/
   send_proc_obj->id[count] = contact_proc;

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = nalu_hypre_max(contact_size, 10);
      elength += index;
      send_proc_obj->elements = nalu_hypre_TReAlloc(send_proc_obj->elements,
                                               NALU_HYPRE_BigInt,  elength, NALU_HYPRE_MEMORY_HOST);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   for (i = 0; i < contact_size; i++)
   {
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count + 1] = index;
   send_proc_obj->length++;

   /*output - no message to return (confirmation) */
   *response_message_size = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParVectorLocalSumElts
 *
 * Return the sum of all local elements of the vector
 *--------------------------------------------------------------------*/

NALU_HYPRE_Complex
nalu_hypre_ParVectorLocalSumElts( nalu_hypre_ParVector *vector )
{
   return nalu_hypre_SeqVectorSumElts( nalu_hypre_ParVectorLocalVector(vector) );
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParVectorGetValuesHost
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorGetValuesHost(nalu_hypre_ParVector *vector,
                             NALU_HYPRE_Int        num_values,
                             NALU_HYPRE_BigInt    *indices,
                             NALU_HYPRE_BigInt     base,
                             NALU_HYPRE_Complex   *values)
{
   NALU_HYPRE_BigInt    first_index  = nalu_hypre_ParVectorFirstIndex(vector);
   NALU_HYPRE_BigInt    last_index   = nalu_hypre_ParVectorLastIndex(vector);
   nalu_hypre_Vector   *local_vector = nalu_hypre_ParVectorLocalVector(vector);

   NALU_HYPRE_Int       component    = nalu_hypre_VectorComponent(local_vector);
   NALU_HYPRE_Int       vecstride    = nalu_hypre_VectorVectorStride(local_vector);
   NALU_HYPRE_Int       idxstride    = nalu_hypre_VectorIndexStride(local_vector);
   NALU_HYPRE_Complex  *data         = nalu_hypre_VectorData(local_vector);
   NALU_HYPRE_Int       vecoffset    = component * vecstride;

   NALU_HYPRE_Int       i, ierr = 0;

   if (indices)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) reduction(+:ierr) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_values; i++)
      {
         NALU_HYPRE_BigInt index = indices[i] - base;
         if (index < first_index || index > last_index)
         {
            ierr++;
         }
         else
         {
            NALU_HYPRE_Int local_index = (NALU_HYPRE_Int) (index - first_index);
            values[i] = data[vecoffset + local_index * idxstride];
         }
      }

      if (ierr)
      {
         nalu_hypre_error_in_arg(3);
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Index out of range! -- nalu_hypre_ParVectorGetValues.");
         nalu_hypre_printf("Index out of range! -- nalu_hypre_ParVectorGetValues\n");
      }
   }
   else
   {
      if (num_values > nalu_hypre_VectorSize(local_vector))
      {
         nalu_hypre_error_in_arg(2);
         return nalu_hypre_error_flag;
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_values; i++)
      {
         values[i] = data[vecoffset + i * idxstride];
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParVectorGetValues2
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorGetValues2(nalu_hypre_ParVector *vector,
                          NALU_HYPRE_Int        num_values,
                          NALU_HYPRE_BigInt    *indices,
                          NALU_HYPRE_BigInt     base,
                          NALU_HYPRE_Complex   *values)
{
#if defined(NALU_HYPRE_USING_GPU)
   if (NALU_HYPRE_EXEC_DEVICE == nalu_hypre_GetExecPolicy1( nalu_hypre_ParVectorMemoryLocation(vector) ))
   {
      nalu_hypre_ParVectorGetValuesDevice(vector, num_values, indices, base, values);
   }
   else
#endif
   {
      nalu_hypre_ParVectorGetValuesHost(vector, num_values, indices, base, values);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParVectorGetValues
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParVectorGetValues(nalu_hypre_ParVector *vector,
                         NALU_HYPRE_Int        num_values,
                         NALU_HYPRE_BigInt    *indices,
                         NALU_HYPRE_Complex   *values)
{
   return nalu_hypre_ParVectorGetValues2(vector, num_values, indices, 0, values);
}
