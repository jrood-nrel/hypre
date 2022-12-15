/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "par_multivector.h"
#include "seq_multivector.h"

#include "_nalu_hypre_utilities.h"

/* for temporary implementation of multivectorRead, multivectorPrint */
#include "seq_mv.h"
#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_ParMultiVector  *
nalu_hypre_ParMultiVectorCreate(MPI_Comm comm, NALU_HYPRE_Int global_size, NALU_HYPRE_Int *partitioning,
                           NALU_HYPRE_Int num_vectors)
{
   nalu_hypre_ParMultiVector *vector;
   NALU_HYPRE_Int num_procs, my_id;

   vector = nalu_hypre_CTAlloc(nalu_hypre_ParMultiVector,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (! partitioning)
   {
      nalu_hypre_MPI_Comm_size(comm, &num_procs);
      nalu_hypre_GeneratePartitioning(global_size, num_procs, &partitioning);
   }

   nalu_hypre_ParMultiVectorComm(vector) = comm;
   nalu_hypre_ParMultiVectorGlobalSize(vector) = global_size;
   nalu_hypre_ParMultiVectorPartitioning(vector) = partitioning;
   nalu_hypre_ParMultiVectorNumVectors(vector) = num_vectors;

   nalu_hypre_ParMultiVectorLocalVector(vector) =
      nalu_hypre_SeqMultivectorCreate((partitioning[my_id + 1] - partitioning[my_id]), num_vectors);

   nalu_hypre_ParMultiVectorFirstIndex(vector) = partitioning[my_id];

   /* we set these 2 defaults exactly as in par_vector.c, although it's questionable */
   nalu_hypre_ParMultiVectorOwnsData(vector) = 1;
   nalu_hypre_ParMultiVectorOwnsPartitioning(vector) = 1;

   return vector;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorDestroy( nalu_hypre_ParMultiVector *pm_vector )
{
   if (NULL != pm_vector)
   {
      if ( nalu_hypre_ParMultiVectorOwnsData(pm_vector) )
      {
         nalu_hypre_SeqMultivectorDestroy(nalu_hypre_ParMultiVectorLocalVector(pm_vector));
      }

      if ( nalu_hypre_ParMultiVectorOwnsPartitioning(pm_vector) )
      {
         nalu_hypre_TFree(nalu_hypre_ParMultiVectorPartitioning(pm_vector), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(pm_vector, NALU_HYPRE_MEMORY_HOST);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorInitialize( nalu_hypre_ParMultiVector *pm_vector )
{
   NALU_HYPRE_Int  ierr;

   ierr = nalu_hypre_SeqMultivectorInitialize(
             nalu_hypre_ParMultiVectorLocalVector(pm_vector));

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorSetDataOwner( nalu_hypre_ParMultiVector *pm_vector,
                                  NALU_HYPRE_Int           owns_data   )
{
   NALU_HYPRE_Int    ierr = 0;

   nalu_hypre_ParMultiVectorOwnsData(pm_vector) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorSetMask
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorSetMask( nalu_hypre_ParMultiVector *pm_vector, NALU_HYPRE_Int *mask)
{

   return nalu_hypre_SeqMultivectorSetMask(pm_vector->local_vector, mask);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorSetConstantValues( nalu_hypre_ParMultiVector *v,
                                       NALU_HYPRE_Complex        value )
{
   nalu_hypre_Multivector *v_local = nalu_hypre_ParMultiVectorLocalVector(v);

   return nalu_hypre_SeqMultivectorSetConstantValues(v_local, value);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorSetRandomValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorSetRandomValues( nalu_hypre_ParMultiVector *v, NALU_HYPRE_Int  seed)
{
   NALU_HYPRE_Int my_id;
   nalu_hypre_Multivector *v_local = nalu_hypre_ParMultiVectorLocalVector(v);

   MPI_Comm    comm = nalu_hypre_ParMultiVectorComm(v);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   seed *= (my_id + 1);

   return nalu_hypre_SeqMultivectorSetRandomValues(v_local, seed);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorCopy(nalu_hypre_ParMultiVector *x, nalu_hypre_ParMultiVector *y)
{
   nalu_hypre_Multivector *x_local = nalu_hypre_ParMultiVectorLocalVector(x);
   nalu_hypre_Multivector *y_local = nalu_hypre_ParMultiVectorLocalVector(y);

   return nalu_hypre_SeqMultivectorCopy(x_local, y_local);
}


NALU_HYPRE_Int
nalu_hypre_ParMultiVectorCopyWithoutMask(nalu_hypre_ParMultiVector *x, nalu_hypre_ParMultiVector *y)
{
   return nalu_hypre_SeqMultivectorCopyWithoutMask(x->local_vector, y->local_vector);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorScale(NALU_HYPRE_Complex alpha, nalu_hypre_ParMultiVector *y)
{
   return 1 ; /* nalu_hypre_SeqMultivectorScale( alpha, y_local, NULL); */
}


/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorMultiScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorMultiScale(NALU_HYPRE_Complex *alpha, nalu_hypre_ParMultiVector *y)
{
   return 1; /* nalu_hypre_SeqMultivectorMultiScale(alpha, y_local, NULL); */
}



/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorAxpy(NALU_HYPRE_Complex alpha, nalu_hypre_ParMultiVector *x,
                         nalu_hypre_ParMultiVector *y)
{
   nalu_hypre_Multivector *x_local = nalu_hypre_ParMultiVectorLocalVector(x);
   nalu_hypre_Multivector *y_local = nalu_hypre_ParMultiVectorLocalVector(y);

   return nalu_hypre_SeqMultivectorAxpy( alpha, x_local, y_local);
}


/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorByDiag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorByDiag(nalu_hypre_ParMultiVector *x, NALU_HYPRE_Int *mask, NALU_HYPRE_Int n,
                           NALU_HYPRE_Complex *alpha, nalu_hypre_ParMultiVector *y)
{
   return nalu_hypre_SeqMultivectorByDiag(x->local_vector, mask, n, alpha,
                                     y->local_vector);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorInnerProd(nalu_hypre_ParMultiVector *x, nalu_hypre_ParMultiVector *y,
                              NALU_HYPRE_Real *results, NALU_HYPRE_Real *workspace )
{
   MPI_Comm           comm;
   NALU_HYPRE_Int                count;
   NALU_HYPRE_Int                ierr;
   /*
    *    NALU_HYPRE_Int                myid;
    *    NALU_HYPRE_Int                i
    */

   /* assuming "results" and "workspace" are arrays of size ("n_active_x" by "n_active_y")
      n_active_x is the number of active vectors in multivector x
      the product "x^T * y" will be stored in "results" column-wise; workspace will be used for
      computation of local matrices; maybe nalu_hypre_MPI_IN_PLACE functionality will be added later */

   nalu_hypre_SeqMultivectorInnerProd(x->local_vector, y->local_vector, workspace);

   comm = x->comm;
   count = (x->local_vector->num_active_vectors) *
           (y->local_vector->num_active_vectors);

   ierr = nalu_hypre_MPI_Allreduce(workspace, results, count, NALU_HYPRE_MPI_REAL,
                              nalu_hypre_MPI_SUM, comm);
   nalu_hypre_assert (ierr == nalu_hypre_MPI_SUCCESS);

   /* debug */

   /*
    *    nalu_hypre_MPI_Comm_rank(comm, &myid);
    *    if (myid==0)
    *       for (i=0; i<count; i++)
    *          nalu_hypre_printf("%22.14e\n",results[i])
    */

   /* ------------ */

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMultiVectorInnerProdDiag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorInnerProdDiag(nalu_hypre_ParMultiVector *x, nalu_hypre_ParMultiVector *y,
                                  NALU_HYPRE_Real *diagResults, NALU_HYPRE_Real *workspace )
{
   NALU_HYPRE_Int   count;
   NALU_HYPRE_Int   ierr;

   nalu_hypre_SeqMultivectorInnerProdDiag(x->local_vector, y->local_vector, workspace);

   count = x->local_vector->num_active_vectors;
   ierr = nalu_hypre_MPI_Allreduce(workspace, diagResults, count, NALU_HYPRE_MPI_REAL,
                              nalu_hypre_MPI_SUM, x->comm);
   nalu_hypre_assert (ierr == nalu_hypre_MPI_SUCCESS);

   return 0;
}

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorByMatrix(nalu_hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                             NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_ParMultiVector * y)
{
   return nalu_hypre_SeqMultivectorByMatrix(x->local_vector, rGHeight, rHeight,
                                       rWidth, rVal, y->local_vector);
}

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorXapy(nalu_hypre_ParMultiVector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                         NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, nalu_hypre_ParMultiVector * y)
{
   return nalu_hypre_SeqMultivectorXapy(x->local_vector, rGHeight, rHeight,
                                   rWidth, rVal, y->local_vector);
}

/* temporary function; allows to do "matvec" and preconditioner in
   vector-by-vector fashion */
NALU_HYPRE_Int
nalu_hypre_ParMultiVectorEval(void (*f)( void*, void*, void* ), void* par,
                         nalu_hypre_ParMultiVector * x, nalu_hypre_ParMultiVector * y)
{
   nalu_hypre_ParVector  *temp_x, *temp_y;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_active_vectors;
   NALU_HYPRE_Int *x_active_indices, *y_active_indices;
   NALU_HYPRE_Complex * x_data, *y_data;
   NALU_HYPRE_Int size;

   nalu_hypre_assert(x->local_vector->num_active_vectors == y->local_vector->num_active_vectors);
   nalu_hypre_assert(x->local_vector->size == y->local_vector->size);

   temp_x = nalu_hypre_ParVectorCreate(x->comm, x->global_size, x->partitioning);
   nalu_hypre_assert(temp_x != NULL);
   temp_x->local_vector->owns_data = 0;
   temp_x->local_vector->vecstride = temp_x->local_vector->size;
   temp_x->local_vector->idxstride = 1;
   /* no initialization for temp_x needed! */

   temp_y = nalu_hypre_ParVectorCreate(y->comm, y->global_size, y->partitioning);
   nalu_hypre_assert(temp_y != NULL);
   temp_y->local_vector->owns_data = 0;
   temp_y->local_vector->vecstride = temp_y->local_vector->size;
   temp_y->local_vector->idxstride = 1;
   /* no initialization for temp_y needed! */

   num_active_vectors = x->local_vector->num_active_vectors;
   x_active_indices = x->local_vector->active_indices;
   y_active_indices = y->local_vector->active_indices;
   x_data = x->local_vector->data;
   y_data = y->local_vector->data;
   size = x->local_vector->size;

   for ( i = 0; i < num_active_vectors; i++ )
   {
      temp_x->local_vector->data = x_data + x_active_indices[i] * size;
      temp_y->local_vector->data = y_data + y_active_indices[i] * size;

      /*** here i make an assumption that "f" will treat temp_x and temp_y like
            "nalu_hypre_ParVector *" variables ***/

      f( par, temp_x, temp_y );
   }

   nalu_hypre_ParVectorDestroy(temp_x);
   nalu_hypre_ParVectorDestroy(temp_y);
   /* 2 lines above won't free data or partitioning */

   return 0;
}

nalu_hypre_ParMultiVector *
nalu_hypre_ParMultiVectorTempRead(MPI_Comm comm, const char *fileName)
/* ***** temporary implementation ****** */
{
   NALU_HYPRE_Int i, n, id;
   NALU_HYPRE_Complex * dest;
   NALU_HYPRE_Complex * src;
   NALU_HYPRE_Int count;
   NALU_HYPRE_Int retcode;
   char temp_string[128];
   nalu_hypre_ParMultiVector * x;
   nalu_hypre_ParVector * temp_vec;

   /* calculate the number of files */
   nalu_hypre_MPI_Comm_rank( comm, &id );
   n = 0;
   do
   {
      nalu_hypre_sprintf( temp_string, "test -f %s.%d.%d", fileName, n, id );
      if (!(retcode = system(temp_string))) /* zero retcode mean file exists */
      {
         n++;
      }
   }
   while (!retcode);

   if ( n == 0 ) { return NULL; }

   /* now read the first vector using nalu_hypre_ParVectorRead into temp_vec */

   nalu_hypre_sprintf(temp_string, "%s.%d", fileName, 0);
   temp_vec = nalu_hypre_ParVectorRead(comm, temp_string);

   /* now create multivector using temp_vec as a sample */

   x = nalu_hypre_ParMultiVectorCreate(nalu_hypre_ParVectorComm(temp_vec),
                                  nalu_hypre_ParVectorGlobalSize(temp_vec), nalu_hypre_ParVectorPartitioning(temp_vec), n);

   nalu_hypre_ParMultiVectorInitialize(x);

   /* read data from first and all other vectors into "x" */

   i = 0;
   do
   {
      /* copy data from current vector */
      dest = x->local_vector->data + i * (x->local_vector->size);
      src = temp_vec->local_vector->data;
      count = temp_vec->local_vector->size;

      nalu_hypre_TMemcpy(dest, src, NALU_HYPRE_Complex, count, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

      /* destroy current vector */
      nalu_hypre_ParVectorDestroy(temp_vec);

      /* read the data to new current vector, if there are more vectors to read */
      if (i < n - 1)
      {
         nalu_hypre_sprintf(temp_string, "%s.%d", fileName, i + 1);
         temp_vec = nalu_hypre_ParVectorRead(comm, temp_string);

      }
   }
   while (++i < n);

   return x;
}

NALU_HYPRE_Int
nalu_hypre_ParMultiVectorTempPrint(nalu_hypre_ParMultiVector *vector, const char *fileName)
{
   NALU_HYPRE_Int i, ierr;
   char fullName[128];
   nalu_hypre_ParVector * temp_vec;

   nalu_hypre_assert( vector != NULL );

   temp_vec = nalu_hypre_ParVectorCreate(vector->comm, vector->global_size, vector->partitioning);
   nalu_hypre_assert(temp_vec != NULL);
   temp_vec->local_vector->owns_data = 0;

   /* no initialization for temp_vec needed! */

   ierr = 0;
   for ( i = 0; i < vector->local_vector->num_vectors; i++ )
   {
      nalu_hypre_sprintf( fullName, "%s.%d", fileName, i );

      temp_vec->local_vector->data = vector->local_vector->data + i *
                                     vector->local_vector->size;

      ierr = ierr || nalu_hypre_ParVectorPrint(temp_vec, fullName);
   }

   ierr = ierr || nalu_hypre_ParVectorDestroy(temp_vec);
   /* line above won't free data or partitioning */

   return ierr;
}
