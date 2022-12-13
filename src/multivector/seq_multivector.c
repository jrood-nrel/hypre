/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

#include "seq_multivector.h"
#include "_hypre_utilities.h"

#include <stdlib.h>
#include <string.h>

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorCreate
 *--------------------------------------------------------------------------*/

hypre_Multivector *
hypre_SeqMultivectorCreate( NALU_HYPRE_Int size, NALU_HYPRE_Int num_vectors  )
{
   hypre_Multivector *mvector;

   mvector = hypre_TAlloc(hypre_Multivector, 1, NALU_HYPRE_MEMORY_HOST);

   hypre_MultivectorNumVectors(mvector) = num_vectors;
   hypre_MultivectorSize(mvector) = size;

   hypre_MultivectorOwnsData(mvector) = 1;
   hypre_MultivectorData(mvector) = NULL;

   mvector->num_active_vectors = 0;
   mvector->active_indices = NULL;

   return mvector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorInitialize( hypre_Multivector *mvector )
{
   NALU_HYPRE_Int    ierr = 0, i, size, num_vectors;

   size        = hypre_MultivectorSize(mvector);
   num_vectors = hypre_MultivectorNumVectors(mvector);

   if (NULL == hypre_MultivectorData(mvector))
      hypre_MultivectorData(mvector) =
         hypre_TAlloc(NALU_HYPRE_Complex, size * num_vectors, NALU_HYPRE_MEMORY_HOST);

   /* now we create a "mask" of "active" vectors; initially all active */
   if (NULL == mvector->active_indices)
   {
      mvector->active_indices hypre_CTAlloc(NALU_HYPRE_Int, num_vectors, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_vectors; i++) { mvector->active_indices[i] = i; }
      mvector->num_active_vectors = num_vectors;
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorSetDataOwner(hypre_Multivector *mvector, NALU_HYPRE_Int owns_data)
{
   NALU_HYPRE_Int    ierr = 0;

   hypre_MultivectorOwnsData(mvector) = owns_data;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorDestroy(hypre_Multivector *mvector)
{
   NALU_HYPRE_Int    ierr = 0;

   if (NULL != mvector)
   {
      if (hypre_MultivectorOwnsData(mvector) && NULL != hypre_MultivectorData(mvector))
      {
         hypre_TFree( hypre_MultivectorData(mvector), NALU_HYPRE_MEMORY_HOST);
      }

      if (NULL != mvector->active_indices)
      {
         hypre_TFree(mvector->active_indices, NALU_HYPRE_MEMORY_HOST);
      }

      hypre_TFree(mvector, NALU_HYPRE_MEMORY_HOST);
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorSetMask
 * (this routine accepts mask in "zeros and ones format, and converts it to
    the one used in the structure "hypre_Multivector")
 *-------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorSetMask(hypre_Multivector *mvector, NALU_HYPRE_Int * mask)
{
   NALU_HYPRE_Int  i, num_vectors = mvector->num_vectors;

   if (mvector->active_indices != NULL) { hypre_TFree(mvector->active_indices, NALU_HYPRE_MEMORY_HOST); }
   mvector->active_indices hypre_CTAlloc(NALU_HYPRE_Int, num_vectors, NALU_HYPRE_MEMORY_HOST);

   mvector->num_active_vectors = 0;

   if (mask != NULL)
      for (i = 0; i < num_vectors; i++)
      {
         if ( mask[i] )
         {
            mvector->active_indices[mvector->num_active_vectors++] = i;
         }
      }
   else
      for (i = 0; i < num_vectors; i++)
      {
         mvector->active_indices[mvector->num_active_vectors++] = i;
      }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorSetConstantValues(hypre_Multivector *v, NALU_HYPRE_Complex value)
{
   NALU_HYPRE_Int    i, j, start_offset, end_offset;
   NALU_HYPRE_Int    size        = hypre_MultivectorSize(v);
   NALU_HYPRE_Complex *vector_data = hypre_MultivectorData(v);

   if (v->num_active_vectors == v->num_vectors)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < v->num_vectors * size; j++) { vector_data[j] = value; }
   }
   else
   {
      for (i = 0; i < v->num_active_vectors; i++)
      {
         start_offset = v->active_indices[i] * size;
         end_offset = start_offset + size;

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (j = start_offset; j < end_offset; j++) { vector_data[j] = value; }
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorSetRandomValues(hypre_Multivector *v, NALU_HYPRE_Int seed)
{
   NALU_HYPRE_Int    i, j, start_offset, end_offset;
   NALU_HYPRE_Int    size        = hypre_MultivectorSize(v);
   NALU_HYPRE_Complex *vector_data = hypre_MultivectorData(v);

   hypre_SeedRand(seed);

   /* comment from vector.c: RDF: threading this loop may cause problems
      because of hypre_Rand() */

   if (v->num_active_vectors == v->num_vectors)
   {
      for (j = 0; j < v->num_vectors * size; j++)
      {
         vector_data[j] = 2.0 * hypre_Rand() - 1.0;
      }
   }
   else
   {
      for (i = 0; i < v->num_active_vectors; i++)
      {
         start_offset = v->active_indices[i] * size;
         end_offset = start_offset + size;
         for (j = start_offset; j < end_offset; j++)
         {
            vector_data[j] = 2.0 * hypre_Rand() - 1.0;
         }
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorCopy
 * copies data from x to y
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorCopy(hypre_Multivector *x, hypre_Multivector *y)
{
   NALU_HYPRE_Int    i, size, num_bytes, num_active_vectors, *x_active_ind, * y_active_ind;
   NALU_HYPRE_Complex *x_data, *y_data, *dest, * src;

   hypre_assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

   num_active_vectors = x->num_active_vectors;
   size = x->size;
   x_data = x->data;
   y_data = y->data;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;

   if (x->num_active_vectors == x->num_vectors &&
       y->num_active_vectors == y->num_vectors)
   {
      num_bytes = x->num_vectors * size;
      hypre_TMemcpy(y_data,  x_data, NALU_HYPRE_Complex, num_bytes, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      num_bytes = size;
      for (i = 0; i < num_active_vectors; i++)
      {
         src = x_data + size * x_active_ind[i];
         dest = y_data + size * y_active_ind[i];
         hypre_TMemcpy(dest, src, NALU_HYPRE_Complex, num_bytes, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
   }
   return 0;
}

NALU_HYPRE_Int
hypre_SeqMultivectorCopyWithoutMask(hypre_Multivector *x,
                                    hypre_Multivector *y)
{
   NALU_HYPRE_Int byte_count;

   hypre_assert (x->size == y->size && x->num_vectors == y->num_vectors);
   byte_count = x->size * x->num_vectors;
   hypre_TMemcpy(y->data, x->data, NALU_HYPRE_Complex, byte_count, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorAxpy(NALU_HYPRE_Complex alpha, hypre_Multivector *x,
                         hypre_Multivector *y)
{
   NALU_HYPRE_Int    i, j, size, num_active_vectors, *x_active_ind, *y_active_ind;
   NALU_HYPRE_Complex *x_data, *y_data, *src, *dest;

   hypre_assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   num_active_vectors = x->num_active_vectors;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;

   if (x->num_active_vectors == x->num_vectors &&
       y->num_active_vectors == y->num_vectors)
   {
      for (i = 0; i < x->num_vectors * size; i++) { dest[i] += alpha * src[i]; }
   }
   else
   {
      for (i = 0; i < num_active_vectors; i++)
      {
         src = x_data + x_active_ind[i] * size;
         dest = y_data + y_active_ind[i] * size;

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif

         for (j = 0; j < size; j++) { dest[j] += alpha * src[j]; }
      }
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorByDiag: " y(<y_mask>) = alpha(<mask>) .* x(<x_mask>) "
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SeqMultivectorByDiag(hypre_Multivector *x, NALU_HYPRE_Int *mask, NALU_HYPRE_Int n,
                           NALU_HYPRE_Complex *alpha, hypre_Multivector *y)
{
   NALU_HYPRE_Int    i, j, size, num_active_vectors, *x_active_ind, *y_active_ind;
   NALU_HYPRE_Int    *al_active_ind, num_active_als;
   NALU_HYPRE_Complex *x_data, *y_data, *dest, *src, current_alpha;

   hypre_assert (x->size == y->size && x->num_active_vectors == y->num_active_vectors);

   /* build list of active indices in alpha */

   al_active_ind = hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   num_active_als = 0;

   if (mask != NULL)
      for (i = 0; i < n; i++)
      {
         if (mask[i])
         {
            al_active_ind[num_active_als++] = i;
         }
      }
   else
      for (i = 0; i < n; i++)
      {
         al_active_ind[num_active_als++] = i;
      }

   hypre_assert (num_active_als == x->num_active_vectors);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   num_active_vectors = x->num_active_vectors;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;

   for (i = 0; i < num_active_vectors; i++)
   {
      src = x_data + x_active_ind[i] * size;
      dest = y_data + y_active_ind[i] * size;
      current_alpha = alpha[ al_active_ind[i] ];

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif

      for (j = 0; j < size; j++)
      {
         dest[j] = current_alpha * src[j];
      }
   }

   hypre_TFree(al_active_ind, NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int hypre_SeqMultivectorInnerProd(hypre_Multivector *x, hypre_Multivector *y,
                                        NALU_HYPRE_Real *results )
{
   NALU_HYPRE_Int      i, j, k, size, *x_active_ind, *y_active_ind;
   NALU_HYPRE_Int      x_num_active_vectors, y_num_active_vectors;
   NALU_HYPRE_Complex *x_data, *y_data, *y_ptr, *x_ptr;
   NALU_HYPRE_Real     current_product;

   hypre_assert (x->size == y->size);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   x_num_active_vectors = x->num_active_vectors;
   y_num_active_vectors = y->num_active_vectors;

   /* we assume that "results" points to contiguous array of (x_num_active_vectors X
      y_num_active_vectors) doubles */

   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;

   for (j = 0; j < y_num_active_vectors; j++)
   {
      y_ptr = y_data + y_active_ind[j] * size;

      for (i = 0; i < x_num_active_vectors; i++)
      {
         x_ptr = x_data + x_active_ind[i] * size;
         current_product = 0.0;

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(k) reduction(+:current_product) NALU_HYPRE_SMP_SCHEDULE
#endif

         for (k = 0; k < size; k++)
         {
            current_product += x_ptr[k] * hypre_conj(y_ptr[k]);
         }

         /* column-wise storage for results */
         *results++ = current_product;
      }
   }

   return 0;
}


/*--------------------------------------------------------------------------
 * hypre_SeqMultivectorInnerProdDiag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int hypre_SeqMultivectorInnerProdDiag(hypre_Multivector *x,
                                            hypre_Multivector *y, NALU_HYPRE_Real *diagResults)
{
   NALU_HYPRE_Complex *x_data, *y_data, *y_ptr, *x_ptr;
   NALU_HYPRE_Real     current_product;
   NALU_HYPRE_Int      i, k, size, num_active_vectors, *x_active_ind, *y_active_ind;

   hypre_assert(x->size == y->size && x->num_active_vectors == y->num_active_vectors);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   num_active_vectors = x->num_active_vectors;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;

   for (i = 0; i < num_active_vectors; i++)
   {
      x_ptr = x_data + x_active_ind[i] * size;
      y_ptr = y_data + y_active_ind[i] * size;
      current_product = 0.0;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(k) reduction(+:current_product) NALU_HYPRE_SMP_SCHEDULE
#endif

      for (k = 0; k < size; k++)
      {
         current_product += x_ptr[k] * hypre_conj(y_ptr[k]);
      }

      *diagResults++ = current_product;
   }
   return 0;
}

NALU_HYPRE_Int
hypre_SeqMultivectorByMatrix(hypre_Multivector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                             NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, hypre_Multivector *y)
{
   NALU_HYPRE_Int    i, j, k, size, gap, *x_active_ind, *y_active_ind;
   NALU_HYPRE_Complex *x_data, *y_data, *x_ptr, *y_ptr, current_coef;

   hypre_assert(rHeight > 0);
   hypre_assert (rHeight == x->num_active_vectors && rWidth == y->num_active_vectors);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;
   gap = rGHeight - rHeight;

   for (j = 0; j < rWidth; j++)
   {
      y_ptr = y_data + y_active_ind[j] * size;

      /* ------ set current "y" to first member in a sum ------ */
      x_ptr = x_data + x_active_ind[0] * size;
      current_coef = *rVal++;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(k) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (k = 0; k < size; k++)
      {
         y_ptr[k] = current_coef * x_ptr[k];
      }

      /* ------ now add all other members of a sum to "y" ----- */
      for (i = 1; i < rHeight; i++)
      {
         x_ptr = x_data + x_active_ind[i] * size;
         current_coef = *rVal++;

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(k) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (k = 0; k < size; k++)
         {
            y_ptr[k] += current_coef * x_ptr[k];
         }
      }

      rVal += gap;
   }

   return 0;
}

NALU_HYPRE_Int
hypre_SeqMultivectorXapy (hypre_Multivector *x, NALU_HYPRE_Int rGHeight, NALU_HYPRE_Int rHeight,
                          NALU_HYPRE_Int rWidth, NALU_HYPRE_Complex* rVal, hypre_Multivector *y)
{
   NALU_HYPRE_Complex *x_data, *y_data, *x_ptr, *y_ptr, current_coef;
   NALU_HYPRE_Int    i, j, k, size, gap, *x_active_ind, *y_active_ind;

   hypre_assert (rHeight == x->num_active_vectors && rWidth == y->num_active_vectors);

   x_data = x->data;
   y_data = y->data;
   size = x->size;
   x_active_ind = x->active_indices;
   y_active_ind = y->active_indices;
   gap = rGHeight - rHeight;

   for (j = 0; j < rWidth; j++)
   {
      y_ptr = y_data + y_active_ind[j] * size;

      for (i = 0; i < rHeight; i++)
      {
         x_ptr = x_data + x_active_ind[i] * size;
         current_coef = *rVal++;

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(k) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (k = 0; k < size; k++)
         {
            y_ptr[k] += current_coef * x_ptr[k];
         }
      }

      rVal += gap;
   }

   return 0;
}

