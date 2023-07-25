/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_AMGDDCompGrid and nalu_hypre_AMGDDCommPkg classes.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.h"

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridLocalIndexBinarySearch( nalu_hypre_AMGDDCompGrid *compGrid,
                                           NALU_HYPRE_Int            global_index )
{
   NALU_HYPRE_Int   *nonowned_global_indices;
   NALU_HYPRE_Int   *inv_map;
   NALU_HYPRE_Int    left;
   NALU_HYPRE_Int    right;
   NALU_HYPRE_Int    index, sorted_index;

   // Set data
   nonowned_global_indices = nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid);
   inv_map = nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid);

   left  = 0;
   right = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid) - 1;
   while (left <= right)
   {
      sorted_index = (left + right) / 2;
      index = inv_map[sorted_index];
      if (nonowned_global_indices[index] < global_index)
      {
         left = sorted_index + 1;
      }
      else if (nonowned_global_indices[index] > global_index)
      {
         right = sorted_index - 1;
      }
      else
      {
         return index;
      }
   }

   return -1;
}

nalu_hypre_AMGDDCompGridMatrix* nalu_hypre_AMGDDCompGridMatrixCreate( void )
{
   nalu_hypre_AMGDDCompGridMatrix *matrix = nalu_hypre_CTAlloc(nalu_hypre_AMGDDCompGridMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_AMGDDCompGridMatrixOwnedDiag(matrix)    = NULL;
   nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)    = NULL;
   nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix) = NULL;
   nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix) = NULL;

   nalu_hypre_AMGDDCompGridMatrixRealReal(matrix)  = NULL;
   nalu_hypre_AMGDDCompGridMatrixRealGhost(matrix) = NULL;

   nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix)  = 0;
   nalu_hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix) = 0;

   return matrix;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridMatrixDestroy( nalu_hypre_AMGDDCompGridMatrix *matrix )
{
   if (matrix)
   {
      if (nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix))
      {
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixOwnedDiag(matrix));
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix));
      }
      else if (nalu_hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix))
      {
         NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(
                                                   nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix));

         if (nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)))
         {
            nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)), memory_location);
         }

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
         nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedData(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)), memory_location);
         nalu_hypre_CsrsvDataDestroy(nalu_hypre_CSRMatrixCsrsvData(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)));
         nalu_hypre_GpuMatDataDestroy(nalu_hypre_CSRMatrixGPUMatData(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix)));
#endif

         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(matrix), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix));
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix));
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixRealReal(matrix));
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_AMGDDCompGridMatrixRealGhost(matrix));

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridMatrixSetupRealMatvec( nalu_hypre_AMGDDCompGridMatrix *A )
{
   nalu_hypre_CSRMatrix  *A_real_real  = nalu_hypre_AMGDDCompGridMatrixRealReal(A);
   nalu_hypre_CSRMatrix  *A_real_ghost = nalu_hypre_AMGDDCompGridMatrixRealGhost(A);
   nalu_hypre_CSRMatrix  *A_diag       = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);

   NALU_HYPRE_Int        *A_i,   *A_j;
   NALU_HYPRE_Int        *A_rri, *A_rrj;
   NALU_HYPRE_Int        *A_rgi, *A_rgj;
   NALU_HYPRE_Complex    *A_data, *A_rrdata, *A_rgdata;

   NALU_HYPRE_Int         num_real = nalu_hypre_CSRMatrixNumRows(A_real_real);
   NALU_HYPRE_Int         A_real_real_nnz;
   NALU_HYPRE_Int         A_real_ghost_nnz;
   NALU_HYPRE_Int         i, j, col_ind;

   // Initialize matrices
   nalu_hypre_CSRMatrixInitialize(A_real_real);
   nalu_hypre_CSRMatrixInitialize(A_real_ghost);

   // Set some data
   A_i      = nalu_hypre_CSRMatrixI(A_diag);
   A_rri    = nalu_hypre_CSRMatrixI(A_real_real);
   A_rgi    = nalu_hypre_CSRMatrixI(A_real_ghost);
   A_j      = nalu_hypre_CSRMatrixJ(A_diag);
   A_rrj    = nalu_hypre_CSRMatrixJ(A_real_real);
   A_rgj    = nalu_hypre_CSRMatrixJ(A_real_ghost);
   A_data   = nalu_hypre_CSRMatrixData(A_diag);
   A_rrdata = nalu_hypre_CSRMatrixData(A_real_real);
   A_rgdata = nalu_hypre_CSRMatrixData(A_real_ghost);

   A_real_real_nnz = A_real_ghost_nnz = 0;
   for (i = 0; i < num_real; i++)
   {
      A_rri[i] = A_real_real_nnz;
      A_rgi[i] = A_real_ghost_nnz;
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         col_ind = A_j[j];
         if (col_ind < num_real)
         {
            A_rrj[A_real_real_nnz]    = col_ind;
            A_rrdata[A_real_real_nnz] = A_data[j];
            A_real_real_nnz++;
         }
         else
         {
            A_rgj[A_real_ghost_nnz]    = col_ind;
            A_rgdata[A_real_ghost_nnz] = A_data[j];
            A_real_ghost_nnz++;
         }
      }
   }

   A_rri[num_real] = A_real_real_nnz;
   A_rgi[num_real] = A_real_ghost_nnz;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridMatvec( NALU_HYPRE_Complex alpha,
                           nalu_hypre_AMGDDCompGridMatrix *A,
                           nalu_hypre_AMGDDCompGridVector *x,
                           NALU_HYPRE_Complex beta,
                           nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_CSRMatrix *owned_diag    = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A);
   nalu_hypre_CSRMatrix *owned_offd    = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A);
   nalu_hypre_CSRMatrix *nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   nalu_hypre_CSRMatrix *nonowned_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   nalu_hypre_Vector *x_owned    = nalu_hypre_AMGDDCompGridVectorOwned(x);
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_owned    = nalu_hypre_AMGDDCompGridVectorOwned(y);
   nalu_hypre_Vector *y_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(y);

   nalu_hypre_CSRMatrixMatvec(alpha, owned_diag, x_owned, beta, y_owned);

   if (owned_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, owned_offd, x_nonowned, 1.0, y_owned);
   }

   if (nonowned_diag)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, nonowned_diag, x_nonowned, beta, y_nonowned);
   }

   if (nonowned_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, nonowned_offd, x_owned, 1.0, y_nonowned);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridRealMatvec( NALU_HYPRE_Complex alpha,
                               nalu_hypre_AMGDDCompGridMatrix *A,
                               nalu_hypre_AMGDDCompGridVector *x,
                               NALU_HYPRE_Complex beta,
                               nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_CSRMatrix *owned_diag    = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A);
   nalu_hypre_CSRMatrix *owned_offd    = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A);
   nalu_hypre_CSRMatrix *nonowned_diag = nalu_hypre_AMGDDCompGridMatrixRealReal(A);
   nalu_hypre_CSRMatrix *nonowned_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   nalu_hypre_Vector *x_owned    = nalu_hypre_AMGDDCompGridVectorOwned(x);
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_owned    = nalu_hypre_AMGDDCompGridVectorOwned(y);
   nalu_hypre_Vector *y_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(y);

   if (!nalu_hypre_CSRMatrixData(nalu_hypre_AMGDDCompGridMatrixRealReal(A)))
   {
      nalu_hypre_AMGDDCompGridMatrixSetupRealMatvec(A);
   }

   nalu_hypre_CSRMatrixMatvec(alpha, owned_diag, x_owned, beta, y_owned);

   if (owned_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, owned_offd, x_nonowned, 1.0, y_owned);
   }

   if (nonowned_diag)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, nonowned_diag, x_nonowned, beta, y_nonowned);
   }

   if (nonowned_offd)
   {
      nalu_hypre_CSRMatrixMatvec(alpha, nonowned_offd, x_owned, 1.0, y_nonowned);
   }

   return nalu_hypre_error_flag;
}

nalu_hypre_AMGDDCompGridVector *nalu_hypre_AMGDDCompGridVectorCreate( void )
{
   nalu_hypre_AMGDDCompGridVector *vector = nalu_hypre_CTAlloc(nalu_hypre_AMGDDCompGridVector, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_AMGDDCompGridVectorOwned(vector)    = NULL;
   nalu_hypre_AMGDDCompGridVectorNonOwned(vector) = NULL;

   nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(vector) = 0;

   return vector;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorInitialize( nalu_hypre_AMGDDCompGridVector *vector,
                                     NALU_HYPRE_Int num_owned,
                                     NALU_HYPRE_Int num_nonowned,
                                     NALU_HYPRE_Int num_real )
{
   nalu_hypre_AMGDDCompGridVectorOwned(vector) = nalu_hypre_SeqVectorCreate(num_owned);
   nalu_hypre_SeqVectorInitialize(nalu_hypre_AMGDDCompGridVectorOwned(vector));
   nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(vector) = 1;
   nalu_hypre_AMGDDCompGridVectorNumReal(vector) = num_real;
   nalu_hypre_AMGDDCompGridVectorNonOwned(vector) = nalu_hypre_SeqVectorCreate(num_nonowned);
   nalu_hypre_SeqVectorInitialize(nalu_hypre_AMGDDCompGridVectorNonOwned(vector));

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridVector *vector)
{
   if (vector)
   {
      if (nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(vector))
      {
         if (nalu_hypre_AMGDDCompGridVectorOwned(vector))
         {
            nalu_hypre_SeqVectorDestroy(nalu_hypre_AMGDDCompGridVectorOwned(vector));
         }
      }

      if (nalu_hypre_AMGDDCompGridVectorNonOwned(vector))
      {
         nalu_hypre_SeqVectorDestroy(nalu_hypre_AMGDDCompGridVectorNonOwned(vector));
      }

      nalu_hypre_TFree(vector, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Real
nalu_hypre_AMGDDCompGridVectorInnerProd( nalu_hypre_AMGDDCompGridVector *x,
                                    nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_Vector *x_owned    = nalu_hypre_AMGDDCompGridVectorOwned(x);
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_owned    = nalu_hypre_AMGDDCompGridVectorOwned(y);
   nalu_hypre_Vector *y_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(y);

   NALU_HYPRE_Real    res;

   res  = nalu_hypre_SeqVectorInnerProd(x_owned, y_owned);
   res += nalu_hypre_SeqVectorInnerProd(x_nonowned, y_nonowned);

   return res;
}

NALU_HYPRE_Real
nalu_hypre_AMGDDCompGridVectorRealInnerProd( nalu_hypre_AMGDDCompGridVector *x,
                                        nalu_hypre_AMGDDCompGridVector *y)
{
   nalu_hypre_Vector *x_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(y);
   NALU_HYPRE_Int     orig_x_size = nalu_hypre_VectorSize(x_nonowned);
   NALU_HYPRE_Int     orig_y_size = nalu_hypre_VectorSize(y_nonowned);
   NALU_HYPRE_Real res;

   nalu_hypre_VectorSize(x_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(x);
   nalu_hypre_VectorSize(y_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(y);

   res = nalu_hypre_AMGDDCompGridVectorInnerProd(x, y);

   nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;
   nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(y)) = orig_y_size;

   return res;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorScale( NALU_HYPRE_Complex              alpha,
                                nalu_hypre_AMGDDCompGridVector *x )
{
   nalu_hypre_SeqVectorScale(alpha, nalu_hypre_AMGDDCompGridVectorOwned(x));
   nalu_hypre_SeqVectorScale(alpha, nalu_hypre_AMGDDCompGridVectorNonOwned(x));

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorRealScale( NALU_HYPRE_Complex              alpha,
                                    nalu_hypre_AMGDDCompGridVector *x )
{
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);

   NALU_HYPRE_Int orig_x_size = nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(x));

   nalu_hypre_VectorSize(x_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(x);

   nalu_hypre_AMGDDCompGridVectorScale(alpha, x);

   nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorAxpy( NALU_HYPRE_Complex              alpha,
                               nalu_hypre_AMGDDCompGridVector *x,
                               nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_Vector *x_owned    = nalu_hypre_AMGDDCompGridVectorOwned(x);
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_owned    = nalu_hypre_AMGDDCompGridVectorOwned(y);
   nalu_hypre_Vector *y_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(y);

   if (x_owned)
   {
      nalu_hypre_SeqVectorAxpy(alpha, x_owned, y_owned);
   }

   if (x_nonowned)
   {
      nalu_hypre_SeqVectorAxpy(alpha, x_nonowned, y_nonowned);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorRealAxpy( NALU_HYPRE_Complex              alpha,
                                   nalu_hypre_AMGDDCompGridVector *x,
                                   nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_Vector *x_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(y);
   NALU_HYPRE_Int     orig_x_size = nalu_hypre_VectorSize(x_nonowned);
   NALU_HYPRE_Int     orig_y_size = nalu_hypre_VectorSize(y_nonowned);

   nalu_hypre_VectorSize(x_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(x);
   nalu_hypre_VectorSize(y_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(y);

   nalu_hypre_AMGDDCompGridVectorAxpy(alpha, x, y);

   nalu_hypre_VectorSize(x_nonowned) = orig_x_size;
   nalu_hypre_VectorSize(y_nonowned) = orig_y_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorSetConstantValues( nalu_hypre_AMGDDCompGridVector *vector,
                                            NALU_HYPRE_Complex              value )
{
   nalu_hypre_Vector *vector_owned = nalu_hypre_AMGDDCompGridVectorOwned(vector);
   nalu_hypre_Vector *vector_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(vector);

   if (vector_owned)
   {
      nalu_hypre_SeqVectorSetConstantValues(vector_owned, value);
   }

   if (vector_nonowned)
   {
      nalu_hypre_SeqVectorSetConstantValues(vector_nonowned, value);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorRealSetConstantValues( nalu_hypre_AMGDDCompGridVector *vector,
                                                NALU_HYPRE_Complex              value )
{
   nalu_hypre_Vector *vector_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(vector);
   NALU_HYPRE_Int     orig_vec_size   = nalu_hypre_VectorSize(vector_nonowned);

   nalu_hypre_VectorSize(vector_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(vector);

   nalu_hypre_AMGDDCompGridVectorSetConstantValues(vector, value);

   nalu_hypre_VectorSize(vector_nonowned) = orig_vec_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorCopy( nalu_hypre_AMGDDCompGridVector *x,
                               nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_Vector *x_owned    = nalu_hypre_AMGDDCompGridVectorOwned(x);
   nalu_hypre_Vector *x_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_owned    = nalu_hypre_AMGDDCompGridVectorOwned(y);
   nalu_hypre_Vector *y_nonowned = nalu_hypre_AMGDDCompGridVectorNonOwned(y);

   if (x_owned && y_owned)
   {
      nalu_hypre_SeqVectorCopy(x_owned, y_owned);
   }
   if (x_nonowned && y_nonowned)
   {
      nalu_hypre_SeqVectorCopy(x_nonowned, y_nonowned);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridVectorRealCopy( nalu_hypre_AMGDDCompGridVector *x,
                                   nalu_hypre_AMGDDCompGridVector *y )
{
   nalu_hypre_Vector *x_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(x);
   nalu_hypre_Vector *y_nonowned  = nalu_hypre_AMGDDCompGridVectorNonOwned(y);
   NALU_HYPRE_Int     orig_x_size = nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(x));
   NALU_HYPRE_Int     orig_y_size = nalu_hypre_VectorSize(nalu_hypre_AMGDDCompGridVectorNonOwned(y));

   nalu_hypre_VectorSize(x_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(x);
   nalu_hypre_VectorSize(y_nonowned) = nalu_hypre_AMGDDCompGridVectorNumReal(y);

   nalu_hypre_AMGDDCompGridVectorCopy(x, y);

   nalu_hypre_VectorSize(x_nonowned) = orig_x_size;
   nalu_hypre_VectorSize(y_nonowned) = orig_y_size;

   return nalu_hypre_error_flag;
}

nalu_hypre_AMGDDCompGrid *nalu_hypre_AMGDDCompGridCreate ( void )
{
   nalu_hypre_AMGDDCompGrid      *compGrid;

   compGrid = nalu_hypre_CTAlloc(nalu_hypre_AMGDDCompGrid, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCompGridMemoryLocation(compGrid) = NALU_HYPRE_MEMORY_UNDEFINED;

   nalu_hypre_AMGDDCompGridFirstGlobalIndex(compGrid)       = 0;
   nalu_hypre_AMGDDCompGridLastGlobalIndex(compGrid)        = 0;
   nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)          = 0;
   nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)       = 0;
   nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid)   = 0;
   nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid)   = 0;

   nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)         = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid)         = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)            = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid)                  = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid)               = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) = NULL;

   nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid) = NULL;

   nalu_hypre_AMGDDCompGridA(compGrid) = NULL;
   nalu_hypre_AMGDDCompGridP(compGrid) = NULL;
   nalu_hypre_AMGDDCompGridR(compGrid) = NULL;

   nalu_hypre_AMGDDCompGridU(compGrid)     = NULL;
   nalu_hypre_AMGDDCompGridF(compGrid)     = NULL;
   nalu_hypre_AMGDDCompGridT(compGrid)     = NULL;
   nalu_hypre_AMGDDCompGridS(compGrid)     = NULL;
   nalu_hypre_AMGDDCompGridQ(compGrid)     = NULL;
   nalu_hypre_AMGDDCompGridTemp(compGrid)  = NULL;
   nalu_hypre_AMGDDCompGridTemp2(compGrid) = NULL;
   nalu_hypre_AMGDDCompGridTemp3(compGrid) = NULL;

   nalu_hypre_AMGDDCompGridL1Norms(compGrid)               = NULL;
   nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)         = NULL;
   nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid)    = NULL;
   nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) = NULL;

   return compGrid;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridDestroy( nalu_hypre_AMGDDCompGrid *compGrid )
{
   NALU_HYPRE_MemoryLocation  memory_location;

   if (compGrid)
   {
      memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);

      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridL1Norms(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid), memory_location);
      nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid), memory_location);

      nalu_hypre_AMGDDCompGridMatrixDestroy(nalu_hypre_AMGDDCompGridA(compGrid));
      nalu_hypre_AMGDDCompGridMatrixDestroy(nalu_hypre_AMGDDCompGridP(compGrid));
      nalu_hypre_AMGDDCompGridMatrixDestroy(nalu_hypre_AMGDDCompGridR(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridU(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridF(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridT(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridS(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridQ(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridTemp(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridTemp2(compGrid));
      nalu_hypre_AMGDDCompGridVectorDestroy(nalu_hypre_AMGDDCompGridTemp3(compGrid));

      nalu_hypre_TFree(compGrid, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridInitialize( nalu_hypre_ParAMGDDData *amgdd_data,
                               NALU_HYPRE_Int           padding,
                               NALU_HYPRE_Int           level )
{
   // Get info from the amg data structure
   nalu_hypre_ParAMGData          *amg_data = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid       *compGrid = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_Int                 *CF_marker_array;

   nalu_hypre_AMGDDCompGridMatrix *A;
   nalu_hypre_CSRMatrix           *A_diag_original;
   nalu_hypre_CSRMatrix           *A_offd_original;

   nalu_hypre_AMGDDCompGridMatrix *P;
   nalu_hypre_CSRMatrix           *P_offd_original;

   nalu_hypre_AMGDDCompGridMatrix *R;
   nalu_hypre_CSRMatrix           *R_offd_original;

   nalu_hypre_ParCSRMatrix       **A_array;
   nalu_hypre_ParCSRMatrix       **P_array;
   nalu_hypre_ParCSRMatrix       **R_array;
   nalu_hypre_ParVector          **F_array;
   NALU_HYPRE_MemoryLocation       memory_location;

   NALU_HYPRE_Int                  avg_nnz_per_row;
   NALU_HYPRE_Int                  num_owned_nodes;
   NALU_HYPRE_Int                  max_nonowned;
   NALU_HYPRE_Int                  max_nonowned_diag_nnz;
   NALU_HYPRE_Int                  max_nonowned_offd_nnz;
   NALU_HYPRE_Int                  coarseIndexCounter, i;

   // Set some data
   A_array         = nalu_hypre_ParAMGDataAArray(amg_data);
   P_array         = nalu_hypre_ParAMGDataPArray(amg_data);
   R_array         = nalu_hypre_ParAMGDataRArray(amg_data);
   F_array         = nalu_hypre_ParAMGDataFArray(amg_data);
   A_diag_original = nalu_hypre_ParCSRMatrixDiag(A_array[level]);
   A_offd_original = nalu_hypre_ParCSRMatrixOffd(A_array[level]);
   if (nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[level])
   {
      CF_marker_array = nalu_hypre_IntArrayData(nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[level]);
   }
   else
   {
      CF_marker_array = NULL;
   }

   nalu_hypre_AMGDDCompGridLevel(compGrid)                = level;
   nalu_hypre_AMGDDCompGridFirstGlobalIndex(compGrid)     = nalu_hypre_ParVectorFirstIndex(F_array[level]);
   nalu_hypre_AMGDDCompGridLastGlobalIndex(compGrid)      = nalu_hypre_ParVectorLastIndex(F_array[level]);
   nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)        = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(
                                                                           F_array[level]));
   nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)     = nalu_hypre_CSRMatrixNumCols(A_offd_original);
   nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid) = 0;
   nalu_hypre_AMGDDCompGridMemoryLocation(compGrid)       = nalu_hypre_ParCSRMatrixMemoryLocation(
                                                          A_array[level]);
   memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);
   num_owned_nodes = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid);

   // !!! Check on how good a guess this is for eventual size of the nononwed dofs and nnz
   max_nonowned = 2 * (padding + nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data)) *
                  nalu_hypre_CSRMatrixNumCols(A_offd_original);
   avg_nnz_per_row = 0;
   if (nalu_hypre_CSRMatrixNumRows(A_diag_original))
   {
      avg_nnz_per_row = (NALU_HYPRE_Int) (nalu_hypre_CSRMatrixNumNonzeros(A_diag_original) / nalu_hypre_CSRMatrixNumRows(
                                        A_diag_original));
   }
   max_nonowned_diag_nnz = max_nonowned * avg_nnz_per_row;
   max_nonowned_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(A_offd_original);

   // Setup CompGridMatrix A
   A = nalu_hypre_AMGDDCompGridMatrixCreate();
   nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A) = A_diag_original;
   nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A) = A_offd_original;
   nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(A) = 0;
   nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A) = nalu_hypre_CSRMatrixCreate(max_nonowned,
                                                                    max_nonowned,
                                                                    max_nonowned_diag_nnz);
   nalu_hypre_CSRMatrixInitialize(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A));
   nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A) = nalu_hypre_CSRMatrixCreate(max_nonowned,
                                                                    num_owned_nodes,
                                                                    max_nonowned_offd_nnz);
   nalu_hypre_CSRMatrixInitialize(nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A));
   nalu_hypre_AMGDDCompGridA(compGrid) = A;
   nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                              max_nonowned_diag_nnz,
                                                                              memory_location);

   // Setup CompGridMatrix P and R if appropriate
   if (level != nalu_hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      P = nalu_hypre_AMGDDCompGridMatrixCreate();
      nalu_hypre_AMGDDCompGridMatrixOwnedDiag(P) = nalu_hypre_ParCSRMatrixDiag(P_array[level]);

      // Use original rowptr and data from P, but need to use new col indices (init to global index, then setup local indices later)
      P_offd_original = nalu_hypre_ParCSRMatrixOffd(P_array[level] );
      nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P) = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumRows(
                                                                       P_offd_original),
                                                                    nalu_hypre_CSRMatrixNumCols(P_offd_original),
                                                                    nalu_hypre_CSRMatrixNumNonzeros(P_offd_original));
      nalu_hypre_CSRMatrixI(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P))    = nalu_hypre_CSRMatrixI(P_offd_original);
      nalu_hypre_CSRMatrixData(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P)) = nalu_hypre_CSRMatrixData(P_offd_original);
      nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P))    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                                 nalu_hypre_CSRMatrixNumNonzeros(P_offd_original),
                                                                                 memory_location);

      // Initialize P owned offd col ind to their global indices
      for (i = 0; i < nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P)); i++)
      {
         nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(P))[i] = nalu_hypre_ParCSRMatrixColMapOffd(
                                                                         P_array[level])[ nalu_hypre_CSRMatrixJ(P_offd_original)[i] ];
      }

      nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(P) = 0;
      nalu_hypre_AMGDDCompGridMatrixOwnsOffdColIndices(P) = 1;
      nalu_hypre_AMGDDCompGridP(compGrid) = P;

      if (nalu_hypre_ParAMGDataRestriction(amg_data))
      {
         R = nalu_hypre_AMGDDCompGridMatrixCreate();
         nalu_hypre_AMGDDCompGridMatrixOwnedDiag(R) = nalu_hypre_ParCSRMatrixDiag(R_array[level]);

         // Use original rowptr and data from R, but need to use new col indices (init to global index, then setup local indices later)
         R_offd_original = nalu_hypre_ParCSRMatrixOffd(R_array[level]);
         nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R) = nalu_hypre_CSRMatrixCreate(nalu_hypre_CSRMatrixNumRows(
                                                                          R_offd_original),
                                                                       nalu_hypre_CSRMatrixNumCols(R_offd_original),
                                                                       nalu_hypre_CSRMatrixNumNonzeros(R_offd_original));
         nalu_hypre_CSRMatrixI(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R))    = nalu_hypre_CSRMatrixI(R_offd_original);
         nalu_hypre_CSRMatrixData(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R)) = nalu_hypre_CSRMatrixData(R_offd_original);
         nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R))    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                                    nalu_hypre_CSRMatrixNumNonzeros(R_offd_original),
                                                                                    memory_location);

         // Initialize R owned offd col ind to their global indices
         for (i = 0; i < nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R)); i++)
         {
            nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(R))[i] = nalu_hypre_ParCSRMatrixColMapOffd(
                                                                            R_array[level])[ nalu_hypre_CSRMatrixJ(R_offd_original)[i] ];
         }

         nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(R) = 0;
         nalu_hypre_AMGDDCompGridMatrixOwnsOffdColIndices(R) = 1;
         nalu_hypre_AMGDDCompGridR(compGrid) = R;
      }
   }

   // Allocate some extra arrays used during AMG-DD setup
   nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nonowned,
                                                                      memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nonowned,
                                                                      memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid)          = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nonowned,
                                                                      memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid)       = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nonowned,
                                                                      memory_location);

   // Initialize nonowned global indices, real marker, and the sort and invsort arrays
   for (i = 0; i < nalu_hypre_CSRMatrixNumCols(A_offd_original); i++)
   {
      nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[i] = nalu_hypre_ParCSRMatrixColMapOffd(
                                                                 A_array[level])[i];
      nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid)[i]          = i;
      nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid)[i]       = i;
      nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)[i]    =
         1; // NOTE: Assume that padding is at least 1, i.e. first layer of points are real
   }

   if (level != nalu_hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nonowned,
                                                                         memory_location);
      nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_owned_nodes,
                                                                         memory_location);

      // Setup the owned coarse indices
      if ( CF_marker_array )
      {
         coarseIndexCounter = 0;
         for (i = 0; i < num_owned_nodes; i++)
         {
            if ( CF_marker_array[i] > 0 )
            {
               nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = coarseIndexCounter++;
            }
            else
            {
               nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = -1;
            }
         }
      }
      else
      {
         for (i = 0; i < num_owned_nodes; i++)
         {
            nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = -1;
         }
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCompGridSetupRelax( nalu_hypre_ParAMGDDData *amgdd_data )
{
   nalu_hypre_ParAMGData      *amg_data = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid   *compGrid;

   nalu_hypre_CSRMatrix       *diag;
   nalu_hypre_CSRMatrix       *offd;

   NALU_HYPRE_Int              total_num_nodes;
   NALU_HYPRE_Int              cf_diag;
   NALU_HYPRE_Int              level, i, j;

   // Default to CFL1 Jacobi
   if (nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 0)
   {
      nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_Jacobi;
   }
   else if (nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 1)
   {
      nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_GaussSeidel;
   }
   else if (nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 2)
   {
      nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_OrderedGaussSeidel;
   }
   else if (nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 3)
   {
      nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_CFL1Jacobi;
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "WARNING: unknown AMGDD FAC relaxation type. Defaulting to CFL1 Jacobi.\n");
      nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_CFL1Jacobi;
      nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) = 3;
   }

   if (nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 3)
   {
      for (level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data); level < nalu_hypre_ParAMGDataNumLevels(amg_data);
           level++)
      {
         compGrid = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];

         // Calculate l1_norms
         total_num_nodes = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid) + nalu_hypre_AMGDDCompGridNumNonOwnedNodes(
                              compGrid);
         nalu_hypre_AMGDDCompGridL1Norms(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_num_nodes,
                                                              nalu_hypre_AMGDDCompGridMemoryLocation(compGrid));
         diag = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid));
         offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridA(compGrid));
         for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
         {
            cf_diag = nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[i];
            for (j = nalu_hypre_CSRMatrixI(diag)[i]; j < nalu_hypre_CSRMatrixI(diag)[i + 1]; j++)
            {
               if (nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[ nalu_hypre_CSRMatrixJ(diag)[j] ] == cf_diag)
               {
                  nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i] += nalu_hypre_cabs(nalu_hypre_CSRMatrixData(diag)[j]);
               }
            }
            for (j = nalu_hypre_CSRMatrixI(offd)[i]; j < nalu_hypre_CSRMatrixI(offd)[i + 1]; j++)
            {
               if (nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[ nalu_hypre_CSRMatrixJ(offd)[j] +
                                                                                         nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid) ] == cf_diag)
               {
                  nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i] += nalu_hypre_cabs(nalu_hypre_CSRMatrixData(offd)[j]);
               }
            }
         }

         diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid));
         offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridA(compGrid));
         for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid); i++)
         {
            cf_diag = nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(
                                                                    compGrid)];
            for (j = nalu_hypre_CSRMatrixI(diag)[i]; j < nalu_hypre_CSRMatrixI(diag)[i + 1]; j++)
            {
               if (nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[ nalu_hypre_CSRMatrixJ(diag)[j] +
                                                                                         nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid) ] == cf_diag)
               {
                  nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += nalu_hypre_cabs(
                                                                                                             nalu_hypre_CSRMatrixData(diag)[j]);
               }
            }
            for (j = nalu_hypre_CSRMatrixI(offd)[i]; j < nalu_hypre_CSRMatrixI(offd)[i + 1]; j++)
            {
               if (nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid)[ nalu_hypre_CSRMatrixJ(offd)[j]] == cf_diag)
               {
                  nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += nalu_hypre_cabs(
                                                                                                             nalu_hypre_CSRMatrixData(offd)[j]);
               }
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCompGridFinalize( nalu_hypre_ParAMGDDData *amgdd_data )
{
   nalu_hypre_ParAMGData     *amg_data     = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid **compGrid     = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   nalu_hypre_AMGDDCommPkg   *amgddCommPkg = nalu_hypre_ParAMGDDDataCommPkg(amgdd_data);
   NALU_HYPRE_Int             num_levels   = nalu_hypre_ParAMGDataNumLevels(amg_data);
   NALU_HYPRE_Int             start_level  = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);

   NALU_HYPRE_MemoryLocation  memory_location;
   nalu_hypre_CSRMatrix      *A_diag;
   nalu_hypre_CSRMatrix      *A_offd;

   NALU_HYPRE_Int             A_diag_nnz;
   NALU_HYPRE_Int             new_col_ind;
   NALU_HYPRE_Int            *new_A_diag_rowPtr;
   NALU_HYPRE_Int            *new_A_diag_colInd;
   NALU_HYPRE_Complex        *new_A_diag_data;

   NALU_HYPRE_Int             A_offd_nnz;
   NALU_HYPRE_Int            *new_A_offd_rowPtr;
   NALU_HYPRE_Int            *new_A_offd_colInd;
   NALU_HYPRE_Complex        *new_A_offd_data;

   NALU_HYPRE_Int             A_real_real_nnz;
   NALU_HYPRE_Int             A_real_ghost_nnz;

   nalu_hypre_CSRMatrix      *P_diag;
   nalu_hypre_CSRMatrix      *P_offd;

   NALU_HYPRE_Int             P_diag_nnz;
   NALU_HYPRE_Int            *new_P_diag_rowPtr;
   NALU_HYPRE_Int            *new_P_diag_colInd;
   NALU_HYPRE_Complex        *new_P_diag_data;

   NALU_HYPRE_Int             P_offd_nnz;
   NALU_HYPRE_Int            *new_P_offd_rowPtr;
   NALU_HYPRE_Int            *new_P_offd_colInd;
   NALU_HYPRE_Complex        *new_P_offd_data;

   nalu_hypre_CSRMatrix      *R_diag;
   nalu_hypre_CSRMatrix      *R_offd;

   NALU_HYPRE_Int             R_diag_nnz;
   NALU_HYPRE_Int            *new_R_diag_rowPtr;
   NALU_HYPRE_Int            *new_R_diag_colInd;
   NALU_HYPRE_Complex        *new_R_diag_data;

   NALU_HYPRE_Int             R_offd_nnz;
   NALU_HYPRE_Int            *new_R_offd_rowPtr;
   NALU_HYPRE_Int            *new_R_offd_colInd;
   NALU_HYPRE_Complex        *new_R_offd_data;

   NALU_HYPRE_Int             A_diag_cnt;
   NALU_HYPRE_Int             A_offd_cnt;
   NALU_HYPRE_Int             P_diag_cnt;
   NALU_HYPRE_Int             P_offd_cnt;
   NALU_HYPRE_Int             R_diag_cnt;
   NALU_HYPRE_Int             R_offd_cnt;
   NALU_HYPRE_Int             node_cnt;

   NALU_HYPRE_Int            *new_indices;
   NALU_HYPRE_Int             num_nonowned;
   NALU_HYPRE_Int             num_owned;
   NALU_HYPRE_Int             num_nonowned_real_nodes;
   NALU_HYPRE_Int             num_send_nodes;
   NALU_HYPRE_Int             num_send_procs;
   NALU_HYPRE_Int             new_num_send_nodes;
   NALU_HYPRE_Int             new_num_send_procs;
   NALU_HYPRE_Int             num_recv_nodes;
   NALU_HYPRE_Int             num_recv_procs;
   NALU_HYPRE_Int             new_num_recv_nodes;
   NALU_HYPRE_Int             new_num_recv_procs;
   NALU_HYPRE_Int             real_cnt, ghost_cnt;
   NALU_HYPRE_Int             proc, outer_level, level, i, j;

   // Post process to remove -1 entries from matrices and reorder so that extra nodes are [real, ghost]
   for (level = start_level; level < num_levels; level++)
   {
      memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid[level]);

      num_nonowned = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
      num_owned = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
      num_nonowned_real_nodes = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            num_nonowned_real_nodes++;
         }
      }
      nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid[level]) = num_nonowned_real_nodes;
      new_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned, memory_location);
      real_cnt = ghost_cnt = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_indices[i] = real_cnt++;
         }
         else
         {
            new_indices[i] = num_nonowned_real_nodes + ghost_cnt++;
         }
      }

      // Transform indices in send_flag and recv_map
      if (amgddCommPkg)
      {
         for (outer_level = start_level; outer_level < num_levels; outer_level++)
         {
            for (proc = 0; proc < nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level]; proc++)
            {
               num_send_nodes = nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
               new_num_send_nodes = 0;
               for (i = 0; i < num_send_nodes; i++)
               {
                  if (nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] >= num_owned)
                  {
                     nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][new_num_send_nodes++] =
                        new_indices[ nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] - num_owned ] +
                        num_owned;
                  }
                  else if (nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][new_num_send_nodes++] =
                        nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i];
                  }
               }
               nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level] = new_num_send_nodes;
            }

            for (proc = 0; proc < nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level]; proc++)
            {
               num_recv_nodes = nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
               new_num_recv_nodes = 0;
               for (i = 0; i < num_recv_nodes; i++)
               {
                  if (nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][new_num_recv_nodes++] =
                        new_indices[nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][i]];
                  }
               }
               nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level] = new_num_recv_nodes;
            }
         }
      }

      // Setup CF marker array
      nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level]) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                        num_owned + num_nonowned, memory_location);
      if (level != num_levels - 1)
      {
         // Setup CF marker array
         for (i = 0; i < num_owned; i++)
         {
            if (nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level])[i] >= 0)
            {
               nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i] = 1;
            }
            else
            {
               nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i] = -1;
            }
         }
         for (i = 0; i < num_nonowned; i++)
         {
            if (nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i] >= 0)
            {
               nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[new_indices[i] + num_owned] = 1;
            }
            else
            {
               nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[new_indices[i] + num_owned] = -1;
            }
         }
      }
      else
      {
         for (i = 0; i < num_owned + num_nonowned; i++)
         {
            nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i] = -1;
         }
      }

      // Reorder nonowned matrices
      A_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid[level]));
      A_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridA(compGrid[level]));

      A_diag_nnz        = nalu_hypre_CSRMatrixI(A_diag)[num_nonowned];
      new_A_diag_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
      new_A_diag_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, A_diag_nnz, memory_location);
      new_A_diag_data   = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, A_diag_nnz, memory_location);

      A_offd_nnz        = nalu_hypre_CSRMatrixI(A_offd)[num_nonowned];
      new_A_offd_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
      new_A_offd_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, A_offd_nnz, memory_location);
      new_A_offd_data   = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, A_offd_nnz, memory_location);

      A_real_real_nnz = 0;
      A_real_ghost_nnz = 0;

      if (level != num_levels - 1 && num_nonowned)
      {
         P_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridP(compGrid[level]));
         P_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridP(compGrid[level]));

         P_diag_nnz = nalu_hypre_CSRMatrixI(P_diag)[num_nonowned];
         new_P_diag_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
         new_P_diag_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, P_diag_nnz, memory_location);
         new_P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, P_diag_nnz, memory_location);

         P_offd_nnz = nalu_hypre_CSRMatrixI(P_offd)[num_nonowned];
         new_P_offd_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
         new_P_offd_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, P_offd_nnz, memory_location);
         new_P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, P_offd_nnz, memory_location);
      }
      if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         R_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level - 1]));
         R_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level - 1]));

         R_diag_nnz = nalu_hypre_CSRMatrixI(R_diag)[num_nonowned];
         new_R_diag_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
         new_R_diag_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, R_diag_nnz, memory_location);
         new_R_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, R_diag_nnz, memory_location);

         R_offd_nnz = nalu_hypre_CSRMatrixI(R_offd)[num_nonowned];
         new_R_offd_rowPtr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonowned + 1, memory_location);
         new_R_offd_colInd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, R_offd_nnz, memory_location);
         new_R_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, R_offd_nnz, memory_location);
      }

      A_diag_cnt = 0;
      A_offd_cnt = 0;
      P_diag_cnt = 0;
      P_offd_cnt = 0;
      R_diag_cnt = 0;
      R_offd_cnt = 0;
      node_cnt = 0;
      // Real nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = nalu_hypre_CSRMatrixI(A_diag)[i]; j < nalu_hypre_CSRMatrixI(A_diag)[i + 1]; j++)
            {
               if (nalu_hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  new_col_ind = new_indices[ nalu_hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_colInd[A_diag_cnt] = new_col_ind;
                  new_A_diag_data[A_diag_cnt] = nalu_hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
                  if (new_col_ind < num_nonowned_real_nodes)
                  {
                     A_real_real_nnz++;
                  }
                  else
                  {
                     A_real_ghost_nnz++;
                  }
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = nalu_hypre_CSRMatrixI(A_offd)[i]; j < nalu_hypre_CSRMatrixI(A_offd)[i + 1]; j++)
            {
               if (nalu_hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = nalu_hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = nalu_hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels - 1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = nalu_hypre_CSRMatrixI(P_diag)[i]; j < nalu_hypre_CSRMatrixI(P_diag)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = nalu_hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = nalu_hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = nalu_hypre_CSRMatrixI(P_offd)[i]; j < nalu_hypre_CSRMatrixI(P_offd)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = nalu_hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = nalu_hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               new_R_diag_rowPtr[node_cnt] = R_diag_cnt;
               for (j = nalu_hypre_CSRMatrixI(R_diag)[i]; j < nalu_hypre_CSRMatrixI(R_diag)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(R_diag)[j] >= 0)
                  {
                     new_R_diag_colInd[R_diag_cnt] = nalu_hypre_CSRMatrixJ(R_diag)[j];
                     new_R_diag_data[R_diag_cnt] = nalu_hypre_CSRMatrixData(R_diag)[j];
                     R_diag_cnt++;
                  }
               }
               new_R_offd_rowPtr[node_cnt] = R_offd_cnt;
               for (j = nalu_hypre_CSRMatrixI(R_offd)[i]; j < nalu_hypre_CSRMatrixI(R_offd)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(R_offd)[j] >= 0)
                  {
                     new_R_offd_colInd[R_offd_cnt] = nalu_hypre_CSRMatrixJ(R_offd)[j];
                     new_R_offd_data[R_offd_cnt] = nalu_hypre_CSRMatrixData(R_offd)[j];
                     R_offd_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      // Ghost nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (!nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = nalu_hypre_CSRMatrixI(A_diag)[i]; j < nalu_hypre_CSRMatrixI(A_diag)[i + 1]; j++)
            {
               if (nalu_hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  new_A_diag_colInd[A_diag_cnt] = new_indices[ nalu_hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_data[A_diag_cnt] = nalu_hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = nalu_hypre_CSRMatrixI(A_offd)[i]; j < nalu_hypre_CSRMatrixI(A_offd)[i + 1]; j++)
            {
               if (nalu_hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = nalu_hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = nalu_hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels - 1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = nalu_hypre_CSRMatrixI(P_diag)[i]; j < nalu_hypre_CSRMatrixI(P_diag)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = nalu_hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = nalu_hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = nalu_hypre_CSRMatrixI(P_offd)[i]; j < nalu_hypre_CSRMatrixI(P_offd)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = nalu_hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = nalu_hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               new_R_diag_rowPtr[node_cnt] = R_diag_cnt;
               for (j = nalu_hypre_CSRMatrixI(R_diag)[i]; j < nalu_hypre_CSRMatrixI(R_diag)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(R_diag)[j] >= 0)
                  {
                     new_R_diag_colInd[R_diag_cnt] = nalu_hypre_CSRMatrixJ(R_diag)[j];
                     new_R_diag_data[R_diag_cnt] = nalu_hypre_CSRMatrixData(R_diag)[j];
                     R_diag_cnt++;
                  }
               }
               new_R_offd_rowPtr[node_cnt] = R_offd_cnt;
               for (j = nalu_hypre_CSRMatrixI(R_offd)[i]; j < nalu_hypre_CSRMatrixI(R_offd)[i + 1]; j++)
               {
                  if (nalu_hypre_CSRMatrixJ(R_offd)[j] >= 0)
                  {
                     new_R_offd_colInd[R_offd_cnt] = nalu_hypre_CSRMatrixJ(R_offd)[j];
                     new_R_offd_data[R_offd_cnt] = nalu_hypre_CSRMatrixData(R_offd)[j];
                     R_offd_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      new_A_diag_rowPtr[num_nonowned] = A_diag_cnt;
      new_A_offd_rowPtr[num_nonowned] = A_offd_cnt;

      // Create these matrices, but don't initialize (will be allocated later if necessary)
      nalu_hypre_AMGDDCompGridMatrixRealReal(nalu_hypre_AMGDDCompGridA(compGrid[level])) = nalu_hypre_CSRMatrixCreate(
                                                                                    num_nonowned_real_nodes, num_nonowned_real_nodes, A_real_real_nnz);
      nalu_hypre_AMGDDCompGridMatrixRealGhost(nalu_hypre_AMGDDCompGridA(compGrid[level])) = nalu_hypre_CSRMatrixCreate(
                                                                                     num_nonowned_real_nodes, num_nonowned, A_real_ghost_nnz);


      if (level != num_levels - 1 && num_nonowned)
      {
         new_P_diag_rowPtr[num_nonowned] = P_diag_cnt;
         new_P_offd_rowPtr[num_nonowned] = P_offd_cnt;
      }
      if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         new_R_diag_rowPtr[num_nonowned] = R_diag_cnt;
         new_R_offd_rowPtr[num_nonowned] = R_offd_cnt;
      }

      // Fix up P col indices on finer level
      if (level != start_level && nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level - 1]))
      {
         P_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridP(compGrid[level - 1]));
         P_offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridP(compGrid[level - 1]));

         for (i = 0;
              i < nalu_hypre_CSRMatrixI(P_diag)[ nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level - 1]) ];
              i++)
         {
            nalu_hypre_CSRMatrixJ(P_diag)[i] = new_indices[ nalu_hypre_CSRMatrixJ(P_diag)[i] ];
         }
         // Also fix up owned offd col indices
         for (i = 0; i < nalu_hypre_CSRMatrixI(P_offd)[ nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level - 1]) ];
              i++)
         {
            nalu_hypre_CSRMatrixJ(P_offd)[i] = new_indices[ nalu_hypre_CSRMatrixJ(P_offd)[i] ];
         }
      }
      // Fix up R col indices on this level
      if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != num_levels - 1 && num_nonowned)
      {
         R_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level]));
         R_offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level]));

         for (i = 0;
              i < nalu_hypre_CSRMatrixI(R_diag)[ nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level + 1]) ];
              i++)
         {
            if (nalu_hypre_CSRMatrixJ(R_diag)[i] >= 0)
            {
               nalu_hypre_CSRMatrixJ(R_diag)[i] = new_indices[ nalu_hypre_CSRMatrixJ(R_diag)[i] ];
            }
         }
         // Also fix up owned offd col indices
         for (i = 0; i < nalu_hypre_CSRMatrixI(R_offd)[ nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level + 1]) ];
              i++)
         {
            if (nalu_hypre_CSRMatrixJ(R_offd)[i] >= 0)
            {
               nalu_hypre_CSRMatrixJ(R_offd)[i] = new_indices[ nalu_hypre_CSRMatrixJ(R_offd)[i] ];
            }
         }
      }

      // Clean up memory, deallocate old arrays and reset pointers to new arrays
      nalu_hypre_TFree(nalu_hypre_CSRMatrixI(A_diag), memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(A_diag), memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixData(A_diag), memory_location);
      nalu_hypre_CSRMatrixI(A_diag) = new_A_diag_rowPtr;
      nalu_hypre_CSRMatrixJ(A_diag) = new_A_diag_colInd;
      nalu_hypre_CSRMatrixData(A_diag) = new_A_diag_data;
      nalu_hypre_CSRMatrixNumRows(A_diag) = num_nonowned;
      nalu_hypre_CSRMatrixNumRownnz(A_diag) = num_nonowned;
      nalu_hypre_CSRMatrixNumCols(A_diag) = num_nonowned;
      nalu_hypre_CSRMatrixNumNonzeros(A_diag) = nalu_hypre_CSRMatrixI(A_diag)[num_nonowned];

      nalu_hypre_TFree(nalu_hypre_CSRMatrixI(A_offd), memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(A_offd), memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixData(A_offd), memory_location);
      nalu_hypre_CSRMatrixI(A_offd) = new_A_offd_rowPtr;
      nalu_hypre_CSRMatrixJ(A_offd) = new_A_offd_colInd;
      nalu_hypre_CSRMatrixData(A_offd) = new_A_offd_data;
      nalu_hypre_CSRMatrixNumRows(A_offd) = num_nonowned;
      nalu_hypre_CSRMatrixNumRownnz(A_offd) = num_nonowned;
      nalu_hypre_CSRMatrixNumCols(A_offd) = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
      nalu_hypre_CSRMatrixNumNonzeros(A_offd) = nalu_hypre_CSRMatrixI(A_offd)[num_nonowned];

      if (level != num_levels - 1 && num_nonowned)
      {
         P_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridP(compGrid[level]));
         P_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridP(compGrid[level]));

         nalu_hypre_TFree(nalu_hypre_CSRMatrixI(P_diag), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(P_diag), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixData(P_diag), memory_location);
         nalu_hypre_CSRMatrixI(P_diag) = new_P_diag_rowPtr;
         nalu_hypre_CSRMatrixJ(P_diag) = new_P_diag_colInd;
         nalu_hypre_CSRMatrixData(P_diag) = new_P_diag_data;
         nalu_hypre_CSRMatrixNumRows(P_diag) = num_nonowned;
         nalu_hypre_CSRMatrixNumRownnz(P_diag) = num_nonowned;
         nalu_hypre_CSRMatrixNumCols(P_diag) = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level + 1]);
         nalu_hypre_CSRMatrixNumNonzeros(P_diag) = nalu_hypre_CSRMatrixI(P_diag)[num_nonowned];

         nalu_hypre_TFree(nalu_hypre_CSRMatrixI(P_offd), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(P_offd), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixData(P_offd), memory_location);
         nalu_hypre_CSRMatrixI(P_offd) = new_P_offd_rowPtr;
         nalu_hypre_CSRMatrixJ(P_offd) = new_P_offd_colInd;
         nalu_hypre_CSRMatrixData(P_offd) = new_P_offd_data;
         nalu_hypre_CSRMatrixNumRows(P_offd) = num_nonowned;
         nalu_hypre_CSRMatrixNumRownnz(P_offd) = num_nonowned;
         nalu_hypre_CSRMatrixNumCols(P_offd) = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level + 1]);
         nalu_hypre_CSRMatrixNumNonzeros(P_offd) = nalu_hypre_CSRMatrixI(P_offd)[num_nonowned];

         nalu_hypre_CSRMatrixNumCols(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridP(
                                                                      compGrid[level]))) = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level + 1]);
      }
      if (nalu_hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         R_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level - 1]));
         R_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level - 1]));

         nalu_hypre_TFree(nalu_hypre_CSRMatrixI(R_diag), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(R_diag), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixData(R_diag), memory_location);
         nalu_hypre_CSRMatrixI(R_diag) = new_R_diag_rowPtr;
         nalu_hypre_CSRMatrixJ(R_diag) = new_R_diag_colInd;
         nalu_hypre_CSRMatrixData(R_diag) = new_R_diag_data;
         nalu_hypre_CSRMatrixNumRows(R_diag) = num_nonowned;
         nalu_hypre_CSRMatrixNumRownnz(R_diag) = num_nonowned;
         nalu_hypre_CSRMatrixNumCols(R_diag) = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level - 1]);
         nalu_hypre_CSRMatrixNumNonzeros(R_diag) = nalu_hypre_CSRMatrixI(R_diag)[num_nonowned];

         nalu_hypre_TFree(nalu_hypre_CSRMatrixI(R_offd), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(R_offd), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixData(R_offd), memory_location);
         nalu_hypre_CSRMatrixI(R_offd) = new_R_offd_rowPtr;
         nalu_hypre_CSRMatrixJ(R_offd) = new_R_offd_colInd;
         nalu_hypre_CSRMatrixData(R_offd) = new_R_offd_data;
         nalu_hypre_CSRMatrixNumRows(R_offd) = num_nonowned;
         nalu_hypre_CSRMatrixNumRownnz(R_offd) = num_nonowned;
         nalu_hypre_CSRMatrixNumCols(R_offd) = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level - 1]);
         nalu_hypre_CSRMatrixNumNonzeros(R_offd) = nalu_hypre_CSRMatrixI(R_offd)[num_nonowned];

         nalu_hypre_CSRMatrixNumCols(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridR(
                                                                      compGrid[level - 1]))) = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level - 1]);
      }

      // Setup comp grid vectors
      nalu_hypre_AMGDDCompGridU(compGrid[level]) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridU(compGrid[level])) = nalu_hypre_ParVectorLocalVector(
                                                                                 nalu_hypre_ParAMGDataUArray(amg_data)[level] );
      nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(nalu_hypre_AMGDDCompGridU(compGrid[level])) = 0;
      nalu_hypre_AMGDDCompGridVectorNumReal(nalu_hypre_AMGDDCompGridU(compGrid[level])) = num_nonowned_real_nodes;
      nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridU(compGrid[level])) = nalu_hypre_SeqVectorCreate(
                                                                                    num_nonowned);
      nalu_hypre_SeqVectorInitialize(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridU(compGrid[level])));

      nalu_hypre_AMGDDCompGridF(compGrid[level]) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridF(compGrid[level])) = nalu_hypre_ParVectorLocalVector(
                                                                                 nalu_hypre_ParAMGDataFArray(amg_data)[level] );
      nalu_hypre_AMGDDCompGridVectorOwnsOwnedVector(nalu_hypre_AMGDDCompGridF(compGrid[level])) = 0;
      nalu_hypre_AMGDDCompGridVectorNumReal(nalu_hypre_AMGDDCompGridF(compGrid[level])) = num_nonowned_real_nodes;
      nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridF(compGrid[level])) = nalu_hypre_SeqVectorCreate(
                                                                                    num_nonowned);
      nalu_hypre_SeqVectorInitialize(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridF(compGrid[level])));

      nalu_hypre_AMGDDCompGridTemp(compGrid[level]) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridTemp(compGrid[level]), num_owned,
                                          num_nonowned, num_nonowned_real_nodes);

      if (level < num_levels)
      {
         nalu_hypre_AMGDDCompGridS(compGrid[level]) = nalu_hypre_AMGDDCompGridVectorCreate();
         nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridS(compGrid[level]), num_owned, num_nonowned,
                                             num_nonowned_real_nodes);

         nalu_hypre_AMGDDCompGridT(compGrid[level]) = nalu_hypre_AMGDDCompGridVectorCreate();
         nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridT(compGrid[level]), num_owned, num_nonowned,
                                             num_nonowned_real_nodes);
      }

      // Free up arrays we no longer need
      if (nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]) = NULL;
      }
      if (nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]) = NULL;
      }
      if (nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid[level]) = NULL;
      }
      if (nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]))
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]), memory_location);
         nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]) = NULL;
      }
      nalu_hypre_TFree(new_indices, memory_location);
   }

   // Setup R = P^T if R not specified
   if (!nalu_hypre_ParAMGDataRestriction(amg_data))
   {
      for (level = start_level; level < num_levels - 1; level++)
      {
         // !!! TODO: if BoomerAMG explicitly stores R = P^T, use those matrices in
         nalu_hypre_AMGDDCompGridR(compGrid[level]) = nalu_hypre_AMGDDCompGridMatrixCreate();
         nalu_hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(nalu_hypre_AMGDDCompGridR(compGrid[level])) = 1;
         nalu_hypre_CSRMatrixTranspose(nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridP(compGrid[level])),
                                  &nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level])), 1);

         if (nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]))
         {
            nalu_hypre_CSRMatrixTranspose(nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridP(
                                                                              compGrid[level])),
                                     &nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level])), 1);
         }

         nalu_hypre_CSRMatrixTranspose(nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridP(compGrid[level])),
                                  &nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level])), 1);

         if (nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]))
         {
            nalu_hypre_CSRMatrixTranspose(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridP(
                                                                              compGrid[level])),
                                     &nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level])), 1);
         }
      }
   }

   // Finish up comm pkg
   if (amgddCommPkg)
   {
      for (outer_level = start_level; outer_level < num_levels; outer_level++)
      {
         num_send_procs = nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level];
         new_num_send_procs = 0;
         for (proc = 0; proc < num_send_procs; proc++)
         {
            nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs] = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs] +=
                  nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
            }
            if (nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs])
            {
               nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[outer_level][new_num_send_procs] =
                  nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[outer_level][proc];
               for (level = outer_level; level < num_levels; level++)
               {
                  nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][new_num_send_procs][level] =
                     nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
                  nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][new_num_send_procs][level] =
                     nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level];
               }
               new_num_send_procs++;
            }
         }

         // Free memory
         for (j = new_num_send_procs; j < num_send_procs; j++)
         {
            nalu_hypre_AMGDDCommPkgSendLevelDestroy(amgddCommPkg, outer_level, j);
         }

         // Update number of send processes
         nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level] = new_num_send_procs;

         num_recv_procs = nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level];
         new_num_recv_procs = 0;
         for (proc = 0; proc < num_recv_procs; proc++)
         {
            nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs] = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs] +=
                  nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
            }
            if (nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs])
            {
               nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[outer_level][new_num_recv_procs] =
                  nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[outer_level][proc];
               for (level = outer_level; level < num_levels; level++)
               {
                  nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][new_num_recv_procs][level] =
                     nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
                  nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][new_num_recv_procs][level] =
                     nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level];
               }
               new_num_recv_procs++;
            }
         }

         // Free memory
         for (j = new_num_recv_procs; j < num_recv_procs; j++)
         {
            nalu_hypre_AMGDDCommPkgRecvLevelDestroy(amgddCommPkg, outer_level, j);
         }

         // Update number of recv processes
         nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level] = new_num_recv_procs;
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridResize( nalu_hypre_AMGDDCompGrid *compGrid,
                           NALU_HYPRE_Int            new_size,
                           NALU_HYPRE_Int            need_coarse_info )
{
   // This function reallocates memory to hold nonowned info for the comp grid
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);
   nalu_hypre_CSRMatrix     *nonowned_diag;
   nalu_hypre_CSRMatrix     *nonowned_offd;
   NALU_HYPRE_Int            old_size = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);

   nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) = nalu_hypre_TReAlloc_v2(
                                                           nalu_hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid), NALU_HYPRE_Int, old_size, NALU_HYPRE_Int, new_size,
                                                           memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid) = nalu_hypre_TReAlloc_v2(
                                                        nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid), NALU_HYPRE_Int, old_size, NALU_HYPRE_Int, new_size,
                                                        memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedSort(compGrid) = nalu_hypre_TReAlloc_v2(nalu_hypre_AMGDDCompGridNonOwnedSort(
                                                                    compGrid), NALU_HYPRE_Int, old_size, NALU_HYPRE_Int, new_size, memory_location);
   nalu_hypre_AMGDDCompGridNonOwnedInvSort(compGrid) = nalu_hypre_TReAlloc_v2(nalu_hypre_AMGDDCompGridNonOwnedInvSort(
                                                                       compGrid), NALU_HYPRE_Int, old_size, NALU_HYPRE_Int, new_size, memory_location);

   nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid));
   nonowned_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridA(compGrid));
   nalu_hypre_CSRMatrixResize(nonowned_diag, new_size, new_size, nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag));
   nalu_hypre_CSRMatrixResize(nonowned_offd, new_size, nalu_hypre_CSRMatrixNumCols(nonowned_offd),
                         nalu_hypre_CSRMatrixNumNonzeros(nonowned_offd));

   if (need_coarse_info)
   {
      nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid) = nalu_hypre_TReAlloc_v2(
                                                              nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid), NALU_HYPRE_Int, old_size, NALU_HYPRE_Int, new_size,
                                                              memory_location);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AMGDDCompGridSetupLocalIndices( nalu_hypre_AMGDDCompGrid **compGrid,
                                      NALU_HYPRE_Int            *nodes_added_on_level,
                                      NALU_HYPRE_Int         ****recv_map,
                                      NALU_HYPRE_Int             num_recv_procs,
                                      NALU_HYPRE_Int           **A_tmp_info,
                                      NALU_HYPRE_Int             current_level,
                                      NALU_HYPRE_Int             num_levels )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   nalu_hypre_AMGDDCompGridMatrix   *A = nalu_hypre_AMGDDCompGridA(compGrid[current_level]);
   nalu_hypre_CSRMatrix             *owned_offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A);
   nalu_hypre_CSRMatrix             *nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   nalu_hypre_CSRMatrix             *nonowned_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // On current_level, need to deal with A_tmp_info
   NALU_HYPRE_Int     row = nalu_hypre_CSRMatrixNumCols(owned_offd) + 1;
   NALU_HYPRE_Int     diag_rowptr = nalu_hypre_CSRMatrixI(nonowned_diag)[ nalu_hypre_CSRMatrixNumCols(owned_offd) ];
   NALU_HYPRE_Int     offd_rowptr = nalu_hypre_CSRMatrixI(nonowned_offd)[ nalu_hypre_CSRMatrixNumCols(owned_offd) ];

   NALU_HYPRE_Int     level, proc, i, j, cnt;
   NALU_HYPRE_Int     global_index, local_index, coarse_index;
   NALU_HYPRE_Int     remaining_dofs;
   NALU_HYPRE_Int     row_size;
   NALU_HYPRE_Int     incoming_index;
   NALU_HYPRE_Int     num_missing_col_ind;
   NALU_HYPRE_Int     is_real;

   for (proc = 0; proc < num_recv_procs; proc++)
   {
      cnt = 0;
      remaining_dofs = A_tmp_info[proc][cnt++];

      for (i = 0; i < remaining_dofs; i++)
      {
         row_size = A_tmp_info[proc][cnt++];
         for (j = 0; j < row_size; j++)
         {
            incoming_index = A_tmp_info[proc][cnt++];

            // Incoming is a global index (could be owned or nonowned)
            if (incoming_index < 0)
            {
               incoming_index = -(incoming_index + 1);
               // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
               if ( incoming_index >= nalu_hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]) &&
                    incoming_index <= nalu_hypre_AMGDDCompGridLastGlobalIndex(compGrid[current_level] ))
               {
                  // Add to offd
                  if (offd_rowptr >= nalu_hypre_CSRMatrixNumNonzeros(nonowned_offd))
                  {
                     nalu_hypre_CSRMatrixResize(nonowned_offd,
                                           nalu_hypre_CSRMatrixNumRows(nonowned_offd),
                                           nalu_hypre_CSRMatrixNumCols(nonowned_offd),
                                           (NALU_HYPRE_Int)nalu_hypre_ceil(1.5 * nalu_hypre_CSRMatrixNumNonzeros(nonowned_offd)));
                  }
                  nalu_hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index -
                                                                   nalu_hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]);
               }
               else
               {
                  // Add to diag (global index, not in buffer, so need to do local binary search)
                  if (diag_rowptr >= nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag))
                  {
                     nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) =
                        nalu_hypre_TReAlloc_v2(nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]),
                                          NALU_HYPRE_Int,
                                          nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag),
                                          NALU_HYPRE_Int,
                                          (NALU_HYPRE_Int)nalu_hypre_ceil(1.5 * nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag)),
                                          nalu_hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
                     nalu_hypre_CSRMatrixResize(nonowned_diag, nalu_hypre_CSRMatrixNumRows(nonowned_diag),
                                           nalu_hypre_CSRMatrixNumCols(nonowned_diag),
                                           (NALU_HYPRE_Int)nalu_hypre_ceil(1.5 * nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag)));
                  }
                  // If we dof not found in comp grid, then mark this as a missing connection
                  nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(
                     compGrid[current_level])[ nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid[current_level])++ ] =
                        diag_rowptr;
                  nalu_hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index + 1);
               }
            }
            // Incoming is an index to dofs within the buffer (by construction, nonowned)
            else
            {
               // Add to diag (index is within buffer, so we can directly go to local index)
               if (diag_rowptr >= nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag))
               {
                  nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) =
                     nalu_hypre_TReAlloc_v2(nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]),
                                       NALU_HYPRE_Int,
                                       nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag),
                                       NALU_HYPRE_Int,
                                       (NALU_HYPRE_Int)nalu_hypre_ceil(1.5 * nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag)),
                                       nalu_hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));

                  nalu_hypre_CSRMatrixResize(nonowned_diag,
                                        nalu_hypre_CSRMatrixNumRows(nonowned_diag),
                                        nalu_hypre_CSRMatrixNumCols(nonowned_diag),
                                        (NALU_HYPRE_Int)nalu_hypre_ceil(1.5 * nalu_hypre_CSRMatrixNumNonzeros(nonowned_diag)));
               }
               local_index = recv_map[current_level][proc][current_level][ incoming_index ];
               if (local_index < 0)
               {
                  local_index = -(local_index + 1);
               }
               nalu_hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = local_index - nalu_hypre_AMGDDCompGridNumOwnedNodes(
                                                                   compGrid[current_level]);
            }
         }

         // Update row pointers
         nalu_hypre_CSRMatrixI(nonowned_offd)[ row ] = offd_rowptr;
         nalu_hypre_CSRMatrixI(nonowned_diag)[ row ] = diag_rowptr;
         row++;
      }
      nalu_hypre_TFree(A_tmp_info[proc], nalu_hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
   }
   nalu_hypre_TFree(A_tmp_info, NALU_HYPRE_MEMORY_HOST);

   // Loop over levels from current to coarsest
   for (level = current_level; level < num_levels; level++)
   {
      A = nalu_hypre_AMGDDCompGridA(compGrid[level]);
      nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);

      // If we have added nodes on this level
      if (nodes_added_on_level[level])
      {
         // Look for missing col ind connections
         num_missing_col_ind = nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid[level]);
         nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid[level]) = 0;
         for (i = 0; i < num_missing_col_ind; i++)
         {
            j = nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level])[i];
            global_index = nalu_hypre_CSRMatrixJ(nonowned_diag)[ j ];
            global_index = -(global_index + 1);
            local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level], global_index);
            // If we dof not found in comp grid, then mark this as a missing connection
            if (local_index == -1)
            {
               local_index = -(global_index + 1);
               nalu_hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(
                  compGrid[level])[ nalu_hypre_AMGDDCompGridNumMissingColIndices(compGrid[level])++ ] = j;
            }
            nalu_hypre_CSRMatrixJ(nonowned_diag)[ j ] = local_index;
         }
      }

      // if we are not on the coarsest level
      if (level != num_levels - 1)
      {
         // loop over indices of non-owned nodes on this level
         // No guarantee that previous ghost dofs converted to real dofs have coarse local indices setup...
         // Thus we go over all non-owned dofs here instead of just the added ones, but we only setup coarse local index where necessary.
         // NOTE: can't use nodes_added_on_level here either because real overwritten by ghost doesn't count as added node (so you can miss setting these up)
         for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]); i++)
         {
            // fix up the coarse local indices
            coarse_index = nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i];
            is_real = nalu_hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i];

            // setup coarse local index if necessary
            if (coarse_index < -1 && is_real)
            {
               coarse_index = -(coarse_index + 2); // Map back to regular global index
               local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level + 1], coarse_index);
               nalu_hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i] = local_index;
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCompGridSetupLocalIndicesP( nalu_hypre_ParAMGDDData *amgdd_data )
{
   nalu_hypre_ParAMGData      *amg_data    = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid  **compGrid    = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   NALU_HYPRE_Int              num_levels  = nalu_hypre_ParAMGDataNumLevels(amg_data);
   NALU_HYPRE_Int              start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);

   nalu_hypre_CSRMatrix       *owned_offd;
   nalu_hypre_CSRMatrix       *nonowned_diag;

   NALU_HYPRE_Int              i, level;
   NALU_HYPRE_Int              local_index;

   for (level = start_level; level < num_levels - 1; level++)
   {
      // Setup owned offd col indices
      owned_offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridP(compGrid[level]));

      for (i = 0; i < nalu_hypre_CSRMatrixI(owned_offd)[nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level])];
           i++)
      {
         local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level + 1],
                                                                 nalu_hypre_CSRMatrixJ(owned_offd)[i]);
         if (local_index == -1)
         {
            nalu_hypre_CSRMatrixJ(owned_offd)[i] = -(nalu_hypre_CSRMatrixJ(owned_offd)[i] + 1);
         }
         else
         {
            nalu_hypre_CSRMatrixJ(owned_offd)[i] = local_index;
         }
      }

      // Setup nonowned diag col indices
      nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridP(compGrid[level]));

      for (i = 0;
           i < nalu_hypre_CSRMatrixI(nonowned_diag)[nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level])]; i++)
      {
         local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level + 1],
                                                                 nalu_hypre_CSRMatrixJ(nonowned_diag)[i]);
         if (local_index == -1)
         {
            nalu_hypre_CSRMatrixJ(nonowned_diag)[i] = -(nalu_hypre_CSRMatrixJ(nonowned_diag)[i] + 1);
         }
         else
         {
            nalu_hypre_CSRMatrixJ(nonowned_diag)[i] = local_index;
         }
      }
   }

   if (nalu_hypre_ParAMGDataRestriction(amg_data))
   {
      for (level = start_level; level < num_levels - 1; level++)
      {
         // Setup owned offd col indices
         owned_offd = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridR(compGrid[level]));

         for (i = 0; i < nalu_hypre_CSRMatrixI(owned_offd)[nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level + 1])];
              i++)
         {
            local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level],
                                                                    nalu_hypre_CSRMatrixJ(owned_offd)[i]);
            if (local_index == -1)
            {
               nalu_hypre_CSRMatrixJ(owned_offd)[i] = -(nalu_hypre_CSRMatrixJ(owned_offd)[i] + 1);
            }
            else
            {
               nalu_hypre_CSRMatrixJ(owned_offd)[i] = local_index;
            }
         }

         // Setup nonowned diag col indices
         nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridR(compGrid[level]));

         for (i = 0;
              i < nalu_hypre_CSRMatrixI(nonowned_diag)[nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level + 1])]; i++)
         {
            local_index = nalu_hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level],
                                                                    nalu_hypre_CSRMatrixJ(nonowned_diag)[i]);
            if (local_index == -1)
            {
               nalu_hypre_CSRMatrixJ(nonowned_diag)[i] = -(nalu_hypre_CSRMatrixJ(nonowned_diag)[i] + 1);
            }
            else
            {
               nalu_hypre_CSRMatrixJ(nonowned_diag)[i] = local_index;
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

nalu_hypre_AMGDDCommPkg* nalu_hypre_AMGDDCommPkgCreate(NALU_HYPRE_Int num_levels)
{
   nalu_hypre_AMGDDCommPkg   *amgddCommPkg;

   amgddCommPkg = nalu_hypre_CTAlloc(nalu_hypre_AMGDDCommPkg, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg) = num_levels;

   nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg)      = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,   num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)      = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,   num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,   num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,   num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)   = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)   = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)       = nalu_hypre_CTAlloc(NALU_HYPRE_Int ***, num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)        = nalu_hypre_CTAlloc(NALU_HYPRE_Int ***, num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)  = nalu_hypre_CTAlloc(NALU_HYPRE_Int ***, num_levels,
                                                                  NALU_HYPRE_MEMORY_HOST);

   return amgddCommPkg;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCommPkgSendLevelDestroy( nalu_hypre_AMGDDCommPkg *amgddCommPkg,
                                              NALU_HYPRE_Int           level,
                                              NALU_HYPRE_Int           proc )
{
   NALU_HYPRE_Int  k;

   if (nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg))
   {
      for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
      {
         if (nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[level][proc][k])
         {
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[level][proc][k],
                        NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree( nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[level][proc],
                   NALU_HYPRE_MEMORY_HOST );
   }

   if (nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg))
   {
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[level][proc],
                  NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCommPkgRecvLevelDestroy( nalu_hypre_AMGDDCommPkg *amgddCommPkg,
                                              NALU_HYPRE_Int           level,
                                              NALU_HYPRE_Int           proc )
{
   NALU_HYPRE_Int  k;

   if (nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg))
   {
      for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
      {
         if (nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[level][proc][k])
         {
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[level][proc][k],
                        NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[level][proc], NALU_HYPRE_MEMORY_HOST);
   }

   if (nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg))
   {
      for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
      {
         if (nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[level][proc][k])
         {
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[level][proc][k],
                        NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[level][proc],
                  NALU_HYPRE_MEMORY_HOST);
   }

   if (nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg))
   {
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[level][proc],
                  NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_AMGDDCommPkgDestroy ( nalu_hypre_AMGDDCommPkg *amgddCommPkg )
{
   NALU_HYPRE_Int  i, j, k;

   if ( nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendProcs(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvProcs(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[i]; j++)
         {
            for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
            {
               if (nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j][k])
               {
                  nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j][k], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgSendFlag(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[i]; j++)
         {
            for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
            {
               if (nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j][k])
               {
                  nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j][k], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvMap(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[i]; j++)
         {
            for (k = 0; k < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
            {
               if (nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[i][j][k])
               {
                  nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[i][j][k], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgRecvRedMarker(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[i]; j++)
         {
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg) )
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[i]; j++)
         {
            nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg) )
   {
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   if ( nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg) )
   {
      nalu_hypre_TFree(nalu_hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg), NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(amgddCommPkg, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
