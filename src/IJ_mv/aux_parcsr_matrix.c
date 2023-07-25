/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_AuxParCSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_IJ_mv.h"
#include "aux_parcsr_matrix.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParCSRMatrixCreate( nalu_hypre_AuxParCSRMatrix **aux_matrix,
                             NALU_HYPRE_Int               local_num_rows,
                             NALU_HYPRE_Int               local_num_cols,
                             NALU_HYPRE_Int              *sizes )
{
   nalu_hypre_AuxParCSRMatrix  *matrix;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_AuxParCSRMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rows;
   nalu_hypre_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;

   nalu_hypre_AuxParCSRMatrixRowSpace(matrix) = sizes;

   /* set defaults */
   nalu_hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   nalu_hypre_AuxParCSRMatrixMaxOffProcElmts(matrix) = 0;
   nalu_hypre_AuxParCSRMatrixCurrentOffProcElmts(matrix) = 0;
   nalu_hypre_AuxParCSRMatrixOffProcIIndx(matrix) = 0;
   nalu_hypre_AuxParCSRMatrixRownnz(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixRowLength(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixAuxJ(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixAuxData(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixIndxOffd(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixDiagSizes(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixOffdSizes(matrix) = NULL;
   /* stash for setting or adding on/off-proc values */
   nalu_hypre_AuxParCSRMatrixOffProcI(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixOffProcJ(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixOffProcData(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix) = NALU_HYPRE_MEMORY_HOST;
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_AuxParCSRMatrixMaxStackElmts(matrix) = 0;
   nalu_hypre_AuxParCSRMatrixCurrentStackElmts(matrix) = 0;
   nalu_hypre_AuxParCSRMatrixStackI(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixStackJ(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixStackData(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixStackSorA(matrix) = NULL;
   nalu_hypre_AuxParCSRMatrixUsrOnProcElmts(matrix) = -1;
   nalu_hypre_AuxParCSRMatrixUsrOffProcElmts(matrix) = -1;
   nalu_hypre_AuxParCSRMatrixInitAllocFactor(matrix) = 5;
   nalu_hypre_AuxParCSRMatrixGrowFactor(matrix) = 2;
#endif

   *aux_matrix = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParCSRMatrixDestroy( nalu_hypre_AuxParCSRMatrix *matrix )
{
   NALU_HYPRE_Int   num_rownnz;
   NALU_HYPRE_Int   num_rows;
   NALU_HYPRE_Int  *rownnz;
   NALU_HYPRE_Int   i;

   if (matrix)
   {
      rownnz     = nalu_hypre_AuxParCSRMatrixRownnz(matrix);
      num_rownnz = nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix);
      num_rows = nalu_hypre_AuxParCSRMatrixLocalNumRows(matrix);

      if (nalu_hypre_AuxParCSRMatrixAuxJ(matrix))
      {
         if (nalu_hypre_AuxParCSRMatrixRownnz(matrix))
         {
            for (i = 0; i < num_rownnz; i++)
            {
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz[i]], NALU_HYPRE_MEMORY_HOST);
            }
         }
         else
         {
            for (i = 0; i < num_rows; i++)
            {
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxJ(matrix)[i], NALU_HYPRE_MEMORY_HOST);
            }
         }

         nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxJ(matrix), NALU_HYPRE_MEMORY_HOST);
      }

      if (nalu_hypre_AuxParCSRMatrixAuxData(matrix))
      {
         if (nalu_hypre_AuxParCSRMatrixRownnz(matrix))
         {
            for (i = 0; i < num_rownnz; i++)
            {
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix)[rownnz[i]], NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix), NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            for (i = 0; i < num_rows; i++)
            {
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix)[i], NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix), NALU_HYPRE_MEMORY_HOST);
         }
      }

      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixRownnz(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixRowLength(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixRowSpace(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixIndxDiag(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixIndxOffd(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixDiagSizes(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixOffdSizes(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixOffProcI(matrix),    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixOffProcJ(matrix),    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixOffProcData(matrix), NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_GPU)
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixStackI(matrix),    nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix));
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixStackJ(matrix),    nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix));
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixStackData(matrix), nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix));
      nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixStackSorA(matrix), nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix));
#endif

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParCSRMatrixSetRownnz
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AuxParCSRMatrixSetRownnz( nalu_hypre_AuxParCSRMatrix *matrix )
{
   NALU_HYPRE_Int   local_num_rows = nalu_hypre_AuxParCSRMatrixLocalNumRows(matrix);
   NALU_HYPRE_Int  *row_space      = nalu_hypre_AuxParCSRMatrixRowSpace(matrix);
   NALU_HYPRE_Int   num_rownnz_old = nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix);
   NALU_HYPRE_Int  *rownnz_old     = nalu_hypre_AuxParCSRMatrixRownnz(matrix);
   NALU_HYPRE_Int  *rownnz;

   NALU_HYPRE_Int   i, ii, local_num_rownnz;

   /* Count number of nonzero rows */
   local_num_rownnz = 0;
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:local_num_rownnz) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < local_num_rows; i++)
   {
      if (row_space[i] > 0)
      {
         local_num_rownnz++;
      }
   }

   if (local_num_rownnz != local_num_rows)
   {
      rownnz = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rownnz, NALU_HYPRE_MEMORY_HOST);

      /* Find nonzero rows */
      local_num_rownnz = 0;
      for (i = 0; i < local_num_rows; i++)
      {
         if (row_space[i] > 0)
         {
            rownnz[local_num_rownnz++] = i;
         }
      }

      /* Free memory if necessary */
      if (rownnz_old && rownnz && (local_num_rownnz < num_rownnz_old))
      {
         ii = 0;
         for (i = 0; i < num_rownnz_old; i++)
         {
            if (rownnz_old[i] == rownnz[ii])
            {
               ii++;
            }
            else
            {
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz_old[i]], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix)[rownnz_old[i]], NALU_HYPRE_MEMORY_HOST);
            }

            if (ii == local_num_rownnz)
            {
               i = i + 1;
               for (; i < num_rownnz_old; i++)
               {
                  nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz_old[i]],
                              NALU_HYPRE_MEMORY_HOST);
                  nalu_hypre_TFree(nalu_hypre_AuxParCSRMatrixAuxData(matrix)[rownnz_old[i]],
                              NALU_HYPRE_MEMORY_HOST);
               }
               break;
            }
         }
      }
      nalu_hypre_TFree(rownnz_old, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rownnz;
      nalu_hypre_AuxParCSRMatrixRownnz(matrix) = rownnz;
   }
   else
   {
      nalu_hypre_TFree(rownnz_old, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rows;
      nalu_hypre_AuxParCSRMatrixRownnz(matrix) = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AuxParCSRMatrixInitialize_v2
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_AuxParCSRMatrixInitialize_v2( nalu_hypre_AuxParCSRMatrix *matrix,
                                    NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int local_num_rows = nalu_hypre_AuxParCSRMatrixLocalNumRows(matrix);
   NALU_HYPRE_Int max_off_proc_elmts = nalu_hypre_AuxParCSRMatrixMaxOffProcElmts(matrix);

   nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix) = memory_location;

   if (local_num_rows < 0)
   {
      return -1;
   }

   if (local_num_rows == 0)
   {
      return 0;
   }

#if defined(NALU_HYPRE_USING_GPU)
   if (memory_location != NALU_HYPRE_MEMORY_HOST)
   {
      /* GPU assembly */
      nalu_hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   }
   else
#endif
   {
      /* CPU assembly */
      /* allocate stash for setting or adding off processor values */
      if (max_off_proc_elmts > 0)
      {
         nalu_hypre_AuxParCSRMatrixOffProcI(matrix)    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, 2 * max_off_proc_elmts,
                                                                  NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_AuxParCSRMatrixOffProcJ(matrix)    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,   max_off_proc_elmts,
                                                                  NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_AuxParCSRMatrixOffProcData(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  max_off_proc_elmts,
                                                                  NALU_HYPRE_MEMORY_HOST);
      }

      if (nalu_hypre_AuxParCSRMatrixNeedAux(matrix))
      {
         NALU_HYPRE_Int      *row_space = nalu_hypre_AuxParCSRMatrixRowSpace(matrix);
         NALU_HYPRE_Int      *rownnz    = nalu_hypre_AuxParCSRMatrixRownnz(matrix);
         NALU_HYPRE_BigInt  **aux_j     = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt *,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
         NALU_HYPRE_Complex **aux_data  = nalu_hypre_CTAlloc(NALU_HYPRE_Complex *, local_num_rows, NALU_HYPRE_MEMORY_HOST);

         NALU_HYPRE_Int       local_num_rownnz;
         NALU_HYPRE_Int       i, ii;

         if (row_space)
         {
            /* Count number of nonzero rows */
            local_num_rownnz = 0;
            for (i = 0; i < local_num_rows; i++)
            {
               if (row_space[i] > 0)
               {
                  local_num_rownnz++;
               }
            }

            if (local_num_rownnz != local_num_rows)
            {
               rownnz = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rownnz, NALU_HYPRE_MEMORY_HOST);

               /* Find nonzero rows */
               local_num_rownnz = 0;
               for (i = 0; i < local_num_rows; i++)
               {
                  if (row_space[i] > 0)
                  {
                     rownnz[local_num_rownnz++] = i;
                  }
               }

               nalu_hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rownnz;
               nalu_hypre_AuxParCSRMatrixRownnz(matrix) = rownnz;
            }
         }

         if (!nalu_hypre_AuxParCSRMatrixRowLength(matrix))
         {
            nalu_hypre_AuxParCSRMatrixRowLength(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows,
                                                                   NALU_HYPRE_MEMORY_HOST);
         }

         if (row_space)
         {
            if (local_num_rownnz != local_num_rows)
            {
               for (i = 0; i < local_num_rownnz; i++)
               {
                  ii = rownnz[i];
                  aux_j[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, row_space[ii], NALU_HYPRE_MEMORY_HOST);
                  aux_data[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, row_space[ii], NALU_HYPRE_MEMORY_HOST);
               }
            }
            else
            {
               for (i = 0; i < local_num_rows; i++)
               {
                  aux_j[i] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, row_space[i], NALU_HYPRE_MEMORY_HOST);
                  aux_data[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, row_space[i], NALU_HYPRE_MEMORY_HOST);
               }
            }
         }
         else
         {
            row_space = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < local_num_rows; i++)
            {
               row_space[i] = 30;
               aux_j[i] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, 30, NALU_HYPRE_MEMORY_HOST);
               aux_data[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, 30, NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
         }
         nalu_hypre_AuxParCSRMatrixAuxJ(matrix) = aux_j;
         nalu_hypre_AuxParCSRMatrixAuxData(matrix) = aux_data;
      }
      else
      {
         nalu_hypre_AuxParCSRMatrixIndxDiag(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_AuxParCSRMatrixIndxOffd(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_AuxParCSRMatrixInitialize(nalu_hypre_AuxParCSRMatrix *matrix)
{
   if (matrix)
   {
      return nalu_hypre_AuxParCSRMatrixInitialize_v2(matrix, nalu_hypre_AuxParCSRMatrixMemoryLocation(matrix));
   }

   return -2;
}
