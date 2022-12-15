/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_IJ_mv.h"
#include "_nalu_hypre_sstruct_ls.h"

#include "nd1_amge_interpolation.h"

/*
  Assume that we are given a fine and coarse topology and the
  coarse degrees of freedom (DOFs) have been chosen. Assume also,
  that the global interpolation matrix dof_DOF has a prescribed
  nonzero pattern. Then, the fine degrees of freedom can be split
  into 4 groups (here "i" stands for "interior"):

  NODEidof - dofs which are interpolated only from the DOF
             in one coarse vertex
  EDGEidof - dofs which are interpolated only from the DOFs
             in one coarse edge
  FACEidof - dofs which are interpolated only from the DOFs
             in one coarse face
  ELEMidof - dofs which are interpolated only from the DOFs
             in one coarse element

  The interpolation operator dof_DOF can be build in 4 steps, by
  consequently filling-in the rows corresponding to the above groups.
  The code below uses harmonic extension to extend the interpolation
  from one group to the next.
*/
NALU_HYPRE_Int nalu_hypre_ND1AMGeInterpolation (nalu_hypre_ParCSRMatrix       * Aee,
                                      nalu_hypre_ParCSRMatrix       * ELEM_idof,
                                      nalu_hypre_ParCSRMatrix       * FACE_idof,
                                      nalu_hypre_ParCSRMatrix       * EDGE_idof,
                                      nalu_hypre_ParCSRMatrix       * ELEM_FACE,
                                      nalu_hypre_ParCSRMatrix       * ELEM_EDGE,
                                      NALU_HYPRE_Int                  num_OffProcRows,
                                      nalu_hypre_MaxwellOffProcRow ** OffProcRows,
                                      nalu_hypre_IJMatrix           * IJ_dof_DOF)
{
   NALU_HYPRE_Int ierr = 0;

   NALU_HYPRE_Int  i, j;
   NALU_HYPRE_BigInt  big_k;
   NALU_HYPRE_BigInt *offproc_rnums;
   NALU_HYPRE_Int *swap;

   nalu_hypre_ParCSRMatrix * dof_DOF = (nalu_hypre_ParCSRMatrix *)nalu_hypre_IJMatrixObject(IJ_dof_DOF);
   nalu_hypre_ParCSRMatrix * ELEM_DOF = ELEM_EDGE;
   nalu_hypre_ParCSRMatrix * ELEM_FACEidof;
   nalu_hypre_ParCSRMatrix * ELEM_EDGEidof;
   nalu_hypre_CSRMatrix *A, *P;
   NALU_HYPRE_Int numELEM = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(ELEM_EDGE));

   NALU_HYPRE_Int getrow_ierr;
   NALU_HYPRE_Int three_dimensional_problem;

   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(Aee);
   NALU_HYPRE_Int      myproc;

   nalu_hypre_MPI_Comm_rank(comm, &myproc);

#if 0
   nalu_hypre_IJMatrix * ij_dof_DOF = nalu_hypre_CTAlloc(nalu_hypre_IJMatrix,  1, NALU_HYPRE_MEMORY_HOST);
   /* Convert dof_DOF to IJ matrix, so we can use AddToValues */
   nalu_hypre_IJMatrixComm(ij_dof_DOF) = nalu_hypre_ParCSRMatrixComm(dof_DOF);
   nalu_hypre_IJMatrixRowPartitioning(ij_dof_DOF) =
      nalu_hypre_ParCSRMatrixRowStarts(dof_DOF);
   nalu_hypre_IJMatrixColPartitioning(ij_dof_DOF) =
      nalu_hypre_ParCSRMatrixColStarts(dof_DOF);
   nalu_hypre_IJMatrixObject(ij_dof_DOF) = dof_DOF;
   nalu_hypre_IJMatrixAssembleFlag(ij_dof_DOF) = 1;
#endif

   /* sort the offproc rows to get quicker comparison for later */
   if (num_OffProcRows)
   {
      offproc_rnums = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_OffProcRows, NALU_HYPRE_MEMORY_HOST);
      swap         = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_OffProcRows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_OffProcRows; i++)
      {
         offproc_rnums[i] = (OffProcRows[i] -> row);
         swap[i]         = i;
      }
   }

   if (num_OffProcRows > 1)
   {
      nalu_hypre_BigQsortbi(offproc_rnums, swap, 0, num_OffProcRows - 1);
   }

   if (FACE_idof == EDGE_idof)
   {
      three_dimensional_problem = 0;
   }
   else
   {
      three_dimensional_problem = 1;
   }

   /* ELEM_FACEidof = ELEM_FACE x FACE_idof */
   if (three_dimensional_problem)
   {
      ELEM_FACEidof = nalu_hypre_ParMatmul(ELEM_FACE, FACE_idof);
   }

   /* ELEM_EDGEidof = ELEM_EDGE x EDGE_idof */
   ELEM_EDGEidof = nalu_hypre_ParMatmul(ELEM_EDGE, EDGE_idof);

   /* Loop over local coarse elements */
   big_k = nalu_hypre_ParCSRMatrixFirstRowIndex(ELEM_EDGE);
   for (i = 0; i < numELEM; i++, big_k++)
   {
      NALU_HYPRE_Int size1, size2;
      NALU_HYPRE_BigInt *col_ind0, *col_ind1, *col_ind2;

      NALU_HYPRE_BigInt *DOF0, *DOF;
      NALU_HYPRE_Int num_DOF;
      NALU_HYPRE_Int num_idof;
      NALU_HYPRE_BigInt *idof0, *idof, *bdof;
      NALU_HYPRE_Int num_bdof;

      NALU_HYPRE_Real *boolean_data;

      /* Determine the coarse DOFs */
      nalu_hypre_ParCSRMatrixGetRow (ELEM_DOF, big_k, &num_DOF, &DOF0, &boolean_data);
      DOF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  num_DOF, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < num_DOF; j++)
      {
         DOF[j] = DOF0[j];
      }
      nalu_hypre_ParCSRMatrixRestoreRow (ELEM_DOF, big_k, &num_DOF, &DOF0, &boolean_data);

      nalu_hypre_BigQsort0(DOF, 0, num_DOF - 1);

      /* Find the fine dofs interior for the current coarse element */
      nalu_hypre_ParCSRMatrixGetRow (ELEM_idof, big_k, &num_idof, &idof0, &boolean_data);
      idof = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  num_idof, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < num_idof; j++)
      {
         idof[j] = idof0[j];
      }
      nalu_hypre_ParCSRMatrixRestoreRow (ELEM_idof, big_k, &num_idof, &idof0, &boolean_data);

      /* Sort the interior dofs according to their global number */
      nalu_hypre_BigQsort0(idof, 0, num_idof - 1);

      /* Find the fine dofs on the boundary of the current coarse element */
      if (three_dimensional_problem)
      {
         nalu_hypre_ParCSRMatrixGetRow (ELEM_FACEidof, big_k, &size1, &col_ind0, &boolean_data);
         col_ind1 = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, size1, NALU_HYPRE_MEMORY_HOST);
         for (j = 0; j < size1; j++)
         {
            col_ind1[j] = col_ind0[j];
         }
         nalu_hypre_ParCSRMatrixRestoreRow (ELEM_FACEidof, big_k, &size1, &col_ind0, &boolean_data);
      }
      else
      {
         size1 = 0;
      }

      nalu_hypre_ParCSRMatrixGetRow (ELEM_EDGEidof, big_k, &size2, &col_ind0, &boolean_data);
      col_ind2 = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, size2, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < size2; j++)
      {
         col_ind2[j] = col_ind0[j];
      }
      nalu_hypre_ParCSRMatrixRestoreRow (ELEM_EDGEidof, big_k, &size2, &col_ind0, &boolean_data);

      /* Merge and sort the boundary dofs according to their global number */
      num_bdof = size1 + size2;
      bdof = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_bdof, NALU_HYPRE_MEMORY_HOST);
      if (three_dimensional_problem)
      {
         nalu_hypre_TMemcpy(bdof, col_ind1, NALU_HYPRE_BigInt, size1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TMemcpy(bdof + size1, col_ind2, NALU_HYPRE_BigInt, size2, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_BigQsort0(bdof, 0, num_bdof - 1);

      /* A = extract_rows(Aee, idof) */
      A = nalu_hypre_CSRMatrixCreate (num_idof, num_idof + num_bdof,
                                 num_idof * (num_idof + num_bdof));
      nalu_hypre_CSRMatrixBigInitialize(A);
      {
         NALU_HYPRE_Int *I = nalu_hypre_CSRMatrixI(A);
         NALU_HYPRE_BigInt *J = nalu_hypre_CSRMatrixBigJ(A);
         NALU_HYPRE_Real *data = nalu_hypre_CSRMatrixData(A);
         NALU_HYPRE_BigInt *tmp_J;
         NALU_HYPRE_Real *tmp_data;

         NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
         NALU_HYPRE_MemoryLocation memory_location_Aee = nalu_hypre_ParCSRMatrixMemoryLocation(Aee);

         I[0] = 0;
         for (j = 0; j < num_idof; j++)
         {
            getrow_ierr = nalu_hypre_ParCSRMatrixGetRow (Aee, idof[j], &size1, &tmp_J, &tmp_data);
            if (getrow_ierr < 0)
            {
               nalu_hypre_printf("getrow Aee off proc[%d] = \n", myproc);
            }
            nalu_hypre_TMemcpy(J, tmp_J, NALU_HYPRE_BigInt, size1, memory_location_A, memory_location_Aee);
            nalu_hypre_TMemcpy(data, tmp_data, NALU_HYPRE_Real, size1, memory_location_A, memory_location_Aee);
            J += size1;
            data += size1;
            nalu_hypre_ParCSRMatrixRestoreRow (Aee, idof[j], &size1, &tmp_J, &tmp_data);
            I[j + 1] = size1 + I[j];
         }
      }

      /* P = extract_rows(dof_DOF, idof+bdof) */
      P = nalu_hypre_CSRMatrixCreate (num_idof + num_bdof, num_DOF,
                                 (num_idof + num_bdof) * num_DOF);
      nalu_hypre_CSRMatrixBigInitialize(P);

      {
         NALU_HYPRE_Int *I = nalu_hypre_CSRMatrixI(P);
         NALU_HYPRE_BigInt *J = nalu_hypre_CSRMatrixBigJ(P);
         NALU_HYPRE_Real *data = nalu_hypre_CSRMatrixData(P);
         NALU_HYPRE_Int     m;

         NALU_HYPRE_BigInt *tmp_J;
         NALU_HYPRE_Real *tmp_data;

         NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_CSRMatrixMemoryLocation(P);
         NALU_HYPRE_MemoryLocation memory_location_d = nalu_hypre_ParCSRMatrixMemoryLocation(dof_DOF);

         I[0] = 0;
         for (j = 0; j < num_idof; j++)
         {
            getrow_ierr = nalu_hypre_ParCSRMatrixGetRow (dof_DOF, idof[j], &size1, &tmp_J, &tmp_data);
            if (getrow_ierr >= 0)
            {
               nalu_hypre_TMemcpy(J, tmp_J, NALU_HYPRE_BigInt, size1, memory_location_P, memory_location_d);
               nalu_hypre_TMemcpy(data, tmp_data, NALU_HYPRE_Real, size1, memory_location_P, memory_location_d);
               J += size1;
               data += size1;
               nalu_hypre_ParCSRMatrixRestoreRow (dof_DOF, idof[j], &size1, &tmp_J, &tmp_data);
               I[j + 1] = size1 + I[j];
            }
            else    /* row offproc */
            {
               nalu_hypre_ParCSRMatrixRestoreRow (dof_DOF, idof[j], &size1, &tmp_J, &tmp_data);
               /* search for OffProcRows */
               m = 0;
               while (m < num_OffProcRows)
               {
                  if (offproc_rnums[m] == idof[j])
                  {
                     break;
                  }
                  else
                  {
                     m++;
                  }
               }
               size1 = (OffProcRows[swap[m]] -> ncols);
               tmp_J = (OffProcRows[swap[m]] -> cols);
               tmp_data = (OffProcRows[swap[m]] -> data);
               nalu_hypre_TMemcpy(J, tmp_J, NALU_HYPRE_BigInt, size1, memory_location_P, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TMemcpy(data, tmp_data, NALU_HYPRE_Real, size1, memory_location_P, NALU_HYPRE_MEMORY_HOST);
               J += size1;
               data += size1;
               I[j + 1] = size1 + I[j];
            }
         }

         for ( ; j < num_idof + num_bdof; j++)
         {
            getrow_ierr = nalu_hypre_ParCSRMatrixGetRow (dof_DOF, bdof[j - num_idof], &size1, &tmp_J, &tmp_data);
            if (getrow_ierr >= 0)
            {
               nalu_hypre_TMemcpy(J, tmp_J, NALU_HYPRE_BigInt, size1, memory_location_P, memory_location_d);
               nalu_hypre_TMemcpy(data, tmp_data, NALU_HYPRE_Real, size1, memory_location_P, memory_location_d);
               J += size1;
               data += size1;
               nalu_hypre_ParCSRMatrixRestoreRow (dof_DOF, bdof[j - num_idof], &size1, &tmp_J, &tmp_data);
               I[j + 1] = size1 + I[j];
            }
            else    /* row offproc */
            {
               nalu_hypre_ParCSRMatrixRestoreRow (dof_DOF, bdof[j - num_idof], &size1, &tmp_J, &tmp_data);
               /* search for OffProcRows */
               m = 0;
               while (m < num_OffProcRows)
               {
                  if (offproc_rnums[m] == bdof[j - num_idof])
                  {
                     break;
                  }
                  else
                  {
                     m++;
                  }
               }
               if (m >= num_OffProcRows) { nalu_hypre_printf("here the mistake\n"); }
               size1 = (OffProcRows[swap[m]] -> ncols);
               tmp_J = (OffProcRows[swap[m]] -> cols);
               tmp_data = (OffProcRows[swap[m]] -> data);
               nalu_hypre_TMemcpy(J, tmp_J, NALU_HYPRE_BigInt, size1, memory_location_P, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TMemcpy(data, tmp_data, NALU_HYPRE_Real, size1, memory_location_P, NALU_HYPRE_MEMORY_HOST);
               J += size1;
               data += size1;
               I[j + 1] = size1 + I[j];
            }
         }
      }

      /* Pi = Aii^{-1} Aib Pb */
      nalu_hypre_HarmonicExtension (A, P, num_DOF, DOF,
                               num_idof, idof, num_bdof, bdof);

      /* Insert Pi in dof_DOF */
      {
         NALU_HYPRE_Int * ncols = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_idof, NALU_HYPRE_MEMORY_HOST);
         NALU_HYPRE_Int * idof_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_idof, NALU_HYPRE_MEMORY_HOST);

         for (j = 0; j < num_idof; j++)
         {
            ncols[j] = num_DOF;
            idof_indexes[j] = j * num_DOF;
         }

         nalu_hypre_IJMatrixAddToValuesParCSR (IJ_dof_DOF,
                                          num_idof, ncols, idof, idof_indexes,
                                          nalu_hypre_CSRMatrixBigJ(P),
                                          nalu_hypre_CSRMatrixData(P));

         nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(idof_indexes, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(DOF, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(idof, NALU_HYPRE_MEMORY_HOST);
      if (three_dimensional_problem)
      {
         nalu_hypre_TFree(col_ind1, NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(col_ind2, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(bdof, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_CSRMatrixDestroy(A);
      nalu_hypre_CSRMatrixDestroy(P);
   }

#if 0
   nalu_hypre_TFree(ij_dof_DOF, NALU_HYPRE_MEMORY_HOST);
#endif

   if (three_dimensional_problem)
   {
      nalu_hypre_ParCSRMatrixDestroy(ELEM_FACEidof);
   }
   nalu_hypre_ParCSRMatrixDestroy(ELEM_EDGEidof);

   if (num_OffProcRows)
   {
      nalu_hypre_TFree(offproc_rnums, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(swap, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}




NALU_HYPRE_Int nalu_hypre_HarmonicExtension (nalu_hypre_CSRMatrix *A,
                                   nalu_hypre_CSRMatrix *P,
                                   NALU_HYPRE_Int num_DOF, NALU_HYPRE_BigInt *DOF,
                                   NALU_HYPRE_Int num_idof, NALU_HYPRE_BigInt *idof,
                                   NALU_HYPRE_Int num_bdof, NALU_HYPRE_BigInt *bdof)
{
   NALU_HYPRE_Int ierr = 0;

   NALU_HYPRE_Int i, j, k, l, m;
   NALU_HYPRE_Real factor;

   NALU_HYPRE_Int *IA = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_BigInt *JA = nalu_hypre_CSRMatrixBigJ(A);
   NALU_HYPRE_Real *dataA = nalu_hypre_CSRMatrixData(A);

   NALU_HYPRE_Int *IP = nalu_hypre_CSRMatrixI(P);
   NALU_HYPRE_BigInt *JP = nalu_hypre_CSRMatrixBigJ(P);
   NALU_HYPRE_Real *dataP = nalu_hypre_CSRMatrixData(P);

   NALU_HYPRE_Real * Aii = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_idof * num_idof, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Real * Pi = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_idof * num_DOF, NALU_HYPRE_MEMORY_HOST);

   /* Loop over the rows of A */
   for (i = 0; i < num_idof; i++)
      for (j = IA[i]; j < IA[i + 1]; j++)
      {
         /* Global to local*/
         k = nalu_hypre_BigBinarySearch(idof, JA[j], num_idof);
         /* If a column is a bdof, compute its participation in Pi = Aib x Pb */
         if (k == -1)
         {
            k = nalu_hypre_BigBinarySearch(bdof, JA[j], num_bdof);
            if (k > -1)
            {
               for (l = IP[k + num_idof]; l < IP[k + num_idof + 1]; l++)
               {
                  m = nalu_hypre_BigBinarySearch(DOF, JP[l], num_DOF);
                  if (m > -1)
                  {
                     m += i * num_DOF;
                     /* Pi[i*num_DOF+m] += dataA[j] * dataP[l];*/
                     Pi[m] += dataA[j] * dataP[l];
                  }
               }
            }
         }
         /* If a column is an idof, put it in Aii */
         else
         {
            Aii[i * num_idof + k] = dataA[j];
         }
      }

   /* Perform Gaussian elimination in [Aii, Pi] */
   for (j = 0; j < num_idof - 1; j++)
      if (Aii[j * num_idof + j] != 0.0)
         for (i = j + 1; i < num_idof; i++)
            if (Aii[i * num_idof + j] != 0.0)
            {
               factor = Aii[i * num_idof + j] / Aii[j * num_idof + j];
               for (m = j + 1; m < num_idof; m++)
               {
                  Aii[i * num_idof + m] -= factor * Aii[j * num_idof + m];
               }
               for (m = 0; m < num_DOF; m++)
               {
                  Pi[i * num_DOF + m] -= factor * Pi[j * num_DOF + m];
               }
            }

   /* Back Substitution */
   for (i = num_idof - 1; i >= 0; i--)
   {
      for (j = i + 1; j < num_idof; j++)
         if (Aii[i * num_idof + j] != 0.0)
            for (m = 0; m < num_DOF; m++)
            {
               Pi[i * num_DOF + m] -= Aii[i * num_idof + j] * Pi[j * num_DOF + m];
            }

      for (m = 0; m < num_DOF; m++)
      {
         Pi[i * num_DOF + m] /= Aii[i * num_idof + i];
      }
   }

   /* Put -Pi back in P. We assume that each idof depends on _all_ DOFs */
   for (i = 0; i < num_idof; i++, JP += num_DOF, dataP += num_DOF)
      for (j = 0; j < num_DOF; j++)
      {
         JP[j]    = DOF[j];
         dataP[j] = -Pi[i * num_DOF + j];
      }

   nalu_hypre_TFree(Aii, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pi, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}
