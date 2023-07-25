/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_JacobiDevice( void     *amgdd_vdata,
                                    NALU_HYPRE_Int level )
{
   nalu_hypre_ParAMGDDData         *amgdd_data      = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid        *compGrid        = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_Real                  relax_weight    = nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);
   NALU_HYPRE_MemoryLocation        memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);

   nalu_hypre_AMGDDCompGridMatrix  *A = nalu_hypre_AMGDDCompGridA(compGrid);
   nalu_hypre_AMGDDCompGridVector  *f = nalu_hypre_AMGDDCompGridF(compGrid);
   nalu_hypre_AMGDDCompGridVector  *u = nalu_hypre_AMGDDCompGridU(compGrid);

   nalu_hypre_CSRMatrix            *diag;
   NALU_HYPRE_Int                   total_real_nodes;
   NALU_HYPRE_Int                   i, j;

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!nalu_hypre_AMGDDCompGridL1Norms(compGrid))
   {
      total_real_nodes = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                         nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
      nalu_hypre_AMGDDCompGridL1Norms(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                                           total_real_nodes,
                                                           memory_location);
      diag = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A);

      for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = nalu_hypre_CSRMatrixI(diag)[i]; j < nalu_hypre_CSRMatrixI(diag)[i + 1]; j++)
         {
            // nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i] += nalu_hypre_abs(nalu_hypre_CSRMatrixData(diag)[j]);
            if (nalu_hypre_CSRMatrixJ(diag)[j] == i)
            {
               nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i] = nalu_hypre_CSRMatrixData(diag)[j];
            }
         }
      }

      diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = nalu_hypre_CSRMatrixI(diag)[i]; j < nalu_hypre_CSRMatrixI(diag)[i + 1]; j++)
         {
            // nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += nalu_hypre_abs(nalu_hypre_CSRMatrixData(diag)[j]);
            if (nalu_hypre_CSRMatrixJ(diag)[j] == i)
            {
               nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(
                                                       compGrid)] = nalu_hypre_CSRMatrixData(diag)[j];
            }
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!nalu_hypre_AMGDDCompGridTemp2(compGrid))
   {
      nalu_hypre_AMGDDCompGridTemp2(compGrid) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridTemp2(compGrid),
                                          nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                                          nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid),
                                          nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   nalu_hypre_AMGDDCompGridVectorCopy(f, nalu_hypre_AMGDDCompGridTemp2(compGrid));

   nalu_hypre_AMGDDCompGridMatvec(-relax_weight, A, u, relax_weight, nalu_hypre_AMGDDCompGridTemp2(compGrid));

   hypreDevice_IVAXPY(nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                      nalu_hypre_AMGDDCompGridL1Norms(compGrid),
                      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid))),
                      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(u)));

   hypreDevice_IVAXPY(nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid),
                      &(nalu_hypre_AMGDDCompGridL1Norms(compGrid)[nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)]),
                      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid))),
                      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(u)));

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice( void      *amgdd_vdata,
                                        NALU_HYPRE_Int  level,
                                        NALU_HYPRE_Int  relax_set )
{
   nalu_hypre_ParAMGDDData    *amgdd_data      = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid   *compGrid        = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_Real             relax_weight    = nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);
   nalu_hypre_Vector          *owned_u         = nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridU(
                                                                              compGrid));
   nalu_hypre_Vector          *nonowned_u      = nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridU(
                                                                                 compGrid));
   NALU_HYPRE_Int              num_owned       = nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid);
   NALU_HYPRE_Int              num_nonowned    = nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
   NALU_HYPRE_Int              num_nonowned_r  = nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);

   nalu_hypre_Vector          *owned_tmp;
   nalu_hypre_Vector          *nonowned_tmp;

   // Allocate temporary vector if necessary
   if (!nalu_hypre_AMGDDCompGridTemp2(compGrid))
   {
      nalu_hypre_AMGDDCompGridTemp2(compGrid) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridTemp2(compGrid),
                                          num_owned,
                                          num_nonowned,
                                          num_nonowned_r);
   }

   nalu_hypre_AMGDDCompGridVectorCopy(nalu_hypre_AMGDDCompGridF(compGrid),
                                 nalu_hypre_AMGDDCompGridTemp2(compGrid));

   nalu_hypre_AMGDDCompGridMatvec(-relax_weight,
                             nalu_hypre_AMGDDCompGridA(compGrid),
                             nalu_hypre_AMGDDCompGridU(compGrid),
                             relax_weight,
                             nalu_hypre_AMGDDCompGridTemp2(compGrid));

   owned_tmp    = nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid));
   nonowned_tmp = nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid));

   hypreDevice_IVAXPYMarked(num_owned,
                            nalu_hypre_AMGDDCompGridL1Norms(compGrid),
                            nalu_hypre_VectorData(owned_tmp),
                            nalu_hypre_VectorData(owned_u),
                            nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid),
                            relax_set);

   hypreDevice_IVAXPYMarked(num_nonowned_r,
                            &(nalu_hypre_AMGDDCompGridL1Norms(compGrid)[num_owned]),
                            nalu_hypre_VectorData(nonowned_tmp),
                            nalu_hypre_VectorData(nonowned_u),
                            nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid) + num_owned,
                            relax_set);

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_GPU)
