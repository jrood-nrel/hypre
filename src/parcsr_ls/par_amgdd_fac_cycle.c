/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC( void *amgdd_vdata, NALU_HYPRE_Int first_iteration )
{
   nalu_hypre_ParAMGDDData  *amgdd_data  = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   NALU_HYPRE_Int            cycle_type  = nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data);
   NALU_HYPRE_Int            start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);

   if (cycle_type == 1 || cycle_type == 2)
   {
      nalu_hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, start_level, cycle_type, first_iteration);
   }
   else if (cycle_type == 3)
   {
      nalu_hypre_BoomerAMGDD_FAC_FCycle(amgdd_vdata, first_iteration);
   }
   else
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "WARNING: unknown AMG-DD FAC cycle type. Defaulting to 1 (V-cycle).\n");
      nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data) = 1;
      nalu_hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, start_level, 1, first_iteration);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_Cycle( void      *amgdd_vdata,
                             NALU_HYPRE_Int  level,
                             NALU_HYPRE_Int  cycle_type,
                             NALU_HYPRE_Int  first_iteration )
{
   nalu_hypre_ParAMGDDData    *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_ParAMGData      *amg_data   = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid  **compGrid   = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   NALU_HYPRE_Int              num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);

   NALU_HYPRE_Int i;

   // Relax on the real nodes
   nalu_hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, level, 1);

   // Restrict the residual at all fine points (real and ghost) and set residual at coarse points not under the fine grid
   if (num_levels > 1)
   {
      nalu_hypre_BoomerAMGDD_FAC_Restrict(compGrid[level], compGrid[level + 1], first_iteration);
      nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridS(compGrid[level]), 0.0);
      nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridT(compGrid[level]), 0.0);

      //  Either solve on the coarse level or recurse
      if (level + 1 == num_levels - 1)
      {
         nalu_hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, num_levels - 1, 3);
      }
      else for (i = 0; i < cycle_type; i++)
         {
            nalu_hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, level + 1, cycle_type, first_iteration);
            first_iteration = 0;
         }

      // Interpolate up and relax
      nalu_hypre_BoomerAMGDD_FAC_Interpolate(compGrid[level], compGrid[level + 1]);
   }

   nalu_hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, level, 2);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_FCycle( void     *amgdd_vdata,
                              NALU_HYPRE_Int first_iteration )
{
   nalu_hypre_ParAMGDDData    *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_ParAMGData      *amg_data   = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
   nalu_hypre_AMGDDCompGrid  **compGrid   = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   NALU_HYPRE_Int              num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);

   NALU_HYPRE_Int level;

   // ... work down to coarsest ...
   if (!first_iteration)
   {
      for (level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data); level < num_levels - 1; level++)
      {
         nalu_hypre_BoomerAMGDD_FAC_Restrict(compGrid[level], compGrid[level + 1], 0);
         nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridS(compGrid[level]), 0.0);
         nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridT(compGrid[level]), 0.0);
      }
   }

   //  ... solve on coarsest level ...
   nalu_hypre_BoomerAMGDD_FAC_Relax(amgdd_vdata, num_levels - 1, 3);

   // ... and work back up to the finest
   for (level = num_levels - 2; level > -1; level--)
   {
      // Interpolate up and relax
      nalu_hypre_BoomerAMGDD_FAC_Interpolate(compGrid[level], compGrid[level + 1]);

      // V-cycle
      nalu_hypre_BoomerAMGDD_FAC_Cycle(amgdd_vdata, level, 1, 0);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_Interpolate( nalu_hypre_AMGDDCompGrid *compGrid_f,
                                   nalu_hypre_AMGDDCompGrid *compGrid_c )
{
   nalu_hypre_AMGDDCompGridMatvec(1.0, nalu_hypre_AMGDDCompGridP(compGrid_f),
                             nalu_hypre_AMGDDCompGridU(compGrid_c),
                             1.0, nalu_hypre_AMGDDCompGridU(compGrid_f));
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_Restrict( nalu_hypre_AMGDDCompGrid *compGrid_f,
                                nalu_hypre_AMGDDCompGrid *compGrid_c,
                                NALU_HYPRE_Int            first_iteration )
{
   // Recalculate residual on coarse grid
   if (!first_iteration)
   {
      nalu_hypre_AMGDDCompGridMatvec(-1.0, nalu_hypre_AMGDDCompGridA(compGrid_c),
                                nalu_hypre_AMGDDCompGridU(compGrid_c),
                                1.0, nalu_hypre_AMGDDCompGridF(compGrid_c));
   }

   // Get update: s_l <- A_lt_l + s_l
   nalu_hypre_AMGDDCompGridMatvec(1.0, nalu_hypre_AMGDDCompGridA(compGrid_f),
                             nalu_hypre_AMGDDCompGridT(compGrid_f),
                             1.0, nalu_hypre_AMGDDCompGridS(compGrid_f));

   // If we need to preserve the updates on the next level
   if (nalu_hypre_AMGDDCompGridS(compGrid_c))
   {
      nalu_hypre_AMGDDCompGridMatvec(1.0, nalu_hypre_AMGDDCompGridR(compGrid_f),
                                nalu_hypre_AMGDDCompGridS(compGrid_f),
                                0.0, nalu_hypre_AMGDDCompGridS(compGrid_c));

      // Subtract restricted update from recalculated residual: f_{l+1} <- f_{l+1} - s_{l+1}
      nalu_hypre_AMGDDCompGridVectorAxpy(-1.0, nalu_hypre_AMGDDCompGridS(compGrid_c),
                                    nalu_hypre_AMGDDCompGridF(compGrid_c));
   }
   else
   {
      // Restrict and subtract update from recalculated residual: f_{l+1} <- f_{l+1} - P_l^Ts_l
      nalu_hypre_AMGDDCompGridMatvec(-1.0, nalu_hypre_AMGDDCompGridR(compGrid_f),
                                nalu_hypre_AMGDDCompGridS(compGrid_f),
                                1.0, nalu_hypre_AMGDDCompGridF(compGrid_c));
   }

   // Zero out initial guess on coarse grid
   nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridU(compGrid_c), 0.0);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_Relax( void      *amgdd_vdata,
                             NALU_HYPRE_Int  level,
                             NALU_HYPRE_Int  cycle_param )
{
   nalu_hypre_ParAMGDDData   *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid  *compGrid   = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_Int             numRelax   = nalu_hypre_ParAMGDDDataFACNumRelax(amgdd_data);
   NALU_HYPRE_Int             i;

   if (nalu_hypre_AMGDDCompGridT(compGrid) || nalu_hypre_AMGDDCompGridQ(compGrid))
   {
      nalu_hypre_AMGDDCompGridVectorCopy(nalu_hypre_AMGDDCompGridU(compGrid),
                                    nalu_hypre_AMGDDCompGridTemp(compGrid));
      nalu_hypre_AMGDDCompGridVectorScale(-1.0, nalu_hypre_AMGDDCompGridTemp(compGrid));
   }

   for (i = 0; i < numRelax; i++)
   {
      (*nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data))(amgdd_vdata, level, cycle_param);
   }

   if (nalu_hypre_AMGDDCompGridT(compGrid) || nalu_hypre_AMGDDCompGridQ(compGrid))
   {
      nalu_hypre_AMGDDCompGridVectorAxpy(1.0,
                                    nalu_hypre_AMGDDCompGridU(compGrid),
                                    nalu_hypre_AMGDDCompGridTemp(compGrid));

      if (nalu_hypre_AMGDDCompGridT(compGrid))
      {
         nalu_hypre_AMGDDCompGridVectorAxpy(1.0,
                                       nalu_hypre_AMGDDCompGridTemp(compGrid),
                                       nalu_hypre_AMGDDCompGridT(compGrid));
      }
      if (nalu_hypre_AMGDDCompGridQ(compGrid))
      {
         nalu_hypre_AMGDDCompGridVectorAxpy(1.0,
                                       nalu_hypre_AMGDDCompGridTemp(compGrid),
                                       nalu_hypre_AMGDDCompGridQ(compGrid));
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_Jacobi( void      *amgdd_vdata,
                              NALU_HYPRE_Int  level,
                              NALU_HYPRE_Int  cycle_param )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ParAMGDDData      *amgdd_data      = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid     *compGrid        = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);
   NALU_HYPRE_ExecutionPolicy    exec            = nalu_hypre_GetExecPolicy1(memory_location);

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_BoomerAMGDD_FAC_JacobiDevice(amgdd_vdata, level);
   }
   else
#endif
   {
      nalu_hypre_BoomerAMGDD_FAC_JacobiHost(amgdd_vdata, level);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_JacobiHost( void      *amgdd_vdata,
                                  NALU_HYPRE_Int  level )
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

   for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(u))[i] +=
         nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid)))[i] /
         nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i];
   }
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
      nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(u))[i] +=
         nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid)))[i] /
         nalu_hypre_AMGDDCompGridL1Norms(compGrid)[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)];
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_GaussSeidel( void      *amgdd_vdata,
                                   NALU_HYPRE_Int  level,
                                   NALU_HYPRE_Int  cycle_param )
{
   nalu_hypre_ParAMGDDData    *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid   *compGrid   = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];

   nalu_hypre_AMGDDCompGridMatrix      *A = nalu_hypre_AMGDDCompGridA(compGrid);
   nalu_hypre_AMGDDCompGridVector      *f = nalu_hypre_AMGDDCompGridF(compGrid);
   nalu_hypre_AMGDDCompGridVector      *u = nalu_hypre_AMGDDCompGridU(compGrid);

   nalu_hypre_CSRMatrix  *owned_diag      = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A);
   nalu_hypre_CSRMatrix  *owned_offd      = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A);
   nalu_hypre_CSRMatrix  *nonowned_diag   = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   nalu_hypre_CSRMatrix  *nonowned_offd   = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A);
   NALU_HYPRE_Complex    *u_owned_data    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(u));
   NALU_HYPRE_Complex    *u_nonowned_data = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(u));
   NALU_HYPRE_Complex    *f_owned_data    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(f));
   NALU_HYPRE_Complex    *f_nonowned_data = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(f));

   NALU_HYPRE_Int        i, j; // loop variables
   NALU_HYPRE_Complex    diagonal; // placeholder for the diagonal of A

   // Do Gauss-Seidel relaxation on the owned nodes
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
      // Initialize u as RHS
      u_owned_data[i] = f_owned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = nalu_hypre_CSRMatrixI(owned_diag)[i]; j < nalu_hypre_CSRMatrixI(owned_diag)[i + 1]; j++)
      {
         if (nalu_hypre_CSRMatrixJ(owned_diag)[j] == i)
         {
            diagonal = nalu_hypre_CSRMatrixData(owned_diag)[j];
         }
         else
         {
            u_owned_data[i] -= nalu_hypre_CSRMatrixData(owned_diag)[j] * u_owned_data[ nalu_hypre_CSRMatrixJ(
                                                                                     owned_diag)[j] ];
         }
      }

      // Loop over offd entries
      for (j = nalu_hypre_CSRMatrixI(owned_offd)[i]; j < nalu_hypre_CSRMatrixI(owned_offd)[i + 1]; j++)
      {
         u_owned_data[i] -= nalu_hypre_CSRMatrixData(owned_offd)[j] * u_nonowned_data[ nalu_hypre_CSRMatrixJ(
                                                                                     owned_offd)[j] ];
      }

      // Divide by diagonal
      if (diagonal == 0.0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "WARNING: Divide by zero diagonal in nalu_hypre_BoomerAMGDD_FAC_GaussSeidel().\n");
      }
      u_owned_data[i] /= diagonal;
   }

   // Do Gauss-Seidel relaxation on the nonowned nodes
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
      // Initialize u as RHS
      u_nonowned_data[i] = f_nonowned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = nalu_hypre_CSRMatrixI(nonowned_diag)[i]; j < nalu_hypre_CSRMatrixI(nonowned_diag)[i + 1]; j++)
      {
         if (nalu_hypre_CSRMatrixJ(nonowned_diag)[j] == i)
         {
            diagonal = nalu_hypre_CSRMatrixData(nonowned_diag)[j];
         }
         else
         {
            u_nonowned_data[i] -= nalu_hypre_CSRMatrixData(nonowned_diag)[j] * u_nonowned_data[ nalu_hypre_CSRMatrixJ(
                                                                                              nonowned_diag)[j] ];
         }
      }

      // Loop over offd entries
      for (j = nalu_hypre_CSRMatrixI(nonowned_offd)[i]; j < nalu_hypre_CSRMatrixI(nonowned_offd)[i + 1]; j++)
      {
         u_nonowned_data[i] -= nalu_hypre_CSRMatrixData(nonowned_offd)[j] * u_owned_data[ nalu_hypre_CSRMatrixJ(
                                                                                        nonowned_offd)[j] ];
      }

      // Divide by diagonal
      if (diagonal == 0.0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "WARNING: Divide by zero diagonal in nalu_hypre_BoomerAMGDD_FAC_GaussSeidel().\n");
      }
      u_nonowned_data[i] /= diagonal;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_OrderedGaussSeidel( void       *amgdd_vdata,
                                          NALU_HYPRE_Int   level,
                                          NALU_HYPRE_Int   cycle_param )
{
   nalu_hypre_ParAMGDDData         *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid        *compGrid   = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];

   nalu_hypre_AMGDDCompGridMatrix  *A = nalu_hypre_AMGDDCompGridA(compGrid);
   nalu_hypre_AMGDDCompGridVector  *f = nalu_hypre_AMGDDCompGridF(compGrid);
   nalu_hypre_AMGDDCompGridVector  *u = nalu_hypre_AMGDDCompGridU(compGrid);

   NALU_HYPRE_Int                   unordered_i, i, j; // loop variables
   NALU_HYPRE_Complex               diagonal; // placeholder for the diagonal of A

   if (!nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid))
   {
      nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                      nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                                                                      nalu_hypre_AMGDDCompGridMemoryLocation(compGrid));
      nalu_hypre_topo_sort(nalu_hypre_CSRMatrixI(nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(
                                                                             compGrid))),
                      nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid))),
                      nalu_hypre_CSRMatrixData(nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid))),
                      nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid),
                      nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid));
   }

   if (!nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid))
   {
      nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                                         nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid),
                                                                         nalu_hypre_AMGDDCompGridMemoryLocation(compGrid));
      nalu_hypre_topo_sort(nalu_hypre_CSRMatrixI(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(
                                                                                compGrid))),
                      nalu_hypre_CSRMatrixJ(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid))),
                      nalu_hypre_CSRMatrixData(nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrid))),
                      nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid),
                      nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid));
   }

   // Get all the info
   NALU_HYPRE_Complex   *u_owned_data    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(u));
   NALU_HYPRE_Complex   *u_nonowned_data = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(u));
   NALU_HYPRE_Complex   *f_owned_data    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(f));
   NALU_HYPRE_Complex   *f_nonowned_data = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(f));
   nalu_hypre_CSRMatrix *owned_diag      = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(A);
   nalu_hypre_CSRMatrix *owned_offd      = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(A);
   nalu_hypre_CSRMatrix *nonowned_diag   = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   nalu_hypre_CSRMatrix *nonowned_offd   = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // Do Gauss-Seidel relaxation on the nonowned real nodes
   for (unordered_i = 0; unordered_i < nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
        unordered_i++)
   {
      i = nalu_hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid)[unordered_i];

      // Initialize u as RHS
      u_nonowned_data[i] = f_nonowned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = nalu_hypre_CSRMatrixI(nonowned_diag)[i]; j < nalu_hypre_CSRMatrixI(nonowned_diag)[i + 1]; j++)
      {
         if (nalu_hypre_CSRMatrixJ(nonowned_diag)[j] == i)
         {
            diagonal = nalu_hypre_CSRMatrixData(nonowned_diag)[j];
         }
         else
         {
            u_nonowned_data[i] -= nalu_hypre_CSRMatrixData(nonowned_diag)[j] * u_nonowned_data[ nalu_hypre_CSRMatrixJ(
                                                                                              nonowned_diag)[j] ];
         }
      }

      // Loop over offd entries
      for (j = nalu_hypre_CSRMatrixI(nonowned_offd)[i]; j < nalu_hypre_CSRMatrixI(nonowned_offd)[i + 1]; j++)
      {
         u_nonowned_data[i] -= nalu_hypre_CSRMatrixData(nonowned_offd)[j] * u_owned_data[ nalu_hypre_CSRMatrixJ(
                                                                                        nonowned_offd)[j] ];
      }

      // Divide by diagonal
      if (diagonal == 0.0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "WARNING: Divide by zero diagonal in nalu_hypre_BoomerAMGDD_FAC_OrderedGaussSeidel().\n");
      }
      u_nonowned_data[i] /= diagonal;
   }

   // Do Gauss-Seidel relaxation on the owned nodes
   for (unordered_i = 0; unordered_i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); unordered_i++)
   {
      i = nalu_hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid)[unordered_i];

      // Initialize u as RHS
      u_owned_data[i] = f_owned_data[i];
      diagonal = 0.0;

      // Loop over diag entries
      for (j = nalu_hypre_CSRMatrixI(owned_diag)[i]; j < nalu_hypre_CSRMatrixI(owned_diag)[i + 1]; j++)
      {
         if (nalu_hypre_CSRMatrixJ(owned_diag)[j] == i)
         {
            diagonal = nalu_hypre_CSRMatrixData(owned_diag)[j];
         }
         else
         {
            u_owned_data[i] -= nalu_hypre_CSRMatrixData(owned_diag)[j] * u_owned_data[ nalu_hypre_CSRMatrixJ(
                                                                                     owned_diag)[j] ];
         }
      }

      // Loop over offd entries
      for (j = nalu_hypre_CSRMatrixI(owned_offd)[i]; j < nalu_hypre_CSRMatrixI(owned_offd)[i + 1]; j++)
      {
         u_owned_data[i] -= nalu_hypre_CSRMatrixData(owned_offd)[j] * u_nonowned_data[ nalu_hypre_CSRMatrixJ(
                                                                                     owned_offd)[j] ];
      }

      // Divide by diagonal
      if (diagonal == 0.0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "WARNING: Divide by zero diagonal in nalu_hypre_BoomerAMGDD_FAC_OrderedGaussSeidel().\n");
      }
      u_owned_data[i] /= diagonal;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_CFL1Jacobi( void      *amgdd_vdata,
                                  NALU_HYPRE_Int  level,
                                  NALU_HYPRE_Int  cycle_param )
{
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ParAMGDDData      *amgdd_data      = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid     *compGrid        = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_MemoryLocation     memory_location = nalu_hypre_AMGDDCompGridMemoryLocation(compGrid);
   NALU_HYPRE_ExecutionPolicy    exec            = nalu_hypre_GetExecPolicy1(memory_location);

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      if (cycle_param == 1)
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice(amgdd_vdata, level, 1);
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice(amgdd_vdata, level, -1);
      }
      else if (cycle_param == 2)
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice(amgdd_vdata, level, -1);
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice(amgdd_vdata, level, 1);
      }
      else
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiDevice(amgdd_vdata, level, -1);
      }
   }
   else
#endif
   {
      if (cycle_param == 1)
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost(amgdd_vdata, level, 1);
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost(amgdd_vdata, level, -1);
      }
      else if (cycle_param == 2)
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost(amgdd_vdata, level, -1);
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost(amgdd_vdata, level, 1);
      }
      else
      {
         nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost(amgdd_vdata, level, -1);
      }
   }

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_FAC_CFL1JacobiHost( void      *amgdd_vdata,
                                      NALU_HYPRE_Int  level,
                                      NALU_HYPRE_Int  relax_set )
{
   nalu_hypre_ParAMGDDData   *amgdd_data   = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_AMGDDCompGrid  *compGrid     = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   NALU_HYPRE_Real            relax_weight = nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   nalu_hypre_CSRMatrix      *owned_diag    = nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(
                                                                               compGrid));
   nalu_hypre_CSRMatrix      *owned_offd    = nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridA(
                                                                               compGrid));
   nalu_hypre_CSRMatrix      *nonowned_diag = nalu_hypre_AMGDDCompGridMatrixNonOwnedDiag(nalu_hypre_AMGDDCompGridA(
                                                                                  compGrid));
   nalu_hypre_CSRMatrix      *nonowned_offd = nalu_hypre_AMGDDCompGridMatrixNonOwnedOffd(nalu_hypre_AMGDDCompGridA(
                                                                                  compGrid));

   NALU_HYPRE_Complex        *owned_u       = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(
                                                             nalu_hypre_AMGDDCompGridU(compGrid)));
   NALU_HYPRE_Complex        *nonowned_u    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(
                                                             nalu_hypre_AMGDDCompGridU(compGrid)));
   NALU_HYPRE_Complex        *owned_f       = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(
                                                             nalu_hypre_AMGDDCompGridF(compGrid)));
   NALU_HYPRE_Complex        *nonowned_f    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(
                                                             nalu_hypre_AMGDDCompGridF(compGrid)));

   NALU_HYPRE_Real           *l1_norms      = nalu_hypre_AMGDDCompGridL1Norms(compGrid);
   NALU_HYPRE_Int            *cf_marker     = nalu_hypre_AMGDDCompGridCFMarkerArray(compGrid);

   NALU_HYPRE_Complex        *owned_tmp;
   NALU_HYPRE_Complex        *nonowned_tmp;

   NALU_HYPRE_Int             i, j;
   NALU_HYPRE_Real            res;

   /*-----------------------------------------------------------------
    * Create and initialize Temp2 vector if not done before.
    *-----------------------------------------------------------------*/

   if (!nalu_hypre_AMGDDCompGridTemp2(compGrid))
   {
      nalu_hypre_AMGDDCompGridTemp2(compGrid) = nalu_hypre_AMGDDCompGridVectorCreate();
      nalu_hypre_AMGDDCompGridVectorInitialize(nalu_hypre_AMGDDCompGridTemp2(compGrid),
                                          nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                                          nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid),
                                          nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }
   owned_tmp    = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridTemp2(compGrid)));
   nonowned_tmp = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridTemp2(
                                                                        compGrid)));

   /*-----------------------------------------------------------------
    * Copy current approximation into temporary vector.
    *-----------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
      owned_tmp[i] = owned_u[i];
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedNodes(compGrid); i++)
   {
      nonowned_tmp[i] = nonowned_u[i];
   }

   /*-----------------------------------------------------------------
   * Relax only C or F points as determined by relax_points.
   *-----------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,res) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
   {
      if (cf_marker[i] == relax_set)
      {
         res = owned_f[i];
         for (j = nalu_hypre_CSRMatrixI(owned_diag)[i]; j < nalu_hypre_CSRMatrixI(owned_diag)[i + 1]; j++)
         {
            res -= nalu_hypre_CSRMatrixData(owned_diag)[j] * owned_tmp[ nalu_hypre_CSRMatrixJ(owned_diag)[j] ];
         }
         for (j = nalu_hypre_CSRMatrixI(owned_offd)[i]; j < nalu_hypre_CSRMatrixI(owned_offd)[i + 1]; j++)
         {
            res -= nalu_hypre_CSRMatrixData(owned_offd)[j] * nonowned_tmp[ nalu_hypre_CSRMatrixJ(owned_offd)[j] ];
         }
         owned_u[i] += (relax_weight * res) / l1_norms[i];
      }
   }
   for (i = 0; i < nalu_hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
   {
      if (cf_marker[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)] == relax_set)
      {
         res = nonowned_f[i];
         for (j = nalu_hypre_CSRMatrixI(nonowned_diag)[i]; j < nalu_hypre_CSRMatrixI(nonowned_diag)[i + 1]; j++)
         {
            res -= nalu_hypre_CSRMatrixData(nonowned_diag)[j] * nonowned_tmp[ nalu_hypre_CSRMatrixJ(nonowned_diag)[j] ];
         }
         for (j = nalu_hypre_CSRMatrixI(nonowned_offd)[i]; j < nalu_hypre_CSRMatrixI(nonowned_offd)[i + 1]; j++)
         {
            res -= nalu_hypre_CSRMatrixData(nonowned_offd)[j] * owned_tmp[ nalu_hypre_CSRMatrixJ(nonowned_offd)[j] ];
         }
         nonowned_u[i] += (relax_weight * res) / l1_norms[i + nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid)];
      }
   }

   return nalu_hypre_error_flag;
}
