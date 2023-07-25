/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMGDD functions
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGDDCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_BoomerAMGDDCreate( void )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = nalu_hypre_CTAlloc(nalu_hypre_ParAMGDDData, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParAMGDDDataAMG(amgdd_data) = (nalu_hypre_ParAMGData*) nalu_hypre_BoomerAMGCreate();

   nalu_hypre_ParAMGDDDataFACNumCycles(amgdd_data)   = 2;
   nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data)   = 1;
   nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data)   = 3;
   nalu_hypre_ParAMGDDDataFACNumRelax(amgdd_data)    = 1;
   nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data) = 1.0;
   nalu_hypre_ParAMGDDDataPadding(amgdd_data)        = 1;
   nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data) = 1;
   nalu_hypre_ParAMGDDDataCommPkg(amgdd_data)        = NULL;
   nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)       = NULL;
   nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = nalu_hypre_BoomerAMGDD_FAC_CFL1Jacobi;

   return (void *) amgdd_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGDDDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDDestroy( void *data )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;
   nalu_hypre_ParAMGData    *amg_data;
   NALU_HYPRE_Int            num_levels;
   NALU_HYPRE_Int            i;

   if (amgdd_data)
   {
      amg_data   = nalu_hypre_ParAMGDDDataAMG(amgdd_data);
      num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);

      /* destroy amgdd composite grids and commpkg */
      if (nalu_hypre_ParAMGDDDataCompGrid(amgdd_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            nalu_hypre_AMGDDCompGridDestroy(nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)[i]);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDDDataCompGrid(amgdd_data), NALU_HYPRE_MEMORY_HOST);
      }

      if (nalu_hypre_ParAMGDDDataCommPkg(amgdd_data))
      {
         nalu_hypre_AMGDDCommPkgDestroy(nalu_hypre_ParAMGDDDataCommPkg(amgdd_data));
      }

      /* destroy temporary vector */
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDDDataZtemp(amgdd_data));

      /* destroy the underlying amg */
      nalu_hypre_BoomerAMGDestroy((void*) amg_data);

      nalu_hypre_TFree(amgdd_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set parameters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetStartLevel( void     *data,
                                NALU_HYPRE_Int start_level )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataStartLevel(amgdd_data) = start_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetStartLevel( void      *data,
                                NALU_HYPRE_Int *start_level )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetFACNumRelax( void     *data,
                                 NALU_HYPRE_Int fac_num_relax )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataFACNumRelax(amgdd_data) = fac_num_relax;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetFACNumRelax( void      *data,
                                 NALU_HYPRE_Int *fac_num_relax )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fac_num_relax = nalu_hypre_ParAMGDDDataFACNumRelax(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetFACNumCycles( void     *data,
                                  NALU_HYPRE_Int fac_num_cycles )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataFACNumCycles(amgdd_data) = fac_num_cycles;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetFACNumCycles( void      *data,
                                  NALU_HYPRE_Int *fac_num_cycles )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fac_num_cycles = nalu_hypre_ParAMGDDDataFACNumCycles(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetFACCycleType( void     *data,
                                  NALU_HYPRE_Int fac_cycle_type )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data) = fac_cycle_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetFACCycleType( void      *data,
                                  NALU_HYPRE_Int *fac_cycle_type )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fac_cycle_type = nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetFACRelaxType( void     *data,
                                  NALU_HYPRE_Int fac_relax_type )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data) = fac_relax_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetFACRelaxType( void      *data,
                                  NALU_HYPRE_Int *fac_relax_type )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fac_relax_type = nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetFACRelaxWeight( void       *data,
                                    NALU_HYPRE_Real  fac_relax_weight )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data) = fac_relax_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetFACRelaxWeight( void       *data,
                                    NALU_HYPRE_Real *fac_relax_weight )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fac_relax_weight = nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetPadding( void      *data,
                             NALU_HYPRE_Int  padding )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataPadding(amgdd_data) = padding;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetPadding( void      *data,
                             NALU_HYPRE_Int *padding )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *padding = nalu_hypre_ParAMGDDDataPadding(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetNumGhostLayers( void      *data,
                                    NALU_HYPRE_Int  num_ghost_layers )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data) = num_ghost_layers;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetNumGhostLayers( void      *data,
                                    NALU_HYPRE_Int *num_ghost_layers )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *num_ghost_layers = nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetUserFACRelaxation( void *data,
                                       NALU_HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level, NALU_HYPRE_Int cycle_param ))
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = userFACRelaxation;

   return 0;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDGetAMG( void   *data,
                         void  **amg_solver )
{
   nalu_hypre_ParAMGDDData  *amgdd_data = (nalu_hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *amg_solver = (void*) nalu_hypre_ParAMGDDDataAMG(amgdd_data);

   return nalu_hypre_error_flag;
}
