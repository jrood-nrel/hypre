/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParAMGDD_DATA_HEADER
#define hypre_ParAMGDD_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParAMGDDData
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* The underlying AMG hierarchy */
   hypre_ParAMGData          *amg_data;

   /* AMG-DD parameters and info */
   NALU_HYPRE_Int                 start_level;
   NALU_HYPRE_Int                 fac_num_cycles;
   NALU_HYPRE_Int                 fac_cycle_type;
   NALU_HYPRE_Int                 fac_relax_type;
   NALU_HYPRE_Int                 fac_num_relax;
   NALU_HYPRE_Real                fac_relax_weight;
   NALU_HYPRE_Int                 padding;
   NALU_HYPRE_Int                 num_ghost_layers;
   hypre_AMGDDCompGrid     **amgdd_comp_grid;
   hypre_AMGDDCommPkg       *amgdd_comm_pkg;
   hypre_ParVector          *Ztemp;

   NALU_HYPRE_Int       (*amgddUserFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level,
                                              NALU_HYPRE_Int cycle_param );
} hypre_ParAMGDDData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_AMGDDData structure
 *--------------------------------------------------------------------------*/
#define hypre_ParAMGDDDataAMG(amgdd_data)               ((amgdd_data)->amg_data)
#define hypre_ParAMGDDDataStartLevel(amgdd_data)        ((amgdd_data)->start_level)
#define hypre_ParAMGDDDataFACNumCycles(amgdd_data)      ((amgdd_data)->fac_num_cycles)
#define hypre_ParAMGDDDataFACCycleType(amgdd_data)      ((amgdd_data)->fac_cycle_type)
#define hypre_ParAMGDDDataFACRelaxType(amgdd_data)      ((amgdd_data)->fac_relax_type)
#define hypre_ParAMGDDDataFACNumRelax(amgdd_data)       ((amgdd_data)->fac_num_relax)
#define hypre_ParAMGDDDataFACRelaxWeight(amgdd_data)    ((amgdd_data)->fac_relax_weight)
#define hypre_ParAMGDDDataPadding(amgdd_data)           ((amgdd_data)->padding)
#define hypre_ParAMGDDDataNumGhostLayers(amgdd_data)    ((amgdd_data)->num_ghost_layers)
#define hypre_ParAMGDDDataCompGrid(amgdd_data)          ((amgdd_data)->amgdd_comp_grid)
#define hypre_ParAMGDDDataCommPkg(amgdd_data)           ((amgdd_data)->amgdd_comm_pkg)
#define hypre_ParAMGDDDataZtemp(amg_data)               ((amgdd_data)->Ztemp)
#define hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) ((amgdd_data)->amgddUserFACRelaxation)

#endif
