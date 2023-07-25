/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ParAMGDD_DATA_HEADER
#define nalu_hypre_ParAMGDD_DATA_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGDDData
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* The underlying AMG hierarchy */
   nalu_hypre_ParAMGData          *amg_data;

   /* AMG-DD parameters and info */
   NALU_HYPRE_Int                 start_level;
   NALU_HYPRE_Int                 fac_num_cycles;
   NALU_HYPRE_Int                 fac_cycle_type;
   NALU_HYPRE_Int                 fac_relax_type;
   NALU_HYPRE_Int                 fac_num_relax;
   NALU_HYPRE_Real                fac_relax_weight;
   NALU_HYPRE_Int                 padding;
   NALU_HYPRE_Int                 num_ghost_layers;
   nalu_hypre_AMGDDCompGrid     **amgdd_comp_grid;
   nalu_hypre_AMGDDCommPkg       *amgdd_comm_pkg;
   nalu_hypre_ParVector          *Ztemp;

   NALU_HYPRE_Int       (*amgddUserFACRelaxation)( void *amgdd_vdata, NALU_HYPRE_Int level,
                                              NALU_HYPRE_Int cycle_param );
} nalu_hypre_ParAMGDDData;

/*--------------------------------------------------------------------------
 * Accessor functions for the nalu_hypre_AMGDDData structure
 *--------------------------------------------------------------------------*/
#define nalu_hypre_ParAMGDDDataAMG(amgdd_data)               ((amgdd_data)->amg_data)
#define nalu_hypre_ParAMGDDDataStartLevel(amgdd_data)        ((amgdd_data)->start_level)
#define nalu_hypre_ParAMGDDDataFACNumCycles(amgdd_data)      ((amgdd_data)->fac_num_cycles)
#define nalu_hypre_ParAMGDDDataFACCycleType(amgdd_data)      ((amgdd_data)->fac_cycle_type)
#define nalu_hypre_ParAMGDDDataFACRelaxType(amgdd_data)      ((amgdd_data)->fac_relax_type)
#define nalu_hypre_ParAMGDDDataFACNumRelax(amgdd_data)       ((amgdd_data)->fac_num_relax)
#define nalu_hypre_ParAMGDDDataFACRelaxWeight(amgdd_data)    ((amgdd_data)->fac_relax_weight)
#define nalu_hypre_ParAMGDDDataPadding(amgdd_data)           ((amgdd_data)->padding)
#define nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data)    ((amgdd_data)->num_ghost_layers)
#define nalu_hypre_ParAMGDDDataCompGrid(amgdd_data)          ((amgdd_data)->amgdd_comp_grid)
#define nalu_hypre_ParAMGDDDataCommPkg(amgdd_data)           ((amgdd_data)->amgdd_comm_pkg)
#define nalu_hypre_ParAMGDDDataZtemp(amg_data)               ((amgdd_data)->Ztemp)
#define nalu_hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) ((amgdd_data)->amgddUserFACRelaxation)

#endif
