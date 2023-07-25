/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_Schwarz_DATA_HEADER
#define nalu_hypre_Schwarz_DATA_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SchwarzData
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int      variant;
   NALU_HYPRE_Int      domain_type;
   NALU_HYPRE_Int      overlap;
   NALU_HYPRE_Int      num_functions;
   NALU_HYPRE_Int      use_nonsymm;
   NALU_HYPRE_Real   relax_weight;

   nalu_hypre_CSRMatrix *domain_structure;
   nalu_hypre_CSRMatrix *A_boundary;
   nalu_hypre_ParVector *Vtemp;
   NALU_HYPRE_Real  *scale;
   NALU_HYPRE_Int     *dof_func;
   NALU_HYPRE_Int     *pivots;



} nalu_hypre_SchwarzData;

/*--------------------------------------------------------------------------
 * Accessor functions for the nalu_hypre_SchwarzData structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SchwarzDataVariant(schwarz_data) ((schwarz_data)->variant)
#define nalu_hypre_SchwarzDataDomainType(schwarz_data) ((schwarz_data)->domain_type)
#define nalu_hypre_SchwarzDataOverlap(schwarz_data) ((schwarz_data)->overlap)
#define nalu_hypre_SchwarzDataNumFunctions(schwarz_data) \
((schwarz_data)->num_functions)
#define nalu_hypre_SchwarzDataUseNonSymm(schwarz_data) \
((schwarz_data)->use_nonsymm)
#define nalu_hypre_SchwarzDataRelaxWeight(schwarz_data) \
((schwarz_data)->relax_weight)
#define nalu_hypre_SchwarzDataDomainStructure(schwarz_data) \
((schwarz_data)->domain_structure)
#define nalu_hypre_SchwarzDataABoundary(schwarz_data) ((schwarz_data)->A_boundary)
#define nalu_hypre_SchwarzDataVtemp(schwarz_data) ((schwarz_data)->Vtemp)
#define nalu_hypre_SchwarzDataScale(schwarz_data) ((schwarz_data)->scale)
#define nalu_hypre_SchwarzDataDofFunc(schwarz_data) ((schwarz_data)->dof_func)
#define nalu_hypre_SchwarzDataPivots(schwarz_data) ((schwarz_data)->pivots)

#endif



