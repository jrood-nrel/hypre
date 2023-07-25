/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructOwnInfo data structure
 * This structure is for the coarsen fboxes that are on this processor,
 * and the cboxes of cgrid/(all coarsened fboxes) on this processor (i.e.,
 * the coarse boxes of the composite cgrid (no underlying) on this processor).
 *--------------------------------------------------------------------------*/
#ifndef nalu_hypre_OWNINFODATA_HEADER
#define nalu_hypre_OWNINFODATA_HEADER


typedef struct
{
   NALU_HYPRE_Int             size;

   nalu_hypre_BoxArrayArray  *own_boxes;    /* size of fgrid */
   NALU_HYPRE_Int           **own_cboxnums; /* local cbox number- each fbox
                                          leads to an array of cboxes */

   nalu_hypre_BoxArrayArray  *own_composite_cboxes;  /* size of cgrid */
   NALU_HYPRE_Int             own_composite_size;
} nalu_hypre_SStructOwnInfoData;


/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructOwnInfoData;
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructOwnInfoDataSize(own_data)       ((own_data) -> size)
#define nalu_hypre_SStructOwnInfoDataOwnBoxes(own_data)   ((own_data) -> own_boxes)
#define nalu_hypre_SStructOwnInfoDataOwnBoxNums(own_data) \
((own_data) -> own_cboxnums)
#define nalu_hypre_SStructOwnInfoDataCompositeCBoxes(own_data) \
((own_data) -> own_composite_cboxes)
#define nalu_hypre_SStructOwnInfoDataCompositeSize(own_data) \
((own_data) -> own_composite_size)

#endif
