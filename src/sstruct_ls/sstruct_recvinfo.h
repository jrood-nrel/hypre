/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef nalu_hypre_RECVINFODATA_HEADER
#define nalu_hypre_RECVINFODATA_HEADER


typedef struct
{
   NALU_HYPRE_Int             size;

   nalu_hypre_BoxArrayArray  *recv_boxes;
   NALU_HYPRE_Int           **recv_procs;

} nalu_hypre_SStructRecvInfoData;

#endif
