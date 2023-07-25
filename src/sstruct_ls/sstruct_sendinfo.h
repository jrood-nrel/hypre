/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef nalu_hypre_SENDINFODATA_HEADER
#define nalu_hypre_SENDINFODATA_HEADER


typedef struct
{
   NALU_HYPRE_Int             size;

   nalu_hypre_BoxArrayArray  *send_boxes;
   NALU_HYPRE_Int           **send_procs;
   NALU_HYPRE_Int           **send_remote_boxnums;

} nalu_hypre_SStructSendInfoData;

#endif
