/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct
{
   NALU_HYPRE_Int             size;

   hypre_BoxArrayArray  *send_boxes;
   NALU_HYPRE_Int           **send_procs;
   NALU_HYPRE_Int           **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
