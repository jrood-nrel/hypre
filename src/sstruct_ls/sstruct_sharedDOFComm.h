/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

typedef struct
{
   NALU_HYPRE_BigInt row;

   NALU_HYPRE_Int ncols;
   NALU_HYPRE_BigInt      *cols;
   NALU_HYPRE_Real   *data;

} nalu_hypre_MaxwellOffProcRow;

