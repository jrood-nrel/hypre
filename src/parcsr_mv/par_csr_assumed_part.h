/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef nalu_hypre_PARCSR_ASSUMED_PART
#define nalu_hypre_PARCSR_ASSUMED_PART

typedef struct
{
   NALU_HYPRE_Int                   length;
   NALU_HYPRE_BigInt                row_start;
   NALU_HYPRE_BigInt                row_end;
   NALU_HYPRE_Int                   storage_length;
   NALU_HYPRE_Int                  *proc_list;
   NALU_HYPRE_BigInt               *row_start_list;
   NALU_HYPRE_BigInt               *row_end_list;
   NALU_HYPRE_Int                  *sort_index;
} nalu_hypre_IJAssumedPart;

#endif /* nalu_hypre_PARCSR_ASSUMED_PART */

