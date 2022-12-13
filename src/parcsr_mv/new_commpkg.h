/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG

typedef struct
{
   NALU_HYPRE_Int       length;
   NALU_HYPRE_Int       storage_length;
   NALU_HYPRE_Int      *id;
   NALU_HYPRE_Int      *vec_starts;
   NALU_HYPRE_Int       element_storage_length;
   NALU_HYPRE_BigInt   *elements;
   NALU_HYPRE_Real     *d_elements; /* Is this used anywhere? */
   void           *v_elements;
}  hypre_ProcListElements;

#endif /* hypre_NEW_COMMPKG */

