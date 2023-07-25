/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_PAR_MAKE_SYSTEM
#define nalu_hypre_PAR_MAKE_SYSTEM

typedef struct
{
   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_ParVector    *x;
   nalu_hypre_ParVector    *b;
} NALU_HYPRE_ParCSR_System_Problem;

#endif /* nalu_hypre_PAR_MAKE_SYSTEM */

