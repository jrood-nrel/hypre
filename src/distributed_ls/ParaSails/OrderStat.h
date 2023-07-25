/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * OrderStat.h header file.
 *
 *****************************************************************************/

#ifndef _ORDERSTAT_H
#define _ORDERSTAT_H

#include "_nalu_hypre_utilities.h"

NALU_HYPRE_Real randomized_select(NALU_HYPRE_Real *a, NALU_HYPRE_Int p, NALU_HYPRE_Int r, NALU_HYPRE_Int i);
void nalu_hypre_shell_sort(const NALU_HYPRE_Int n, NALU_HYPRE_Int x[]);

#endif /* _ORDERSTAT_H */
