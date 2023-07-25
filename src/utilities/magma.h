/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_MAGMA_HEADER
#define NALU_HYPRE_MAGMA_HEADER

#include "NALU_HYPRE_config.h"

#if defined(NALU_HYPRE_USING_MAGMA)

#include "error.h"

#ifdef __cplusplus
extern "C++"
{
#endif

#if !defined(MAGMA_GLOBAL)
#define ADD_
#endif
#include <magma_v2.h>

#ifdef __cplusplus
}
#endif

#define NALU_HYPRE_MAGMA_CALL(call) do {                   \
   magma_int_t err = call;                            \
   if (MAGMA_SUCCESS != err) {                        \
      printf("MAGMA ERROR (code = %d) at %s:%d\n",    \
            err, __FILE__, __LINE__);                 \
      nalu_hypre_assert(0);                                \
   } } while(0)

#endif /* NALU_HYPRE_USING_MAGMA */
#endif /* NALU_HYPRE_MAGMA_HEADER */
