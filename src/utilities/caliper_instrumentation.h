/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for Caliper instrumentation macros
 *
 *****************************************************************************/

#ifndef CALIPER_INSTRUMENTATION_HEADER
#define CALIPER_INSTRUMENTATION_HEADER

#include "NALU_HYPRE_config.h"

#ifdef NALU_HYPRE_USING_CALIPER

#ifdef __cplusplus
extern "C++"
{
#endif

#include <caliper/cali.h>

#ifdef __cplusplus
}
#endif

#define NALU_HYPRE_ANNOTATE_FUNC_BEGIN          CALI_MARK_FUNCTION_BEGIN
#define NALU_HYPRE_ANNOTATE_FUNC_END            CALI_MARK_FUNCTION_END
#define NALU_HYPRE_ANNOTATE_LOOP_BEGIN(id, str) CALI_MARK_LOOP_BEGIN(id, str)
#define NALU_HYPRE_ANNOTATE_LOOP_END(id)        CALI_MARK_LOOP_END(id)
#define NALU_HYPRE_ANNOTATE_ITER_BEGIN(id, it)  CALI_MARK_ITERATION_BEGIN(id, it)
#define NALU_HYPRE_ANNOTATE_ITER_END(id)        CALI_MARK_ITERATION_END(id)
#define NALU_HYPRE_ANNOTATE_REGION_BEGIN(...)\
{\
   char nalu_hypre__markname[1024];\
   nalu_hypre_sprintf(nalu_hypre__markname, __VA_ARGS__);\
   CALI_MARK_BEGIN(nalu_hypre__markname);\
}
#define NALU_HYPRE_ANNOTATE_REGION_END(...)\
{\
   char nalu_hypre__markname[1024];\
   nalu_hypre_sprintf(nalu_hypre__markname, __VA_ARGS__);\
   CALI_MARK_END(nalu_hypre__markname);\
}
#define NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)\
{\
   char nalu_hypre__levelname[16];\
   nalu_hypre_sprintf(nalu_hypre__levelname, "MG level %d", lvl);\
   CALI_MARK_BEGIN(nalu_hypre__levelname);\
}
#define NALU_HYPRE_ANNOTATE_MGLEVEL_END(lvl)\
{\
   char nalu_hypre__levelname[16];\
   nalu_hypre_sprintf(nalu_hypre__levelname, "MG level %d", lvl);\
   CALI_MARK_END(nalu_hypre__levelname);\
}

#else

#define NALU_HYPRE_ANNOTATE_FUNC_BEGIN
#define NALU_HYPRE_ANNOTATE_FUNC_END
#define NALU_HYPRE_ANNOTATE_LOOP_BEGIN(id, str)
#define NALU_HYPRE_ANNOTATE_LOOP_END(id)
#define NALU_HYPRE_ANNOTATE_ITER_BEGIN(id, it)
#define NALU_HYPRE_ANNOTATE_ITER_END(id)
#define NALU_HYPRE_ANNOTATE_REGION_BEGIN(...)
#define NALU_HYPRE_ANNOTATE_REGION_END(...)
#define NALU_HYPRE_ANNOTATE_MAX_MGLEVEL(lvl)
#define NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(lvl)
#define NALU_HYPRE_ANNOTATE_MGLEVEL_END(lvl)

#endif

#endif /* CALIPER_INSTRUMENTATION_HEADER */
