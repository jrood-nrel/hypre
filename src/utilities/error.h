/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ERROR_HEADER
#define nalu_hypre_ERROR_HEADER

#include <assert.h>

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int  error_flag;
   NALU_HYPRE_Int  print_to_memory;
   char      *memory;
   NALU_HYPRE_Int  mem_sz;
   NALU_HYPRE_Int  msg_sz;

} nalu_hypre_Error;

extern nalu_hypre_Error nalu_hypre__global_error;
#define nalu_hypre_error_flag  nalu_hypre__global_error.error_flag

/*--------------------------------------------------------------------------
 * NALU_HYPRE error macros
 *--------------------------------------------------------------------------*/

void nalu_hypre_error_handler(const char *filename, NALU_HYPRE_Int line, NALU_HYPRE_Int ierr, const char *msg);

#define nalu_hypre_error(IERR)  nalu_hypre_error_handler(__FILE__, __LINE__, IERR, NULL)
#define nalu_hypre_error_w_msg(IERR, msg)  nalu_hypre_error_handler(__FILE__, __LINE__, IERR, msg)
#define nalu_hypre_error_in_arg(IARG)  nalu_hypre_error(NALU_HYPRE_ERROR_ARG | IARG<<3)

#if defined(NALU_HYPRE_DEBUG)
/* host assert */
#define nalu_hypre_assert(EX) do { if (!(EX)) { fprintf(stderr, "[%s, %d] nalu_hypre_assert failed: %s\n", __FILE__, __LINE__, #EX); nalu_hypre_error(1); assert(0); } } while (0)
/* device assert */
#if defined(NALU_HYPRE_USING_CUDA)
#define nalu_hypre_device_assert(EX) assert(EX)
#elif defined(NALU_HYPRE_USING_HIP)
/* FIXME: Currently, asserts in device kernels in HIP do not behave well */
#define nalu_hypre_device_assert(EX) do { if (0) { static_cast<void> (EX); } } while (0)
#elif defined(NALU_HYPRE_USING_SYCL)
#define nalu_hypre_device_assert(EX) assert(EX)
#endif
#else /* #ifdef NALU_HYPRE_DEBUG */
/* this is to silence compiler's unused variable warnings */
#ifdef __cplusplus
#define nalu_hypre_assert(EX) do { if (0) { static_cast<void> (EX); } } while (0)
#else
#define nalu_hypre_assert(EX) do { if (0) { (void) (EX); } } while (0)
#endif
#define nalu_hypre_device_assert(EX)
#endif

#endif /* nalu_hypre_ERROR_HEADER */

