/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_OMP_DEVICE_H
#define NALU_HYPRE_OMP_DEVICE_H

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)

#include "omp.h"

/* OpenMP 4.5 device memory management */
extern NALU_HYPRE_Int nalu_hypre__global_offload;
extern NALU_HYPRE_Int nalu_hypre__offload_device_num;
extern NALU_HYPRE_Int nalu_hypre__offload_host_num;

/* stats */
extern size_t nalu_hypre__target_allc_count;
extern size_t nalu_hypre__target_free_count;
extern size_t nalu_hypre__target_allc_bytes;
extern size_t nalu_hypre__target_free_bytes;
extern size_t nalu_hypre__target_htod_count;
extern size_t nalu_hypre__target_dtoh_count;
extern size_t nalu_hypre__target_htod_bytes;
extern size_t nalu_hypre__target_dtoh_bytes;

/* CHECK MODE: check if offloading has effect (turned on when configured with --enable-debug)
 * if we ``enter'' an address, it should not exist in device [o.w NO EFFECT]
 * if we ``exit'' or ''update'' an address, it should exist in device [o.w ERROR]
 * nalu_hypre__offload_flag: 0 == OK; 1 == WRONG
 */
#ifdef NALU_HYPRE_DEVICE_OPENMP_CHECK
#define NALU_HYPRE_OFFLOAD_FLAG(devnum, hptr, type) NALU_HYPRE_Int nalu_hypre__offload_flag = (type[1] == 'n') == omp_target_is_present(hptr, devnum);
#else
#define NALU_HYPRE_OFFLOAD_FLAG(...) NALU_HYPRE_Int nalu_hypre__offload_flag = 0; /* non-debug mode, always OK */
#endif

/* OMP 4.5 offloading macro */
#define nalu_hypre_omp_device_offload(devnum, hptr, datatype, offset, count, type1, type2) \
{\
   /* devnum: device number \
    * hptr: host poiter \
    * datatype \
    * type1: ``e(n)ter'', ''e(x)it'', or ``u(p)date'' \
    * type2: ``(a)lloc'', ``(t)o'', ``(d)elete'', ''(f)rom'' \
    */ \
   datatype *nalu_hypre__offload_hptr = (datatype *) hptr; \
   /* if nalu_hypre__global_offload ==    0, or
    *    hptr (host pointer)   == NULL,
    *    this offload will be IGNORED */ \
   if (nalu_hypre__global_offload && nalu_hypre__offload_hptr != NULL) { \
      /* offloading offset and size (in datatype) */ \
      size_t nalu_hypre__offload_offset = offset, nalu_hypre__offload_size = count; \
      /* in the CHECK mode, we test if this offload has effect */ \
      NALU_HYPRE_OFFLOAD_FLAG(devnum, nalu_hypre__offload_hptr, type1) \
      if (nalu_hypre__offload_flag) { \
         printf("[!NO Effect! %s %d] device %d target: %6s %6s, data %p, [%ld:%ld]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)nalu_hypre__offload_hptr, nalu_hypre__offload_offset, nalu_hypre__offload_size); exit(0); \
      } else { \
         size_t offload_bytes = count * sizeof(datatype); \
         /* printf("[            %s %d] device %d target: %6s %6s, data %p, [%d:%d]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)nalu_hypre__offload_hptr, nalu_hypre__offload_offset, nalu_hypre__offload_size); */ \
         if (type1[1] == 'n' && type2[0] == 't') { \
            /* enter to */\
            nalu_hypre__target_allc_count ++; \
            nalu_hypre__target_allc_bytes += offload_bytes; \
            nalu_hypre__target_htod_count ++; \
            nalu_hypre__target_htod_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target enter data map(to:nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else if (type1[1] == 'n' && type2[0] == 'a') { \
            /* enter alloc */ \
            nalu_hypre__target_allc_count ++; \
            nalu_hypre__target_allc_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target enter data map(alloc:nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'd') { \
            /* exit delete */\
            nalu_hypre__target_free_count ++; \
            nalu_hypre__target_free_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target exit data map(delete:nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'f') {\
            /* exit from */ \
            nalu_hypre__target_free_count ++; \
            nalu_hypre__target_free_bytes += offload_bytes; \
            nalu_hypre__target_dtoh_count ++; \
            nalu_hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target exit data map(from:nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 't') { \
            /* update to */ \
            nalu_hypre__target_htod_count ++; \
            nalu_hypre__target_htod_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target update to(nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 'f') {\
            /* update from */ \
            nalu_hypre__target_dtoh_count ++; \
            nalu_hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (NALU_HYPRE_XSTR(omp target update from(nalu_hypre__offload_hptr[nalu_hypre__offload_offset:nalu_hypre__offload_size]))) \
         } else {\
            printf("error: unrecognized offloading type combination!\n"); exit(-1); \
         } \
      } \
   } \
}

NALU_HYPRE_Int NALU_HYPRE_OMPOffload(NALU_HYPRE_Int device, void *ptr, size_t num, const char *type1,
                           const char *type2);
NALU_HYPRE_Int NALU_HYPRE_OMPPtrIsMapped(void *p, NALU_HYPRE_Int device_num);
NALU_HYPRE_Int NALU_HYPRE_OMPOffloadOn(void);
NALU_HYPRE_Int NALU_HYPRE_OMPOffloadOff(void);
NALU_HYPRE_Int NALU_HYPRE_OMPOffloadStatPrint(void);

#endif /* NALU_HYPRE_USING_DEVICE_OPENMP */
#endif /* NALU_HYPRE_OMP_DEVICE_H */

