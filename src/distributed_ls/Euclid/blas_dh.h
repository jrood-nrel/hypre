/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef THREADED_BLAS_DH
#define THREADED_BLAS_DH

/* notes: 1. all calls are threaded with OpenMP.
          2. for mpi MatVec, see "Mat_dhMatvec()" in Mat_dh.h
          3. MPI calls use nalu_hypre_MPI_COMM_WORLD for the communicator,
             where applicable.
*/

/* #include "euclid_common.h" */

#ifdef SEQUENTIAL_MODE
#define MatVec       matvec_euclid_seq
#endif

extern void matvec_euclid_seq(NALU_HYPRE_Int n, NALU_HYPRE_Int *rp, NALU_HYPRE_Int *cval, NALU_HYPRE_Real *aval, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);
extern NALU_HYPRE_Real InnerProd(NALU_HYPRE_Int local_n, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);
extern NALU_HYPRE_Real Norm2(NALU_HYPRE_Int local_n, NALU_HYPRE_Real *x);
extern void Axpy(NALU_HYPRE_Int n, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);
extern NALU_HYPRE_Real Norm2(NALU_HYPRE_Int n, NALU_HYPRE_Real *x);
extern void CopyVec(NALU_HYPRE_Int n, NALU_HYPRE_Real *xIN, NALU_HYPRE_Real *yOUT);
extern void ScaleVec(NALU_HYPRE_Int n, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x);

#endif
