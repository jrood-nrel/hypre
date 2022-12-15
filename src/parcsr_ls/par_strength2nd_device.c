/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_mv.h"
#include "NALU_HYPRE_IJ_mv.h"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_utilities.hpp"

#include "NALU_HYPRE_parcsr_ls.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_utilities.h"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

//-----------------------------------------------------------------------
NALU_HYPRE_Int
nalu_hypre_BoomerAMGCreate2ndSDevice( nalu_hypre_ParCSRMatrix  *S,
                                 NALU_HYPRE_Int           *CF_marker,
                                 NALU_HYPRE_Int            num_paths,
                                 NALU_HYPRE_BigInt        *coarse_row_starts,
                                 nalu_hypre_ParCSRMatrix **S2_ptr)
{
   NALU_HYPRE_Int           S_nr_local = nalu_hypre_ParCSRMatrixNumRows(S);
   nalu_hypre_CSRMatrix    *S_diag     = nalu_hypre_ParCSRMatrixDiag(S);
   nalu_hypre_CSRMatrix    *S_offd     = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int           S_diag_nnz = nalu_hypre_CSRMatrixNumNonzeros(S_diag);
   NALU_HYPRE_Int           S_offd_nnz = nalu_hypre_CSRMatrixNumNonzeros(S_offd);
   nalu_hypre_CSRMatrix    *Id, *SI_diag;
   nalu_hypre_ParCSRMatrix *S_XC, *S_CX, *S2;
   NALU_HYPRE_Int          *new_end;
   NALU_HYPRE_Complex       coeff = 2.0;

   /*
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(S);
   NALU_HYPRE_Int num_proc, myid;
   nalu_hypre_MPI_Comm_size(comm, &num_proc);
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   */

   /* 1. Create new matrix with added diagonal */
   nalu_hypre_GpuProfilingPushRange("Setup");

   /* give S data arrays */
   nalu_hypre_CSRMatrixData(S_diag) = nalu_hypre_TAlloc(NALU_HYPRE_Complex, S_diag_nnz, NALU_HYPRE_MEMORY_DEVICE );
   hypreDevice_ComplexFilln( nalu_hypre_CSRMatrixData(S_diag),
                             S_diag_nnz,
                             1.0 );

   nalu_hypre_CSRMatrixData(S_offd) = nalu_hypre_TAlloc(NALU_HYPRE_Complex, S_offd_nnz, NALU_HYPRE_MEMORY_DEVICE );
   hypreDevice_ComplexFilln( nalu_hypre_CSRMatrixData(S_offd),
                             S_offd_nnz,
                             1.0 );

   if (!nalu_hypre_ParCSRMatrixCommPkg(S))
   {
      nalu_hypre_MatvecCommPkgCreate(S);
   }

   /* S(C, :) and S(:, C) */
   nalu_hypre_ParCSRMatrixGenerate1DCFDevice(S, CF_marker, coarse_row_starts, NULL, &S_CX, &S_XC);

   nalu_hypre_assert(S_nr_local == nalu_hypre_ParCSRMatrixNumCols(S_CX));

   /* add coeff*I to S_CX */
   Id = nalu_hypre_CSRMatrixCreate( nalu_hypre_ParCSRMatrixNumRows(S_CX),
                               nalu_hypre_ParCSRMatrixNumCols(S_CX),
                               nalu_hypre_ParCSRMatrixNumRows(S_CX) );

   nalu_hypre_CSRMatrixInitialize_v2(Id, 0, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_sequence( nalu_hypre_CSRMatrixI(Id),
                       nalu_hypre_CSRMatrixI(Id) + nalu_hypre_ParCSRMatrixNumRows(S_CX) + 1,
                       0 );

   oneapi::dpl::counting_iterator<NALU_HYPRE_Int> count(0);
   new_end = hypreSycl_copy_if( count,
                                count + nalu_hypre_ParCSRMatrixNumCols(S_CX),
                                CF_marker,
                                nalu_hypre_CSRMatrixJ(Id),
                                is_nonnegative<NALU_HYPRE_Int>()  );
#else
   NALU_HYPRE_THRUST_CALL( sequence,
                      nalu_hypre_CSRMatrixI(Id),
                      nalu_hypre_CSRMatrixI(Id) + nalu_hypre_ParCSRMatrixNumRows(S_CX) + 1,
                      0  );

   new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(nalu_hypre_ParCSRMatrixNumCols(S_CX)),
                                CF_marker,
                                nalu_hypre_CSRMatrixJ(Id),
                                is_nonnegative<NALU_HYPRE_Int>()  );
#endif

   nalu_hypre_assert(new_end - nalu_hypre_CSRMatrixJ(Id) == nalu_hypre_ParCSRMatrixNumRows(S_CX));

   hypreDevice_ComplexFilln( nalu_hypre_CSRMatrixData(Id),
                             nalu_hypre_ParCSRMatrixNumRows(S_CX),
                             coeff );

   SI_diag = nalu_hypre_CSRMatrixAddDevice(1.0, nalu_hypre_ParCSRMatrixDiag(S_CX), 1.0, Id);

   nalu_hypre_CSRMatrixDestroy(Id);

   /* global nnz has changed, but we do not care about it */
   /*
   nalu_hypre_ParCSRMatrixSetNumNonzeros(S_CX);
   nalu_hypre_ParCSRMatrixDNumNonzeros(S_CX) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(S_CX);
   */

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(S_CX));
   nalu_hypre_ParCSRMatrixDiag(S_CX) = SI_diag;

   nalu_hypre_GpuProfilingPopRange();

   /* 2. Perform matrix-matrix multiplication */
   nalu_hypre_GpuProfilingPushRange("Matrix-matrix mult");

   S2 = nalu_hypre_ParCSRMatMatDevice(S_CX, S_XC);

   nalu_hypre_ParCSRMatrixDestroy(S_CX);
   nalu_hypre_ParCSRMatrixDestroy(S_XC);

   nalu_hypre_GpuProfilingPopRange();

   // Clean up matrix before returning it.
   if (num_paths == 2)
   {
      // If num_paths = 2, prune elements < 2.
      nalu_hypre_ParCSRMatrixDropSmallEntries(S2, 1.5, 0);
   }

   nalu_hypre_TFree(nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(S2)), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(S2)), NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixRemoveDiagonalDevice(nalu_hypre_ParCSRMatrixDiag(S2));

   /* global nnz has changed, but we do not care about it */

   nalu_hypre_MatvecCommPkgCreate(S2);

   *S2_ptr = S2;

   return 0;
}

#endif /* #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL) */
