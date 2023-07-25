/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */

#include "_nalu_hypre_onedpl.hpp"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_GPU)

/**
  Generates global coarse_size and dof_func for next coarser level

  Notes:
  \begin{itemize}
  \item The routine returns the following:
  \begin{itemize}
  \item an integer array containing the
  function values for the local coarse points
  \item the global number of coarse points
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  _nalu_hypre_parcsr_ls.h

  @return Error code.

  @param comm [IN]
  MPI Communicator
  @param local_num_variables [IN]
  number of points on local processor
  @param dof_func [IN]
  array that contains the function numbers for all local points
  @param CF_marker [IN]
  marker array for coarse points
  @param coarse_dof_func_ptr [OUT]
  pointer to array which contains the function numbers for local coarse points
  @param coarse_pnts_global [OUT]
  pointer to array which contains the number of the first coarse point on each  processor and the total number of coarse points in its last element

  @see */
/*--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCoarseParmsDevice(MPI_Comm          comm,
                                 NALU_HYPRE_Int         local_num_variables,
                                 NALU_HYPRE_Int         num_functions,
                                 nalu_hypre_IntArray   *dof_func,
                                 nalu_hypre_IntArray   *CF_marker,
                                 nalu_hypre_IntArray  **coarse_dof_func_ptr,
                                 NALU_HYPRE_BigInt     *coarse_pnts_global)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_COARSE_PARAMS] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int      ierr = 0;
   NALU_HYPRE_BigInt   local_coarse_size = 0;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_SYCL)
   local_coarse_size = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                          nalu_hypre_IntArrayData(CF_marker),
                                          nalu_hypre_IntArrayData(CF_marker) + local_num_variables,
                                          equal<NALU_HYPRE_Int>(1) );
#else
   local_coarse_size = NALU_HYPRE_THRUST_CALL( count_if,
                                          nalu_hypre_IntArrayData(CF_marker),
                                          nalu_hypre_IntArrayData(CF_marker) + local_num_variables,
                                          equal<NALU_HYPRE_Int>(1) );
#endif

   if (num_functions > 1)
   {
      *coarse_dof_func_ptr = nalu_hypre_IntArrayCreate(local_coarse_size);
      nalu_hypre_IntArrayInitialize_v2(*coarse_dof_func_ptr, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      hypreSycl_copy_if( nalu_hypre_IntArrayData(dof_func),
                         nalu_hypre_IntArrayData(dof_func) + local_num_variables,
                         nalu_hypre_IntArrayData(CF_marker),
                         nalu_hypre_IntArrayData(*coarse_dof_func_ptr),
                         equal<NALU_HYPRE_Int>(1) );
#else
      NALU_HYPRE_THRUST_CALL( copy_if,
                         nalu_hypre_IntArrayData(dof_func),
                         nalu_hypre_IntArrayData(dof_func) + local_num_variables,
                         nalu_hypre_IntArrayData(CF_marker),
                         nalu_hypre_IntArrayData(*coarse_dof_func_ptr),
                         equal<NALU_HYPRE_Int>(1) );
#endif
   }

   {
      NALU_HYPRE_BigInt scan_recv;
      nalu_hypre_MPI_Scan(&local_coarse_size, &scan_recv, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      /* first point in my range */
      coarse_pnts_global[0] = scan_recv - local_coarse_size;

      /* first point in next proc's range */
      coarse_pnts_global[1] = scan_recv;
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_COARSE_PARAMS] += nalu_hypre_MPI_Wtime();
#endif

   return (ierr);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGInitDofFuncDevice( NALU_HYPRE_Int *dof_func,
                                  NALU_HYPRE_Int  local_size,
                                  NALU_HYPRE_Int  offset,
                                  NALU_HYPRE_Int  num_functions )
{
#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_sequence(dof_func,
                      dof_func + local_size,
                      offset);

   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      dof_func,
                      dof_func + local_size,
                      dof_func,
                      modulo<NALU_HYPRE_Int>(num_functions) );
#else
   NALU_HYPRE_THRUST_CALL( sequence,
                      dof_func,
                      dof_func + local_size,
                      offset,
                      1 );

   NALU_HYPRE_THRUST_CALL( transform,
                      dof_func,
                      dof_func + local_size,
                      dof_func,
                      modulo<NALU_HYPRE_Int>(num_functions) );
#endif

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_GPU)

