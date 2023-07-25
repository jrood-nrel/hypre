/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

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
nalu_hypre_BoomerAMGCoarseParmsHost(MPI_Comm         comm,
                               NALU_HYPRE_Int        local_num_variables,
                               NALU_HYPRE_Int        num_functions,
                               nalu_hypre_IntArray  *dof_func,
                               nalu_hypre_IntArray  *CF_marker,
                               nalu_hypre_IntArray **coarse_dof_func_ptr,
                               NALU_HYPRE_BigInt    *coarse_pnts_global)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_COARSE_PARAMS] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int     i;
   NALU_HYPRE_BigInt  local_coarse_size = 0;
   NALU_HYPRE_Int    *coarse_dof_func;

   /*--------------------------------------------------------------
    *----------------------------------------------------------------*/

   for (i = 0; i < local_num_variables; i++)
   {
      if (nalu_hypre_IntArrayData(CF_marker)[i] == 1)
      {
         local_coarse_size++;
      }
   }

   if (num_functions > 1)
   {
      *coarse_dof_func_ptr = nalu_hypre_IntArrayCreate(local_coarse_size);
      nalu_hypre_IntArrayInitialize(*coarse_dof_func_ptr);
      coarse_dof_func = nalu_hypre_IntArrayData(*coarse_dof_func_ptr);

      local_coarse_size = 0;
      for (i = 0; i < local_num_variables; i++)
      {
         if (nalu_hypre_IntArrayData(CF_marker)[i] == 1)
         {
            coarse_dof_func[local_coarse_size++] = nalu_hypre_IntArrayData(dof_func)[i];
         }
      }
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

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCoarseParms(MPI_Comm         comm,
                           NALU_HYPRE_Int        local_num_variables,
                           NALU_HYPRE_Int        num_functions,
                           nalu_hypre_IntArray  *dof_func,
                           nalu_hypre_IntArray  *CF_marker,
                           nalu_hypre_IntArray **coarse_dof_func_ptr,
                           NALU_HYPRE_BigInt    *coarse_pnts_global)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec;

   if (num_functions > 1)
   {
      exec = nalu_hypre_GetExecPolicy2(nalu_hypre_IntArrayMemoryLocation(CF_marker),
                                  nalu_hypre_IntArrayMemoryLocation(dof_func));
   }
   else
   {
      exec = nalu_hypre_GetExecPolicy1(nalu_hypre_IntArrayMemoryLocation(CF_marker));
   }

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      return nalu_hypre_BoomerAMGCoarseParmsDevice(comm, local_num_variables, num_functions, dof_func,
                                              CF_marker, coarse_dof_func_ptr, coarse_pnts_global);
   }
   else
#endif
   {
      return nalu_hypre_BoomerAMGCoarseParmsHost(comm, local_num_variables, num_functions, dof_func,
                                            CF_marker, coarse_dof_func_ptr, coarse_pnts_global);
   }
}
