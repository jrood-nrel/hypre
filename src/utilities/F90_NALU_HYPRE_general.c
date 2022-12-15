/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

void
nalu_hypre_F90_IFACE(nalu_hypre_init, NALU_HYPRE_INIT)
(nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_Init();
}

void
nalu_hypre_F90_IFACE(nalu_hypre_finalize, NALU_HYPRE_FINALIZE)
(nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_Finalize();
}

void
nalu_hypre_F90_IFACE(nalu_hypre_setmemorylocation, NALU_HYPRE_SETMEMORYLOCATION)
(nalu_hypre_F90_Int *memory_location, nalu_hypre_F90_Int *ierr)
{
   NALU_HYPRE_MemoryLocation loc = (NALU_HYPRE_MemoryLocation) * memory_location;
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SetMemoryLocation(loc);
}

void
nalu_hypre_F90_IFACE(nalu_hypre_setexecutionpolicy, NALU_HYPRE_SETEXECUTIONPOLICY)
(nalu_hypre_F90_Int *exec_policy, nalu_hypre_F90_Int *ierr)
{
   NALU_HYPRE_ExecutionPolicy exec = (NALU_HYPRE_ExecutionPolicy) * exec_policy;

   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SetExecutionPolicy(exec);
}

void
nalu_hypre_F90_IFACE(nalu_hypre_setspgemmusevendor, NALU_HYPRE_SETSPGEMMUSEVENDOR)
(nalu_hypre_F90_Int *use_vendor, nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_SetSpGemmUseVendor(*use_vendor);
}

#ifdef __cplusplus
}
#endif
