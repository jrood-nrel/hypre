/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

void
hypre_F90_IFACE(hypre_init, NALU_HYPRE_INIT)
(hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_Init();
}

void
hypre_F90_IFACE(hypre_finalize, NALU_HYPRE_FINALIZE)
(hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_Finalize();
}

void
hypre_F90_IFACE(hypre_setmemorylocation, NALU_HYPRE_SETMEMORYLOCATION)
(hypre_F90_Int *memory_location, hypre_F90_Int *ierr)
{
   NALU_HYPRE_MemoryLocation loc = (NALU_HYPRE_MemoryLocation) * memory_location;
   *ierr = (hypre_F90_Int) NALU_HYPRE_SetMemoryLocation(loc);
}

void
hypre_F90_IFACE(hypre_setexecutionpolicy, NALU_HYPRE_SETEXECUTIONPOLICY)
(hypre_F90_Int *exec_policy, hypre_F90_Int *ierr)
{
   NALU_HYPRE_ExecutionPolicy exec = (NALU_HYPRE_ExecutionPolicy) * exec_policy;

   *ierr = (hypre_F90_Int) NALU_HYPRE_SetExecutionPolicy(exec);
}

void
hypre_F90_IFACE(hypre_setspgemmusevendor, NALU_HYPRE_SETSPGEMMUSEVENDOR)
(hypre_F90_Int *use_vendor, hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_SetSpGemmUseVendor(*use_vendor);
}

#ifdef __cplusplus
}
#endif
