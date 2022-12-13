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
hypre_F90_IFACE(hypre_geterror, NALU_HYPRE_GETERROR)
(hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) NALU_HYPRE_GetError();
}

void
hypre_F90_IFACE(hypre_checkerror, NALU_HYPRE_CHECKERROR)
(hypre_F90_Int *ierr,
 hypre_F90_Int *hypre_error_code,
 hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) NALU_HYPRE_CheckError(
                hypre_F90_PassInt(ierr),
                hypre_F90_PassInt(hypre_error_code));
}

void
hypre_F90_IFACE(hypre_geterrorarg, NALU_HYPRE_GETERRORARG)
(hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) NALU_HYPRE_GetErrorArg();
}

void
hypre_F90_IFACE(hypre_clearallerrors, NALU_HYPRE_CLEARALLERRORS)
(hypre_F90_Int *result)
{
   *result = NALU_HYPRE_ClearAllErrors();
}

void
hypre_F90_IFACE(hypre_clearerror, NALU_HYPRE_CLEARERROR)
(hypre_F90_Int *hypre_error_code,
 hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) NALU_HYPRE_ClearError(
                hypre_F90_PassInt(hypre_error_code));
}

#ifdef __cplusplus
}
#endif
