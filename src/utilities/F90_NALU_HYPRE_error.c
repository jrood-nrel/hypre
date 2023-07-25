/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

void
nalu_hypre_F90_IFACE(nalu_hypre_geterror, NALU_HYPRE_GETERROR)
(nalu_hypre_F90_Int *result)
{
   *result = (nalu_hypre_F90_Int) NALU_HYPRE_GetError();
}

void
nalu_hypre_F90_IFACE(nalu_hypre_checkerror, NALU_HYPRE_CHECKERROR)
(nalu_hypre_F90_Int *ierr,
 nalu_hypre_F90_Int *nalu_hypre_error_code,
 nalu_hypre_F90_Int *result)
{
   *result = (nalu_hypre_F90_Int) NALU_HYPRE_CheckError(
                nalu_hypre_F90_PassInt(ierr),
                nalu_hypre_F90_PassInt(nalu_hypre_error_code));
}

void
nalu_hypre_F90_IFACE(nalu_hypre_geterrorarg, NALU_HYPRE_GETERRORARG)
(nalu_hypre_F90_Int *result)
{
   *result = (nalu_hypre_F90_Int) NALU_HYPRE_GetErrorArg();
}

void
nalu_hypre_F90_IFACE(nalu_hypre_clearallerrors, NALU_HYPRE_CLEARALLERRORS)
(nalu_hypre_F90_Int *result)
{
   *result = NALU_HYPRE_ClearAllErrors();
}

void
nalu_hypre_F90_IFACE(nalu_hypre_clearerror, NALU_HYPRE_CLEARERROR)
(nalu_hypre_F90_Int *nalu_hypre_error_code,
 nalu_hypre_F90_Int *result)
{
   *result = (nalu_hypre_F90_Int) NALU_HYPRE_ClearError(
                nalu_hypre_F90_PassInt(nalu_hypre_error_code));
}

#ifdef __cplusplus
}
#endif
