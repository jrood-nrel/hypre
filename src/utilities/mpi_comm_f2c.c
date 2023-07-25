/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <NALU_HYPRE_config.h>
#include "fortran.h"
#ifndef NALU_HYPRE_SEQUENTIAL
#include <mpi.h>
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0 /* This function is problematic and no longer needed anyway. */
void
nalu_hypre_F90_IFACE(nalu_hypre_mpi_comm_f2c, NALU_HYPRE_MPI_COMM_F2C)
(nalu_hypre_F90_Obj  *c_comm,
 nalu_hypre_F90_Comm *f_comm,
 nalu_hypre_F90_Int  *ierr)
{
   *c_comm = (nalu_hypre_F90_Obj) nalu_hypre_MPI_Comm_f2c( (nalu_hypre_int) * f_comm );
   *ierr = 0;
}
#endif
