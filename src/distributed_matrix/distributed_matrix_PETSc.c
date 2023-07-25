/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_DistributedMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

/* Public headers and prototypes for PETSc matrix library */
#ifdef PETSC_AVAILABLE
#include "sles.h"
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixDestroyPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixDestroyPETSc( nalu_hypre_DistributedMatrix *distributed_matrix )
{
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) nalu_hypre_DistributedMatrixLocalStorage(distributed_matrix);

   MatDestroy( PETSc_matrix );
#endif

   return(0);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixPrintPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixPrintPETSc( nalu_hypre_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   ierr = MatView( PETSc_matrix, VIEWER_STDOUT_WORLD );
#endif
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetLocalRangePETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixGetLocalRangePETSc( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt *start,
                             NALU_HYPRE_BigInt *end )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (!PETSc_matrix) return(-1);


   ierr = MatGetOwnershipRange( PETSc_matrix, start, end ); CHKERRA(ierr);
/*

  Since PETSc's MatGetOwnershipRange actually returns 
  end = "one more than the global index of the last local row",
  we need to subtract one; hypre assumes we return the index
  of the last row itself.

*/
   *end = *end - 1;
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetRowPETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixGetRowPETSc( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (!PETSc_matrix) return(-1);

   ierr = MatGetRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixRestoreRowPETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixRestoreRowPETSc( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) nalu_hypre_DistributedMatrixLocalStorage(matrix);

   if (PETSc_matrix == NULL) return(-1);

   ierr = MatRestoreRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}
