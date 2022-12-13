/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_DistributedMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

/* Public headers and prototypes for PETSc matrix library */
#ifdef PETSC_AVAILABLE
#include "sles.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixDestroyPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixDestroyPETSc( hypre_DistributedMatrix *distributed_matrix )
{
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(distributed_matrix);

   MatDestroy( PETSc_matrix );
#endif

   return(0);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrintPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixPrintPETSc( hypre_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   ierr = MatView( PETSc_matrix, VIEWER_STDOUT_WORLD );
#endif
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangePETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixGetLocalRangePETSc( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt *start,
                             NALU_HYPRE_BigInt *end )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

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
 * hypre_DistributedMatrixGetRowPETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixGetRowPETSc( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   if (!PETSc_matrix) return(-1);

   ierr = MatGetRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowPETSc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixRestoreRowPETSc( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   if (PETSc_matrix == NULL) return(-1);

   ierr = MatRestoreRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}
