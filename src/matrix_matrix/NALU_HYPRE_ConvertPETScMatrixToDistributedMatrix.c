/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Routine for building a DistributedMatrix from a MPIAIJ Mat, i.e. PETSc matrix
 *
 *****************************************************************************/

#ifdef NALU_HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include <NALU_HYPRE_config.h>

#include "general.h"

#include "NALU_HYPRE.h"
#include "NALU_HYPRE_utilities.h"

/* Prototypes for DistributedMatrix */
#include "NALU_HYPRE_distributed_matrix_types.h"
#include "NALU_HYPRE_distributed_matrix_protos.h"

#ifdef PETSC_AVAILABLE

/* Matrix structure from PETSc */
#include "sles.h"
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ConvertPETScMatrixToDistributedMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ConvertPETScMatrixToDistributedMatrix(
   Mat PETSc_matrix,
   NALU_HYPRE_DistributedMatrix *DistributedMatrix )
{
   NALU_HYPRE_Int ierr;
   MPI_Comm hypre_MPI_Comm;
   NALU_HYPRE_BigInt M, N;
#ifdef NALU_HYPRE_TIMING
   NALU_HYPRE_Int           timer;
#endif



   if (!PETSc_matrix) { return (-1); }

#ifdef NALU_HYPRE_TIMING
   timer = hypre_InitializeTiming( "ConvertPETScMatrixToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   ierr = PetscObjectGetComm( (PetscObject) PETSc_matrix, &MPI_Comm); CHKERRA(ierr);

   ierr = NALU_HYPRE_DistributedMatrixCreate( MPI_Comm, DistributedMatrix );
   /* if(ierr) return(ierr); */

   ierr = NALU_HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix,
                                                      NALU_HYPRE_PETSC );
   /* if(ierr) return(ierr);*/

   ierr = NALU_HYPRE_DistributedMatrixInitialize( *DistributedMatrix );
   /* if(ierr) return(ierr);*/

   ierr = NALU_HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, PETSc_matrix );
   /* if(ierr) return(ierr); */
   /* Note that this is kind of cheating, since the Mat structure contains more
      than local information... the alternative is to extract the global info
      from the Mat and put it into DistributedMatrixAuxiliaryStorage. However,
      the latter is really a "just in case" option, and so if we don't *have*
      to use it, we won't.*/

   ierr = MatGetSize( PETSc_matrix, &M, &N);
   if (ierr) { return (ierr); }
   ierr = NALU_HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   ierr = NALU_HYPRE_DistributedMatrixAssemble( *DistributedMatrix );
   /* if(ierr) return(ierr);*/

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return (0);
}

#endif
