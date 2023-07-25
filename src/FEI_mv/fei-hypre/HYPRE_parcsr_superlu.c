/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSR_SuperLU interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "NALU_HYPRE_FEI.h"

/*---------------------------------------------------------------------------
 * SUPERLU include files
 *-------------------------------------------------------------------------*/

#ifdef HAVE_SUPERLU_20
#include "dsp_defs.h"
#include "superlu_util.h"

typedef struct NALU_HYPRE_SuperLU_Struct
{
   int          factorized_;
   int          *permR_;
   int          *permC_;
   SuperMatrix  SLU_Lmat;
   SuperMatrix  SLU_Umat;
   int          outputLevel_;
}
NALU_HYPRE_SuperLU;
#endif

#ifdef HAVE_SUPERLU
#include "slu_ddefs.h"
#include "slu_util.h"

typedef struct NALU_HYPRE_SuperLU_Struct
{
   int          factorized_;
   int          *permR_;
   int          *permC_;
   SuperMatrix  SLU_Lmat;
   SuperMatrix  SLU_Umat;
   int          outputLevel_;
}
NALU_HYPRE_SuperLU;
#endif

/***************************************************************************
 * NALU_HYPRE_ParCSR_SuperLUCreate - Return a SuperLU object "solver".
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSR_SuperLUCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
#ifdef HAVE_SUPERLU
   int           nprocs;
   NALU_HYPRE_SuperLU *sluPtr;

   MPI_Comm_size(comm, &nprocs);
   if ( nprocs > 1 )
   {
      printf("NALU_HYPRE_ParCSR_SuperLUCreate ERROR - too many processors.\n");
      return -1;
   }
   sluPtr = nalu_hypre_TAlloc(NALU_HYPRE_SuperLU, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert ( sluPtr != NULL );
   sluPtr->factorized_  = 0;
   sluPtr->permR_       = NULL;
   sluPtr->permC_       = NULL;
   sluPtr->outputLevel_ = 0;
   *solver = (NALU_HYPRE_Solver) sluPtr;
   return 0;
#else
   printf("NALU_HYPRE_ParCSR_SuperLUCreate ERROR - SuperLU not enabled.\n");
   *solver = (NALU_HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * NALU_HYPRE_ParCSR_SuperLUDestroy - Destroy a SuperLU object.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSR_SuperLUDestroy( NALU_HYPRE_Solver solver )
{
#ifdef HAVE_SUPERLU
   NALU_HYPRE_SuperLU *sluPtr = (NALU_HYPRE_SuperLU *) solver;
   nalu_hypre_assert ( sluPtr != NULL );
   nalu_hypre_TFree(sluPtr->permR_, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sluPtr->permC_, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sluPtr, NALU_HYPRE_MEMORY_HOST);
   return 0;
#else
   printf("NALU_HYPRE_ParCSR_SuperLUDestroy ERROR - SuperLU not enabled.\n");
   *solver = (NALU_HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * NALU_HYPRE_ParCSR_SuperLUSetOutputLevel - Set debug level
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSR_SuperLUSetOutputLevel(NALU_HYPRE_Solver solver, int level)
{
#ifdef HAVE_SUPERLU
   NALU_HYPRE_SuperLU *sluPtr = (NALU_HYPRE_SuperLU *) solver;
   nalu_hypre_assert ( sluPtr != NULL );
   sluPtr->outputLevel_ = level;
   return 0;
#else
   printf("NALU_HYPRE_ParCSR_SuperLUSetOutputLevel ERROR - SuperLU not enabled.\n");
   *solver = (NALU_HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * NALU_HYPRE_ParCSR_SuperLUSetup - Set up function for SuperLU.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSR_SuperLUSetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                              NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x )
{
#ifdef HAVE_SUPERLU
   int    startRow, endRow, nrows, *partition, *AdiagI, *AdiagJ, nnz;
   int    irow, colNum, index, *cscI, *cscJ, jcol, *colLengs;
   int    *etree, permcSpec, lwork, panelSize, relax, info;
   double *AdiagA, *cscA, diagPivotThresh, dropTol;
   char              refact[1];
   nalu_hypre_CSRMatrix   *Adiag;
   NALU_HYPRE_SuperLU     *sluPtr;
   SuperMatrix       sluAmat, auxAmat;
   superlu_options_t slu_options;
   SuperLUStat_t     slu_stat;

   /* ---------------------------------------------------------------- */
   /* get matrix information                                           */
   /* ---------------------------------------------------------------- */

   sluPtr = (NALU_HYPRE_SuperLU *) solver;
   nalu_hypre_assert ( sluPtr != NULL );
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   startRow = partition[0];
   endRow   = partition[1] - 1;
   nrows    = endRow - startRow + 1;
   nalu_hypre_TFree(partition, NALU_HYPRE_MEMORY_HOST);
   if ( startRow != 0 )
   {
      printf("NALU_HYPRE_ParCSR_SuperLUSetup ERROR - start row != 0.\n");
      return -1;
   }

   /* ---------------------------------------------------------------- */
   /* get hypre matrix                                                 */
   /* ---------------------------------------------------------------- */

   Adiag  = nalu_hypre_ParCSRMatrixDiag((nalu_hypre_ParCSRMatrix *) A_csr);
   AdiagI = nalu_hypre_CSRMatrixI(Adiag);
   AdiagJ = nalu_hypre_CSRMatrixJ(Adiag);
   AdiagA = nalu_hypre_CSRMatrixData(Adiag);
   nnz    = AdiagI[nrows];

   /* ---------------------------------------------------------------- */
   /* convert the csr matrix into csc matrix                           */
   /* ---------------------------------------------------------------- */

   colLengs = nalu_hypre_TAlloc(int, nrows , NALU_HYPRE_MEMORY_HOST);
   for ( irow = 0; irow < nrows; irow++ ) colLengs[irow] = 0;
   for ( irow = 0; irow < nrows; irow++ )
      for ( jcol = AdiagI[irow]; jcol < AdiagI[irow+1]; jcol++ )
         colLengs[AdiagJ[jcol]]++;
   cscJ = nalu_hypre_TAlloc(int,  (nrows+1) , NALU_HYPRE_MEMORY_HOST);
   cscI = nalu_hypre_TAlloc(int,  nnz , NALU_HYPRE_MEMORY_HOST);
   cscA = nalu_hypre_TAlloc(double,  nnz , NALU_HYPRE_MEMORY_HOST);
   cscJ[0] = 0;
   nnz = 0;
   for ( jcol = 1; jcol <= nrows; jcol++ )
   {
      nnz += colLengs[jcol-1];
      cscJ[jcol] = nnz;
   }
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( jcol = AdiagI[irow]; jcol < AdiagI[irow+1]; jcol++ )
      {
         colNum = AdiagJ[jcol];
         index  = cscJ[colNum]++;
         cscI[index] = irow;
         cscA[index] = AdiagA[jcol];
      }
   }
   cscJ[0] = 0;
   nnz = 0;
   for ( jcol = 1; jcol <= nrows; jcol++ )
   {
      nnz += colLengs[jcol-1];
      cscJ[jcol] = nnz;
   }
   nalu_hypre_TFree(colLengs, NALU_HYPRE_MEMORY_HOST);

   /* ---------------------------------------------------------------- */
   /* create SuperMatrix                                                */
   /* ---------------------------------------------------------------- */

   dCreate_CompCol_Matrix(&sluAmat,nrows,nrows,cscJ[nrows],cscA,cscI,
                          cscJ, SLU_NC, SLU_D, SLU_GE);
   etree   = nalu_hypre_TAlloc(int, nrows , NALU_HYPRE_MEMORY_HOST);
   sluPtr->permC_  = nalu_hypre_TAlloc(int, nrows , NALU_HYPRE_MEMORY_HOST);
   sluPtr->permR_  = nalu_hypre_TAlloc(int, nrows , NALU_HYPRE_MEMORY_HOST);
   permcSpec = 0;
   get_perm_c(permcSpec, &sluAmat, sluPtr->permC_);
   slu_options.Fact = DOFACT;
   slu_options.SymmetricMode = NO;
   sp_preorder(&slu_options, &sluAmat, sluPtr->permC_, etree, &auxAmat);
   diagPivotThresh = 1.0;
   dropTol = 0.0;
   panelSize = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(&slu_stat);
   lwork = 0;
   slu_options.ColPerm = MY_PERMC;
   slu_options.DiagPivotThresh = diagPivotThresh;

   dgstrf(&slu_options, &auxAmat, dropTol, relax, panelSize,
          etree, NULL, lwork, sluPtr->permC_, sluPtr->permR_,
          &(sluPtr->SLU_Lmat), &(sluPtr->SLU_Umat), &slu_stat, &info);
   Destroy_CompCol_Permuted(&auxAmat);
   Destroy_CompCol_Matrix(&sluAmat);
   nalu_hypre_TFree(etree, NALU_HYPRE_MEMORY_HOST);
   sluPtr->factorized_ = 1;
   StatFree(&slu_stat);
   return 0;
#else
   printf("NALU_HYPRE_ParCSR_SuperLUSetup ERROR - SuperLU not enabled.\n");
   *solver = (NALU_HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * NALU_HYPRE_ParCSR_SuperLUSolve - Solve function for SuperLU.
 *--------------------------------------------------------------------------*/

int NALU_HYPRE_ParCSR_SuperLUSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A,
                              NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x )
{
#ifdef HAVE_SUPERLU
   int    nrows, i, info;
   double *bData, *xData;
   SuperMatrix B;
   SuperLUStat_t slu_stat;
   trans_t       trans;
   NALU_HYPRE_SuperLU *sluPtr = (NALU_HYPRE_SuperLU *) solver;

   /* ---------------------------------------------------------------- */
   /* make sure setup has been called                                  */
   /* ---------------------------------------------------------------- */

   nalu_hypre_assert ( sluPtr != NULL );
   if ( ! (sluPtr->factorized_) )
   {
      printf("NALU_HYPRE_ParCSR_SuperLUSolve ERROR - not factorized yet.\n");
      return -1;
   }

   /* ---------------------------------------------------------------- */
   /* fetch right hand side and solution vector                        */
   /* ---------------------------------------------------------------- */

   xData = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector *)x));
   bData = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector((nalu_hypre_ParVector *)b));
   nrows = nalu_hypre_ParVectorGlobalSize((nalu_hypre_ParVector *)x);
   for (i = 0; i < nrows; i++) xData[i] = bData[i];

   /* ---------------------------------------------------------------- */
   /* solve                                                            */
   /* ---------------------------------------------------------------- */

   dCreate_Dense_Matrix(&B, nrows, 1, bData, nrows, SLU_DN, SLU_D,SLU_GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   trans = NOTRANS;
   StatInit(&slu_stat);
   dgstrs (trans, &(sluPtr->SLU_Lmat), &(sluPtr->SLU_Umat),
           sluPtr->permC_, sluPtr->permR_, &B, &slu_stat, &info);
   Destroy_SuperMatrix_Store(&B);
   StatFree(&slu_stat);
   return 0;
#else
   printf("NALU_HYPRE_ParCSR_SuperLUSolve ERROR - SuperLU not enabled.\n");
   *solver = (NALU_HYPRE_Solver) NULL;
   return -1;
#endif
}

