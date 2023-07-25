/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_fei_matrix functions
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "fei_mv.h"
//NEW FEI 2.23.02
#include "fei_Data.hpp"

/*****************************************************************************/
/* NALU_HYPRE_FEMatrixCreate function                                             */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMatrixCreate(MPI_Comm comm, NALU_HYPRE_FEMesh mesh, NALU_HYPRE_FEMatrix *matrix)
{
   NALU_HYPRE_FEMatrix myMatrix;
   myMatrix = (NALU_HYPRE_FEMatrix) nalu_hypre_TAlloc(NALU_HYPRE_FEMatrix, 1, NALU_HYPRE_MEMORY_HOST);
   myMatrix->comm_ = comm;
   myMatrix->mesh_ = mesh;
   (*matrix) = myMatrix;
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMatrixDestroy - Destroy a FEMatrix object.                        */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMatrixDestroy(NALU_HYPRE_FEMatrix matrix)
{
   if (matrix)
   {
      free(matrix);
   }
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEMatrixGetObject                                                   */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEMatrixGetObject(NALU_HYPRE_FEMatrix matrix, void **object)
{
   int                ierr=0;
   NALU_HYPRE_FEMesh       mesh;
   LinearSystemCore*  lsc;
   Data               dataObj;
   NALU_HYPRE_IJMatrix     A;
   NALU_HYPRE_ParCSRMatrix ACSR;

   if (matrix == NULL)
      ierr = 1;
   else
   {
      mesh = matrix->mesh_;
      if (mesh == NULL)
         ierr = 1;
      else
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         if (lsc != NULL)
         {
            lsc->copyOutMatrix(1.0e0, dataObj); 
            A = (NALU_HYPRE_IJMatrix) dataObj.getDataPtr();
            NALU_HYPRE_IJMatrixGetObject(A, (void **) &ACSR);
            (*object) = (void *) ACSR;
         }
         else
         {
            (*object) = NULL;
            ierr = 1;
         }
      }
   }
   return ierr;
}

