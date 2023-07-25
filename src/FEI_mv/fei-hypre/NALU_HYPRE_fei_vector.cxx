/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_fei_vector functions
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef HAVE_FEI
#include "FEI_Implementation.h"
#endif
#include "LLNL_FEI_Impl.h"
#include "fei_mv.h"
//New FEI 2.23.02
#include "fei_Data.hpp"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"

/*****************************************************************************/
/* NALU_HYPRE_FEVectorCreate function                                             */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEVectorCreate(MPI_Comm comm, NALU_HYPRE_FEMesh mesh, NALU_HYPRE_FEVector *vector)
{
   NALU_HYPRE_FEVector myVector;
   myVector = (NALU_HYPRE_FEVector) nalu_hypre_TAlloc(NALU_HYPRE_FEVector, 1, NALU_HYPRE_MEMORY_HOST);
   myVector->mesh_ = mesh;
   myVector->comm_ = comm;
   (*vector) = myVector;
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEVectorDestroy - Destroy a FEVector object.                        */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEVectorDestroy(NALU_HYPRE_FEVector vector)
{
   if (vector)
   {
      free(vector);
   }
   return 0;
}

/*****************************************************************************/
/* NALU_HYPRE_FEVectorGetRHS                                                      */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEVectorGetRHS(NALU_HYPRE_FEVector vector, void **object)
{
   int               ierr=0;
   NALU_HYPRE_FEMesh      mesh;
   LinearSystemCore* lsc;
   Data              dataObj;
   NALU_HYPRE_IJVector    X;
   NALU_HYPRE_ParVector   XCSR;

   if (vector == NULL)
      ierr = 1;
   else
   {
      mesh = vector->mesh_;
      if (mesh == NULL)
         ierr = 1;
      else
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         if (lsc != NULL)
         {
            lsc->copyOutRHSVector(1.0e0, dataObj); 
            X = (NALU_HYPRE_IJVector) dataObj.getDataPtr();
            NALU_HYPRE_IJVectorGetObject(X, (void **) &XCSR);
            (*object) = (void *) XCSR;
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

/*****************************************************************************/
/* NALU_HYPRE_FEVectorSetSol                                                      */
/*---------------------------------------------------------------------------*/

extern "C" int
NALU_HYPRE_FEVectorSetSol(NALU_HYPRE_FEVector vector, void *object)
{
   int                ierr=0;
   NALU_HYPRE_FEMesh       mesh;
   LinearSystemCore   *lsc;
   Data               dataObj;
   LLNL_FEI_Impl      *fei1;
#ifdef HAVE_FEI
   FEI_Implementation *fei2;
#endif

   if (vector == NULL)
      ierr = 1;
   else
   {
      mesh = vector->mesh_;
      if (mesh == NULL)
         ierr = 1;
      else
      {
         lsc = (LinearSystemCore *) mesh->linSys_;
         if (lsc != NULL)
         {
            dataObj.setTypeName("Sol_Vector");
            dataObj.setDataPtr((void*) object);
            lsc->copyInRHSVector(1.0e0, dataObj); 
            if (mesh->feiPtr_ != NULL)
            {
#ifdef HAVE_FEI
               if (mesh->objectType_ == 1)
               {
                  fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
                  ierr = fei1->solve(&ierr);
               }
               if (mesh->objectType_ == 2)
               {
                  fei2 = (FEI_Implementation *) mesh->feiPtr_;
                  ierr = fei2->solve(&ierr);
               }
#else
               fei1 = (LLNL_FEI_Impl *) mesh->feiPtr_;
               ierr = fei1->solve(&ierr);
#endif
            }
         }
         else
         {
            ierr = 1;
         }
      }
   }
   return ierr;
}

