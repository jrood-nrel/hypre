/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "NALU_HYPRE.h"
#include "mli_vector.h"
#include "NALU_HYPRE_IJ_mv.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "mli_utils.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Vector::MLI_Vector( void *invec,const char *inName, MLI_Function *funcPtr )
{
   strncpy(name_, inName, 100);
   vector_ = invec;
   if ( funcPtr != NULL ) destroyFunc_ = (int (*)(void*)) funcPtr->func_;
   else                   destroyFunc_ = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Vector::~MLI_Vector()
{
   if (vector_ != NULL && destroyFunc_ != NULL) destroyFunc_((void*) vector_);
   vector_      = NULL;
   destroyFunc_ = NULL;
}

/******************************************************************************
 * get name of the vector
 *---------------------------------------------------------------------------*/

char *MLI_Vector::getName()
{
   return name_;
}

/******************************************************************************
 * get vector
 *---------------------------------------------------------------------------*/

void *MLI_Vector::getVector()
{
   return (void *) vector_;
}

/******************************************************************************
 * set vector to a constant
 *---------------------------------------------------------------------------*/

int MLI_Vector::setConstantValue(double value)
{
   if ( strcmp( name_, "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::setConstantValue ERROR - type not NALU_HYPRE_ParVector\n");
      exit(1);
   }
   nalu_hypre_ParVector *vec = (nalu_hypre_ParVector *) vector_;
   return (nalu_hypre_ParVectorSetConstantValues( vec, value ));
}

/******************************************************************************
 * inner product
 *---------------------------------------------------------------------------*/

int MLI_Vector::copy(MLI_Vector *vec2)
{
   if ( strcmp( name_, "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (from).\n");
      exit(1);
   }
   if ( strcmp( vec2->getName(), "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::copy ERROR - invalid type (to).\n");
      exit(1);
   }
   nalu_hypre_ParVector *hypreV1 = (nalu_hypre_ParVector *) vector_;
   nalu_hypre_ParVector *hypreV2 = (nalu_hypre_ParVector *) vec2->getVector();
   nalu_hypre_ParVectorCopy( hypreV1, hypreV2 );
   return 0;
}

/******************************************************************************
 * print to a file
 *---------------------------------------------------------------------------*/

int MLI_Vector::print(char *filename)
{
   if ( strcmp( name_, "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   if ( filename == NULL ) return 1;
   nalu_hypre_ParVector *vec = (nalu_hypre_ParVector *) vector_;
   nalu_hypre_ParVectorPrint( vec, filename );
   return 0;
}

/******************************************************************************
 * inner product
 *---------------------------------------------------------------------------*/

double MLI_Vector::norm2()
{
   if ( strcmp( name_, "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::innerProduct ERROR - invalid type.\n");
      exit(1);
   }
   nalu_hypre_ParVector *vec = (nalu_hypre_ParVector *) vector_;
   return (sqrt(nalu_hypre_ParVectorInnerProd( vec, vec )));
}

/******************************************************************************
 * clone a hypre vector
 *---------------------------------------------------------------------------*/

MLI_Vector *MLI_Vector::clone()
{
   char            paramString[100];
   MPI_Comm        comm;
   nalu_hypre_ParVector *newVec;
   nalu_hypre_Vector    *seqVec;
   int             i, nlocals, globalSize, *vpartition, *partitioning;
   int             mypid, nprocs;
   double          *darray;
   MLI_Function    *funcPtr;

   if ( strcmp( name_, "NALU_HYPRE_ParVector" ) )
   {
      printf("MLI_Vector::clone ERROR - invalid type.\n");
      exit(1);
   }
   nalu_hypre_ParVector *vec = (nalu_hypre_ParVector *) vector_;
   comm = nalu_hypre_ParVectorComm(vec);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&nprocs);
   vpartition = nalu_hypre_ParVectorPartitioning(vec);
   partitioning = nalu_hypre_CTAlloc(int,nprocs+1, NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < nprocs+1; i++ ) partitioning[i] = vpartition[i];
   globalSize = nalu_hypre_ParVectorGlobalSize(vec);
   newVec = nalu_hypre_CTAlloc(nalu_hypre_ParVector, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParVectorComm(newVec) = comm;
   nalu_hypre_ParVectorGlobalSize(newVec) = globalSize;
   nalu_hypre_ParVectorFirstIndex(newVec) = partitioning[mypid];
   nalu_hypre_ParVectorPartitioning(newVec) = partitioning;
   nalu_hypre_ParVectorOwnsData(newVec) = 1;
   nlocals = partitioning[mypid+1] - partitioning[mypid];
   seqVec = nalu_hypre_SeqVectorCreate(nlocals);
   nalu_hypre_SeqVectorInitialize(seqVec);
   darray = nalu_hypre_VectorData(seqVec);
   for (i = 0; i < nlocals; i++) darray[i] = 0.0;
   nalu_hypre_ParVectorLocalVector(newVec) = seqVec;
   sprintf(paramString,"NALU_HYPRE_ParVector");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParVectorGetDestroyFunc(funcPtr);
   MLI_Vector *mliVec = new MLI_Vector(newVec, paramString, funcPtr);
   delete funcPtr;
   return mliVec;
}
