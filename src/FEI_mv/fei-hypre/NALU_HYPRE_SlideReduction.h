/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// **************************************************************************
// This is the class that handles slide surface reduction
// **************************************************************************

#ifndef __NALU_HYPRE_SLIDEREDUCTION__
#define __NALU_HYPRE_SLIDEREDUCTION__

// **************************************************************************
// system libraries used
// --------------------------------------------------------------------------

#include "utilities/_hypre_utilities.h"
#include "IJ_mv/_hypre_IJ_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

// **************************************************************************
// class definition
// --------------------------------------------------------------------------

class NALU_HYPRE_SlideReduction
{
   MPI_Comm       mpiComm_;
   NALU_HYPRE_IJMatrix Amat_;
   NALU_HYPRE_IJMatrix A21mat_;
   NALU_HYPRE_IJMatrix invA22mat_;
   NALU_HYPRE_IJMatrix reducedAmat_;
   NALU_HYPRE_IJVector reducedBvec_;
   NALU_HYPRE_IJVector reducedXvec_;
   NALU_HYPRE_IJVector reducedRvec_;
   int            outputLevel_;
   int            *procNConstr_;
   int            *slaveEqnList_;
   int            *slaveEqnListAux_;
   int            *gSlaveEqnList_;
   int            *gSlaveEqnListAux_;
   int            *constrBlkInfo_;
   int            *constrBlkSizes_;
   int            *eqnStatuses_;
   double         blockMinNorm_;
   NALU_HYPRE_ParCSRMatrix hypreRAP_;
   double         truncTol_;
   double         *ADiagISqrts_;
   int            scaleMatrixFlag_;
   int            useSimpleScheme_;

 public:

   NALU_HYPRE_SlideReduction(MPI_Comm);
   virtual ~NALU_HYPRE_SlideReduction();
   int    setOutputLevel(int level);
   int    setUseSimpleScheme();
   int    setTruncationThreshold(double trunc);
   int    setScaleMatrix();
   int    setBlockMinNorm(double norm);

   int    getMatrixNumRows(); 
   double *getMatrixDiagonal();
   int    getReducedMatrix(NALU_HYPRE_IJMatrix *mat); 
   int    getReducedRHSVector(NALU_HYPRE_IJVector *rhs);
   int    getReducedSolnVector(NALU_HYPRE_IJVector *sol);
   int    getReducedAuxVector(NALU_HYPRE_IJVector *auxV);
   int    getProcConstraintMap(int **map);
   int    getSlaveEqnList(int **slist);
   int    getPerturbationMatrix(NALU_HYPRE_ParCSRMatrix *matrix);
   int    setup(NALU_HYPRE_IJMatrix , NALU_HYPRE_IJVector , NALU_HYPRE_IJVector );
   int    buildReducedSolnVector(NALU_HYPRE_IJVector x, NALU_HYPRE_IJVector b);
   int    buildModifiedSolnVector(NALU_HYPRE_IJVector x);

 private:

   int    findConstraints();
   int    findSlaveEqns1();
   int    findSlaveEqnsBlock(int blkSize);
   int    composeGlobalList();
   int    buildSubMatrices();
   int    buildModifiedRHSVector(NALU_HYPRE_IJVector, NALU_HYPRE_IJVector);
   int    buildReducedMatrix();
   int    buildReducedRHSVector(NALU_HYPRE_IJVector);
   int    buildA21Mat();
   int    buildInvA22Mat();
   int    scaleMatrixVector();
   double matrixCondEst(int, int, int *, int);

   int    findSlaveEqns2(int **couplings);
   int    buildReducedMatrix2();
};

#endif

