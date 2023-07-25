/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// **************************************************************************
// This is the class that handles slide surface reduction
// **************************************************************************

#ifndef __NALU_HYPRE_UZAWA__
#define __NALU_HYPRE_UZAWA__

// **************************************************************************
// system libraries used
// --------------------------------------------------------------------------

#include "utilities/_nalu_hypre_utilities.h"
#include "IJ_mv/_nalu_hypre_IJ_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"


// *************************************************************************
// Solver-Preconditioner parameter data structure
// -------------------------------------------------------------------------

typedef struct NALU_HYPRE_Uzawa_PARAMS_Struct
{
   int            SolverID_;      // solver ID
   int            PrecondID_;     // preconditioner ID
   double         Tol_;           // tolerance for Krylov solver
   int            MaxIter_;       // max iterations for Krylov solver
   int            PSNLevels_;     // Nlevels for ParaSails
   double         PSThresh_;      // threshold for ParaSails
   double         PSFilter_;      // filter for ParaSails
   double         AMGThresh_;     // threshold for BoomerAMG
   int            AMGNSweeps_;    // no. of relaxations for BoomerAMG
   int            AMGSystemSize_; // system size for BoomerAMG
   int            PilutFillin_;   // Fillin for Pilut
   double         PilutDropTol_;  // drop tolerance for Pilut
   int            EuclidNLevels_; // nlevels for Euclid
   double         EuclidThresh_;  // threshold for Euclid
   double         MLIThresh_;     // threshold for MLI SA
   double         MLIPweight_;    // Pweight for MLI SA
   int            MLINSweeps_;    // no. of relaxations for MLI
   int            MLINodeDOF_;    // system size for BoomerAMG
   int            MLINullDim_;    // null space dimension for MLI SA
}
NALU_HYPRE_Uzawa_PARAMS;

// **************************************************************************
// class definition
// --------------------------------------------------------------------------

class NALU_HYPRE_LSI_Uzawa
{
   MPI_Comm           mpiComm_;
   int                outputLevel_;
   int                modifiedScheme_;
   int                S22Scheme_;
   int                maxIterations_;
   double             tolerance_;
   double             S22SolverDampFactor_;
   int                numIterations_;
   NALU_HYPRE_ParCSRMatrix Amat_;
   NALU_HYPRE_ParCSRMatrix A11mat_;
   NALU_HYPRE_ParCSRMatrix A12mat_;
   NALU_HYPRE_ParCSRMatrix S22mat_;
   int                *procA22Sizes_;
   NALU_HYPRE_Solver       A11Solver_;        // solver for A11 matrix
   NALU_HYPRE_Solver       A11Precond_;       // preconditioner for A11 matrix
   NALU_HYPRE_Solver       S22Solver_;        // solver for S12 
   NALU_HYPRE_Solver       S22Precond_;       // preconditioner for S12 
   NALU_HYPRE_Uzawa_PARAMS A11Params_;
   NALU_HYPRE_Uzawa_PARAMS S22Params_;

 public:

   NALU_HYPRE_LSI_Uzawa(MPI_Comm);
   virtual ~NALU_HYPRE_LSI_Uzawa();
   int    setOutputLevel(int level) {outputLevel_ = level; return 0;}
   int    setParams(char *paramString);
   int    setMaxIterations(int);
   int    setTolerance(double);
   int    getNumIterations(int&);
   int    setup(NALU_HYPRE_ParCSRMatrix , NALU_HYPRE_ParVector , NALU_HYPRE_ParVector );
   int    solve(NALU_HYPRE_ParVector , NALU_HYPRE_ParVector );

 private:
   int    findA22BlockSize();
   int    buildBlockMatrices();
   int    buildA11A12Mat();
   int    buildS22Mat();
   int    setupSolver(NALU_HYPRE_Solver *,NALU_HYPRE_ParCSRMatrix, NALU_HYPRE_ParVector, 
                      NALU_HYPRE_ParVector, NALU_HYPRE_Solver, NALU_HYPRE_Uzawa_PARAMS);
   int    setupPrecon(NALU_HYPRE_Solver *,NALU_HYPRE_ParCSRMatrix,NALU_HYPRE_Uzawa_PARAMS);
};

#endif

