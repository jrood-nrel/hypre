/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *************************************************************************
// This is the NALU_HYPRE implementation of Schur reduction
// *************************************************************************

#ifndef __NALU_HYPRE_LSI_SCHURH__
#define __NALU_HYPRE_LSI_SCHURH__

// *************************************************************************
// system libraries used
// -------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"

// *************************************************************************
// local defines
// -------------------------------------------------------------------------

#include "NALU_HYPRE_FEI_includes.h"

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class NALU_HYPRE_LSI_Schur
{
   NALU_HYPRE_IJMatrix A11mat_;           // mass matrix (should be diagonal)
   NALU_HYPRE_IJMatrix A12mat_;           // gradient (divergence) matrix
   NALU_HYPRE_IJMatrix A22mat_;           // stabilization matrix
   NALU_HYPRE_IJVector F1vec_;            // rhs for block(1,1)
   NALU_HYPRE_IJMatrix Smat_;             // Schur complement matrix
   NALU_HYPRE_IJVector Svec_;             // reduced RHS
   int            *APartition_;      // processor partition of matrix A
   int            P22Size_;          // number of pressure variables
   int            P22GSize_;         // global number of pressure variables
   int            *P22LocalInds_;    // pressure local row indices (global)
   int            *P22GlobalInds_;   // pressure off-processor row indices
   int            *P22Offsets_;      // processor partiton of matrix A22
   int            assembled_;        // set up complete flag
   int            outputLevel_;      // for diagnostics
   Lookup         *lookup_;          // FEI lookup object
   MPI_Comm       mpiComm_;

 public:

   NALU_HYPRE_LSI_Schur();
   virtual ~NALU_HYPRE_LSI_Schur();
   int     setLookup( Lookup *lookup );
   int     setup(NALU_HYPRE_IJMatrix Amat,
                 NALU_HYPRE_IJVector sol,   NALU_HYPRE_IJVector rhs,
                 NALU_HYPRE_IJMatrix *redA, NALU_HYPRE_IJVector *rsol,
                 NALU_HYPRE_IJVector *rrhs,  NALU_HYPRE_IJVector *rres);
   int     computeRHS(NALU_HYPRE_IJVector rhs,  NALU_HYPRE_IJVector *rrhs);
   int     computeSol(NALU_HYPRE_IJVector rsol, NALU_HYPRE_IJVector sol);
   int     print();

 private:
   int     computeBlockInfo();
   int     buildBlocks(NALU_HYPRE_IJMatrix Amat);
};

#endif

