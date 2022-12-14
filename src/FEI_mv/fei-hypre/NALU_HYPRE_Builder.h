/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

// *************************************************************************
// Link to build an FEI_Implementation based on HYPRE
// *************************************************************************

#ifndef _NALU_HYPRE_Builder_h_
#define _NALU_HYPRE_Builder_h_

#include "utilities/_hypre_utilities.h"

#include "NALU_HYPRE.h"
#include "../../IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "../../parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_LinSysCore.h"

#include "FEI_Implementation.h"

class NALU_HYPRE_Builder {
 public:
   static FEI* FEIBuilder(MPI_Comm comm, int masterProc) {
      NALU_HYPRE_LinSysCore* linSysCore = new NALU_HYPRE_LinSysCore(comm);

      return(new FEI_Implementation(linSysCore, comm, masterProc));
   }
};

#endif

