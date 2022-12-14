/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Header file for test drivers
 *--------------------------------------------------------------------------*/
#ifndef NALU_HYPRE_TEST_INCLUDES
#define NALU_HYPRE_TEST_INCLUDES

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "NALU_HYPRE_krylov.h"
#include "_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "NALU_HYPRE_struct_ls.h"
#include "NALU_HYPRE_sstruct_ls.h"

#define NALU_HYPRE_BICGSTAB   99100
#define NALU_HYPRE_BOOMERAMG  99110
#define NALU_HYPRE_CGNR       99120
#define NALU_HYPRE_DIAGSCALE  99130
#define NALU_HYPRE_EUCLID     99140
#define NALU_HYPRE_GMRES      99150
#define NALU_HYPRE_GSMG       99155
#define NALU_HYPRE_HYBRID     99160
#define NALU_HYPRE_JACOBI     99170
#define NALU_HYPRE_PARASAILS  99180
#define NALU_HYPRE_PCG        99190
#define NALU_HYPRE_PFMG_1     99200
#define NALU_HYPRE_PILUT      99210
#define NALU_HYPRE_SCHWARZ    99220
#define NALU_HYPRE_SMG_1      99230
#define NALU_HYPRE_SPARSEMSG  99240
#define NALU_HYPRE_SPLIT      99250
#define NALU_HYPRE_SPLITPFMG  99260
#define NALU_HYPRE_SPLITSMG   99270
#define NALU_HYPRE_SYSPFMG    99280

/****************************************************************************
 * Prototypes for testing routines
 ***************************************************************************/
NALU_HYPRE_Int hypre_set_precond(NALU_HYPRE_Int matrix_id, NALU_HYPRE_Int solver_id, NALU_HYPRE_Int precond_id,
                            void *solver, void *precond);

NALU_HYPRE_Int hypre_set_precond_params(NALU_HYPRE_Int precond_id, void *precond);

NALU_HYPRE_Int hypre_destroy_precond(NALU_HYPRE_Int precond_id, void *precond);

/****************************************************************************
 * Variables for testing routines
 ***************************************************************************/
NALU_HYPRE_Int      k_dim = 5;
NALU_HYPRE_Int      gsmg_samples = 5;
NALU_HYPRE_Int      poutdat = 1;
NALU_HYPRE_Int      hybrid = 1;
NALU_HYPRE_Int      coarsen_type = 6;
NALU_HYPRE_Int      measure_type = 0;
NALU_HYPRE_Int      smooth_type = 6;
NALU_HYPRE_Int      num_functions = 1;
NALU_HYPRE_Int      smooth_num_levels = 0;
NALU_HYPRE_Int      smooth_num_sweeps = 1;
NALU_HYPRE_Int      num_sweep = 1;
NALU_HYPRE_Int      max_levels = 25;
NALU_HYPRE_Int      variant = 0;
NALU_HYPRE_Int      overlap = 1;
NALU_HYPRE_Int      domain_type = 2;
NALU_HYPRE_Int      nonzeros_to_keep = -1;

NALU_HYPRE_Int      interp_type;
NALU_HYPRE_Int      cycle_type;
NALU_HYPRE_Int      relax_default;
NALU_HYPRE_Int     *dof_func;
NALU_HYPRE_Int     *num_grid_sweeps;
NALU_HYPRE_Int     *grid_relax_type;
NALU_HYPRE_Int    **grid_relax_points;

NALU_HYPRE_Real   tol = 1.e-8;
NALU_HYPRE_Real   pc_tol = 0.;
NALU_HYPRE_Real   drop_tol = -1.;
NALU_HYPRE_Real   max_row_sum = 1.;
NALU_HYPRE_Real   schwarz_rlx_weight = 1.;
NALU_HYPRE_Real   sai_threshold = 0.1;
NALU_HYPRE_Real   sai_filter = 0.1;

NALU_HYPRE_Real   strong_threshold;
NALU_HYPRE_Real   trunc_factor;
NALU_HYPRE_Real  *relax_weight;
NALU_HYPRE_Real  *omega;

#endif
