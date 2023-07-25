/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ParFSAI_DATA_HEADER
#define nalu_hypre_ParFSAI_DATA_HEADER

//#define DEBUG_FSAI

/*--------------------------------------------------------------------------
 * nalu_hypre_ParFSAIData
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_ParFSAIData_struct
{
   /* FSAI Setup input data */
   NALU_HYPRE_Int             algo_type;        /* FSAI algorithm type */
   NALU_HYPRE_Int             local_solve_type; /* Local linear solver type */
   NALU_HYPRE_Int             max_steps;        /* Max. iterations run per row */
   NALU_HYPRE_Int             max_step_size;    /* Max. number of nonzeros added to a row of G per step */
   NALU_HYPRE_Int             max_nnz_row;      /* Max. number of nonzeros per row of G */
   NALU_HYPRE_Int             num_levels;       /* Number of levels for computing the candidate pattern */
   NALU_HYPRE_Real            threshold;        /* Filtering threshold for the candidate pattern */
   NALU_HYPRE_Real            kap_tolerance;    /* Min. amount of change between two steps */

   /* FSAI Setup data */
   NALU_HYPRE_Real            density;          /* Density of matrix G wrt. A */
   nalu_hypre_ParCSRMatrix   *Gmat;             /* Matrix holding FSAI factor. M^(-1) = G'G */
   nalu_hypre_ParCSRMatrix   *GTmat;            /* Matrix holding the transpose of the FSAI factor */

   /* Solver data */
   NALU_HYPRE_Int             zero_guess;       /* Flag indicating x0 = 0 */
   NALU_HYPRE_Int             eig_max_iters;    /* Iters for computing max. eigenvalue of G^T*G*A */
   NALU_HYPRE_Int             max_iterations;   /* Maximum iterations run for the solver */
   NALU_HYPRE_Int             num_iterations;   /* Number of iterations the solver ran */
   NALU_HYPRE_Real            omega;            /* Step size for Preconditioned Richardson Solver */
   NALU_HYPRE_Real            tolerance;        /* Tolerance for the solver */
   NALU_HYPRE_Real            rel_resnorm;      /* available if logging > 1 */

   /* Work data */
   nalu_hypre_ParVector      *r_work;           /* work vector used to compute the residual */
   nalu_hypre_ParVector      *z_work;           /* work vector used for applying FSAI */

   /* Log info data */
   NALU_HYPRE_Int             logging;
   NALU_HYPRE_Int             print_level;
} nalu_hypre_ParFSAIData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the nalu_hypre_ParFSAIData structure
 *--------------------------------------------------------------------------*/

/* FSAI Setup input data */
#define nalu_hypre_ParFSAIDataAlgoType(fsai_data)                ((fsai_data) -> algo_type)
#define nalu_hypre_ParFSAIDataLocalSolveType(fsai_data)          ((fsai_data) -> local_solve_type)
#define nalu_hypre_ParFSAIDataMaxSteps(fsai_data)                ((fsai_data) -> max_steps)
#define nalu_hypre_ParFSAIDataMaxStepSize(fsai_data)             ((fsai_data) -> max_step_size)
#define nalu_hypre_ParFSAIDataMaxNnzRow(fsai_data)               ((fsai_data) -> max_nnz_row)
#define nalu_hypre_ParFSAIDataNumLevels(fsai_data)               ((fsai_data) -> num_levels)
#define nalu_hypre_ParFSAIDataThreshold(fsai_data)               ((fsai_data) -> threshold)
#define nalu_hypre_ParFSAIDataKapTolerance(fsai_data)            ((fsai_data) -> kap_tolerance)

/* FSAI Setup data */
#define nalu_hypre_ParFSAIDataGmat(fsai_data)                    ((fsai_data) -> Gmat)
#define nalu_hypre_ParFSAIDataGTmat(fsai_data)                   ((fsai_data) -> GTmat)
#define nalu_hypre_ParFSAIDataDensity(fsai_data)                 ((fsai_data) -> density)

/* Solver problem data */
#define nalu_hypre_ParFSAIDataZeroGuess(fsai_data)               ((fsai_data) -> zero_guess)
#define nalu_hypre_ParFSAIDataEigMaxIters(fsai_data)             ((fsai_data) -> eig_max_iters)
#define nalu_hypre_ParFSAIDataMaxIterations(fsai_data)           ((fsai_data) -> max_iterations)
#define nalu_hypre_ParFSAIDataNumIterations(fsai_data)           ((fsai_data) -> num_iterations)
#define nalu_hypre_ParFSAIDataOmega(fsai_data)                   ((fsai_data) -> omega)
#define nalu_hypre_ParFSAIDataRelResNorm(fsai_data)              ((fsai_data) -> rel_resnorm)
#define nalu_hypre_ParFSAIDataTolerance(fsai_data)               ((fsai_data) -> tolerance)

/* Work data */
#define nalu_hypre_ParFSAIDataRWork(fsai_data)                   ((fsai_data) -> r_work)
#define nalu_hypre_ParFSAIDataZWork(fsai_data)                   ((fsai_data) -> z_work)

/* Log info data */
#define nalu_hypre_ParFSAIDataLogging(fsai_data)                 ((fsai_data) -> logging)
#define nalu_hypre_ParFSAIDataPrintLevel(fsai_data)              ((fsai_data) -> print_level)

#endif
