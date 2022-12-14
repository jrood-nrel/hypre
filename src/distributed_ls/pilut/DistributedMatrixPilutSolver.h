/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER
#define _DISTRIBUTED_MATRIX_PILUT_SOLVER_HEADER

#include "NALU_HYPRE_config.h"
#include "_hypre_utilities.h"
/*
#ifdef NALU_HYPRE_DEBUG
#include <gmalloc.h>
#endif
*/

#include "NALU_HYPRE.h"

#include "NALU_HYPRE_DistributedMatrixPilutSolver_types.h"

#include "NALU_HYPRE_distributed_matrix_types.h"
#include "NALU_HYPRE_distributed_matrix_protos.h"

#include "macros.h" /*contains some macros that are used here */

/*--------------------------------------------------------------------------
 * Global variables for the pilut solver
 *--------------------------------------------------------------------------*/

typedef struct
{
MPI_Comm hypre_MPI_communicator;
NALU_HYPRE_Int _mype, _npes;
NALU_HYPRE_Real _secpertick;
NALU_HYPRE_Int _Mfactor;
NALU_HYPRE_Int *_jr, *_jw, _lastjr, *_lr, _lastlr;	/* Work space */
NALU_HYPRE_Real *_w;				/* Work space */
NALU_HYPRE_Int _firstrow, _lastrow;			/* Matrix distribution parameters */
timer _SerTmr, _ParTmr;
NALU_HYPRE_Int _nrows, _lnrows, _ndone, _ntogo, _nleft; /* Various values used throught out */
NALU_HYPRE_Int _maxnz;
NALU_HYPRE_Int *_map;			        /* Map used for marking rows in the set */

NALU_HYPRE_Int *_vrowdist;

NALU_HYPRE_Int logging; /* if 0, turn off all printings */

/* Buffers for point to point communication */
NALU_HYPRE_Int _pilu_recv[MAX_NPES];
NALU_HYPRE_Int _pilu_send[MAX_NPES];
NALU_HYPRE_Int _lu_recv[MAX_NPES];

#ifdef NALU_HYPRE_TIMING
  /* factorization */
NALU_HYPRE_Int CCI_timer;
NALU_HYPRE_Int SS_timer;
NALU_HYPRE_Int SFR_timer;
NALU_HYPRE_Int CR_timer;
NALU_HYPRE_Int FL_timer;
NALU_HYPRE_Int SLUD_timer;
NALU_HYPRE_Int SLUM_timer;
NALU_HYPRE_Int UL_timer;
NALU_HYPRE_Int FNR_timer;
NALU_HYPRE_Int SDSeptimer;
NALU_HYPRE_Int SDKeeptimer;
NALU_HYPRE_Int SDUSeptimer;
NALU_HYPRE_Int SDUKeeptimer;

  /* solves */
NALU_HYPRE_Int Ll_timer;
NALU_HYPRE_Int Lp_timer;
NALU_HYPRE_Int Up_timer;
NALU_HYPRE_Int Ul_timer;
#endif

} hypre_PilutSolverGlobals;

/* DEFINES for global variables */
#define pilut_comm (globals->hypre_MPI_communicator)
#define mype (globals->_mype)
#define npes (globals->_npes)
#define secpertick (globals->_secpertick)
#define Mfactor (globals->_Mfactor)
#define jr (globals->_jr)
#define jw (globals->_jw)
#define lastjr (globals->_lastjr)
#define hypre_lr (globals->_lr)
#define lastlr (globals->_lastlr)
#define w (globals->_w)
#define firstrow (globals->_firstrow)
#define lastrow (globals->_lastrow)
#define SerTmr (globals->_SerTmr)
#define ParTmr (globals->_ParTmr)
#define nrows (globals->_nrows)
#define lnrows (globals->_lnrows)
#define ndone (globals->_ndone)
#define ntogo (globals->_ntogo)
#define nleft (globals->_nleft)
#define global_maxnz (globals->_maxnz)
#define pilut_map (globals->_map)
#define vrowdist (globals->_vrowdist)
#define pilu_recv (globals->_pilu_recv)
#define pilu_send (globals->_pilu_send)
#define lu_recv (globals->_lu_recv)


#include "./const.h"

/*--------------------------------------------------------------------------
 * pilut structures
 *--------------------------------------------------------------------------*/

#include "./struct.h"

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPilutSolver
 *--------------------------------------------------------------------------*/

typedef struct
{

  /* Input parameters */
  MPI_Comm               comm;
  NALU_HYPRE_DistributedMatrix  Matrix;
  NALU_HYPRE_Int                    gmaxnz;
  NALU_HYPRE_Real             tol;
  NALU_HYPRE_Int                    max_its;

  /* Structure that is used internally and built from matrix */
  DataDistType          *DataDist;

  /* Data that is passed from the factor to the solve */
  FactorMatType         *FactorMat;

  hypre_PilutSolverGlobals *globals;

} hypre_DistributedMatrixPilutSolver;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_DistributedMatrixPilutSolver structure
 *--------------------------------------------------------------------------*/

#define hypre_DistributedMatrixPilutSolverComm(solver)            ((solver) -> comm)
#define hypre_DistributedMatrixPilutSolverDataDist(solver)        ((solver) -> DataDist)
#define hypre_DistributedMatrixPilutSolverMatrix(solver)          ((solver) -> Matrix)
#define hypre_DistributedMatrixPilutSolverGmaxnz(solver)          ((solver) -> gmaxnz)
#define hypre_DistributedMatrixPilutSolverTol(solver)             ((solver) -> tol)
#define hypre_DistributedMatrixPilutSolverMaxIts(solver)          ((solver) -> max_its)
#define hypre_DistributedMatrixPilutSolverFactorMat(solver)       ((solver) -> FactorMat)
#define hypre_DistributedMatrixPilutSolverGlobals(solver)         ((solver) -> globals)

/* Include internal prototypes */
#include "./internal_protos.h"

#endif
