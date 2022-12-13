/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for NALU_HYPRE_mv library
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_SEQ_MV_HEADER
#define NALU_HYPRE_SEQ_MV_HEADER

#include "NALU_HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct hypre_CSRMatrix_struct;
typedef struct hypre_CSRMatrix_struct *NALU_HYPRE_CSRMatrix;
struct hypre_MappedMatrix_struct;
typedef struct hypre_MappedMatrix_struct *NALU_HYPRE_MappedMatrix;
struct hypre_MultiblockMatrix_struct;
typedef struct hypre_MultiblockMatrix_struct *NALU_HYPRE_MultiblockMatrix;
#ifndef NALU_HYPRE_VECTOR_STRUCT
#define NALU_HYPRE_VECTOR_STRUCT
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *NALU_HYPRE_Vector;
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* NALU_HYPRE_csr_matrix.c */
NALU_HYPRE_CSRMatrix NALU_HYPRE_CSRMatrixCreate( NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                       NALU_HYPRE_Int *row_sizes );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixDestroy( NALU_HYPRE_CSRMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixInitialize( NALU_HYPRE_CSRMatrix matrix );
NALU_HYPRE_CSRMatrix NALU_HYPRE_CSRMatrixRead( char *file_name );
void NALU_HYPRE_CSRMatrixPrint( NALU_HYPRE_CSRMatrix matrix, char *file_name );
NALU_HYPRE_Int NALU_HYPRE_CSRMatrixGetNumRows( NALU_HYPRE_CSRMatrix matrix, NALU_HYPRE_Int *num_rows );

/* NALU_HYPRE_mapped_matrix.c */
NALU_HYPRE_MappedMatrix NALU_HYPRE_MappedMatrixCreate( void );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixDestroy( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixLimitedDestroy( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixInitialize( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixAssemble( NALU_HYPRE_MappedMatrix matrix );
void NALU_HYPRE_MappedMatrixPrint( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixGetColIndex( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int j );
void *NALU_HYPRE_MappedMatrixGetMatrix( NALU_HYPRE_MappedMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetMatrix( NALU_HYPRE_MappedMatrix matrix, void *matrix_data );
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetColMap( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int (*ColMap )(NALU_HYPRE_Int,
                                                                                       void *));
NALU_HYPRE_Int NALU_HYPRE_MappedMatrixSetMapData( NALU_HYPRE_MappedMatrix matrix, void *MapData );

/* NALU_HYPRE_multiblock_matrix.c */
NALU_HYPRE_MultiblockMatrix NALU_HYPRE_MultiblockMatrixCreate( void );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixDestroy( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixLimitedDestroy( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixInitialize( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixAssemble( NALU_HYPRE_MultiblockMatrix matrix );
void NALU_HYPRE_MultiblockMatrixPrint( NALU_HYPRE_MultiblockMatrix matrix );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixSetNumSubmatrices( NALU_HYPRE_MultiblockMatrix matrix, NALU_HYPRE_Int n );
NALU_HYPRE_Int NALU_HYPRE_MultiblockMatrixSetSubmatrixType( NALU_HYPRE_MultiblockMatrix matrix, NALU_HYPRE_Int j,
                                                  NALU_HYPRE_Int type );

/* NALU_HYPRE_vector.c */
NALU_HYPRE_Vector NALU_HYPRE_VectorCreate( NALU_HYPRE_Int size );
NALU_HYPRE_Int NALU_HYPRE_VectorDestroy( NALU_HYPRE_Vector vector );
NALU_HYPRE_Int NALU_HYPRE_VectorInitialize( NALU_HYPRE_Vector vector );
NALU_HYPRE_Int NALU_HYPRE_VectorPrint( NALU_HYPRE_Vector vector, char *file_name );
NALU_HYPRE_Vector NALU_HYPRE_VectorRead( char *file_name );

typedef enum NALU_HYPRE_TimerID
{
   // timers for solver phase
   NALU_HYPRE_TIMER_ID_MATVEC = 0,
   NALU_HYPRE_TIMER_ID_BLAS1,
   NALU_HYPRE_TIMER_ID_RELAX,
   NALU_HYPRE_TIMER_ID_GS_ELIM_SOLVE,

   // timers for solve MPI
   NALU_HYPRE_TIMER_ID_PACK_UNPACK, // copying data to/from send/recv buf
   NALU_HYPRE_TIMER_ID_HALO_EXCHANGE, // halo exchange in matvec and relax
   NALU_HYPRE_TIMER_ID_ALL_REDUCE,

   // timers for setup phase
   // coarsening
   NALU_HYPRE_TIMER_ID_CREATES,
   NALU_HYPRE_TIMER_ID_CREATE_2NDS,
   NALU_HYPRE_TIMER_ID_PMIS,

   // interpolation
   NALU_HYPRE_TIMER_ID_EXTENDED_I_INTERP,
   NALU_HYPRE_TIMER_ID_PARTIAL_INTERP,
   NALU_HYPRE_TIMER_ID_MULTIPASS_INTERP,
   NALU_HYPRE_TIMER_ID_INTERP_TRUNC,
   NALU_HYPRE_TIMER_ID_MATMUL, // matrix-matrix multiplication
   NALU_HYPRE_TIMER_ID_COARSE_PARAMS,

   // rap
   NALU_HYPRE_TIMER_ID_RAP,

   // timers for setup MPI
   NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX,
   NALU_HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA,

   // setup etc
   NALU_HYPRE_TIMER_ID_GS_ELIM_SETUP,

   // temporaries
   NALU_HYPRE_TIMER_ID_BEXT_A,
   NALU_HYPRE_TIMER_ID_BEXT_S,
   NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP,
   NALU_HYPRE_TIMER_ID_MERGE,

   // csr matop
   NALU_HYPRE_TIMER_ID_SPGEMM_ROWNNZ,
   NALU_HYPRE_TIMER_ID_SPGEMM_ATTEMPT1,
   NALU_HYPRE_TIMER_ID_SPGEMM_ATTEMPT2,
   NALU_HYPRE_TIMER_ID_SPGEMM_SYMBOLIC,
   NALU_HYPRE_TIMER_ID_SPGEMM_NUMERIC,
   NALU_HYPRE_TIMER_ID_SPGEMM,
   NALU_HYPRE_TIMER_ID_SPADD,
   NALU_HYPRE_TIMER_ID_SPTRANS,

   NALU_HYPRE_TIMER_ID_COUNT
} NALU_HYPRE_TimerID;

extern NALU_HYPRE_Real hypre_profile_times[NALU_HYPRE_TIMER_ID_COUNT];

#ifdef __cplusplus
}

#endif

#endif
