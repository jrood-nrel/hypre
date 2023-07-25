/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef FORTRAN_STYLE_MATRIX
#define FORTRAN_STYLE_MATRIX

#include "NALU_HYPRE_utilities.h"

typedef struct
{
   NALU_HYPRE_BigInt globalHeight;
   NALU_HYPRE_BigInt height;
   NALU_HYPRE_BigInt width;
   NALU_HYPRE_Real* value;
   NALU_HYPRE_Int    ownsValues;
} utilities_FortranMatrix;

#ifdef __cplusplus
extern "C" {
#endif

utilities_FortranMatrix*
utilities_FortranMatrixCreate(void);
void
utilities_FortranMatrixAllocateData( NALU_HYPRE_BigInt h, NALU_HYPRE_BigInt w,
                                     utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixWrap( NALU_HYPRE_Real*, NALU_HYPRE_BigInt gh, NALU_HYPRE_BigInt h, NALU_HYPRE_BigInt w,
                             utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixDestroy( utilities_FortranMatrix* mtx );

NALU_HYPRE_BigInt
utilities_FortranMatrixGlobalHeight( utilities_FortranMatrix* mtx );
NALU_HYPRE_BigInt
utilities_FortranMatrixHeight( utilities_FortranMatrix* mtx );
NALU_HYPRE_BigInt
utilities_FortranMatrixWidth( utilities_FortranMatrix* mtx );
NALU_HYPRE_Real*
utilities_FortranMatrixValues( utilities_FortranMatrix* mtx );

void
utilities_FortranMatrixClear( utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixClearL( utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixSetToIdentity( utilities_FortranMatrix* mtx );

void
utilities_FortranMatrixTransposeSquare( utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixSymmetrize( utilities_FortranMatrix* mtx );

void
utilities_FortranMatrixCopy( utilities_FortranMatrix* src, NALU_HYPRE_Int t,
                             utilities_FortranMatrix* dest );
void
utilities_FortranMatrixIndexCopy( NALU_HYPRE_Int* index,
                                  utilities_FortranMatrix* src, NALU_HYPRE_Int t,
                                  utilities_FortranMatrix* dest );

void
utilities_FortranMatrixSetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* d );
void
utilities_FortranMatrixGetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* d );
void
utilities_FortranMatrixAdd( NALU_HYPRE_Real a,
                            utilities_FortranMatrix* mtxA,
                            utilities_FortranMatrix* mtxB,
                            utilities_FortranMatrix* mtxC );
void
utilities_FortranMatrixDMultiply( utilities_FortranMatrix* d,
                                  utilities_FortranMatrix* mtx );
void
utilities_FortranMatrixMultiplyD( utilities_FortranMatrix* mtx,
                                  utilities_FortranMatrix* d );
void
utilities_FortranMatrixMultiply( utilities_FortranMatrix* mtxA, NALU_HYPRE_Int tA,
                                 utilities_FortranMatrix* mtxB, NALU_HYPRE_Int tB,
                                 utilities_FortranMatrix* mtxC );
NALU_HYPRE_Real
utilities_FortranMatrixFNorm( utilities_FortranMatrix* mtx );

NALU_HYPRE_Real
utilities_FortranMatrixValue( utilities_FortranMatrix* mtx,
                              NALU_HYPRE_BigInt i, NALU_HYPRE_BigInt j );
NALU_HYPRE_Real*
utilities_FortranMatrixValuePtr( utilities_FortranMatrix* mtx,
                                 NALU_HYPRE_BigInt i, NALU_HYPRE_BigInt j );
NALU_HYPRE_Real
utilities_FortranMatrixMaxValue( utilities_FortranMatrix* mtx );

void
utilities_FortranMatrixSelectBlock( utilities_FortranMatrix* mtx,
                                    NALU_HYPRE_BigInt iFrom, NALU_HYPRE_BigInt iTo,
                                    NALU_HYPRE_BigInt jFrom, NALU_HYPRE_BigInt jTo,
                                    utilities_FortranMatrix* block );
void
utilities_FortranMatrixUpperInv( utilities_FortranMatrix* u );

NALU_HYPRE_Int
utilities_FortranMatrixPrint( utilities_FortranMatrix* mtx, const char *fileName);

#ifdef __cplusplus
}
#endif

#endif /* FORTRAN_STYLE_MATRIX */

