/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for NALU_HYPRE_utilities library
 *
 *****************************************************************************/

#ifndef NALU_HYPRE_UTILITIES_HEADER
#define NALU_HYPRE_UTILITIES_HEADER

#include <NALU_HYPRE_config.h>

#ifndef NALU_HYPRE_SEQUENTIAL
#include "mpi.h"
#endif

#ifdef NALU_HYPRE_USING_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Big int stuff
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_BIGINT)
typedef long long int NALU_HYPRE_BigInt;
typedef long long int NALU_HYPRE_Int;
#define NALU_HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define NALU_HYPRE_MPI_INT MPI_LONG_LONG_INT

#elif defined(NALU_HYPRE_MIXEDINT)
typedef long long int NALU_HYPRE_BigInt;
typedef int NALU_HYPRE_Int;
#define NALU_HYPRE_MPI_BIG_INT MPI_LONG_LONG_INT
#define NALU_HYPRE_MPI_INT MPI_INT

#else /* default */
typedef int NALU_HYPRE_BigInt;
typedef int NALU_HYPRE_Int;
#define NALU_HYPRE_MPI_BIG_INT MPI_INT
#define NALU_HYPRE_MPI_INT MPI_INT
#endif

/*--------------------------------------------------------------------------
 * Real and Complex types
 *--------------------------------------------------------------------------*/

#include <float.h>

#if defined(NALU_HYPRE_SINGLE)
typedef float NALU_HYPRE_Real;
#define NALU_HYPRE_REAL_MAX FLT_MAX
#define NALU_HYPRE_REAL_MIN FLT_MIN
#define NALU_HYPRE_REAL_EPSILON FLT_EPSILON
#define NALU_HYPRE_REAL_MIN_EXP FLT_MIN_EXP
#define NALU_HYPRE_MPI_REAL MPI_FLOAT

#elif defined(NALU_HYPRE_LONG_DOUBLE)
typedef long double NALU_HYPRE_Real;
#define NALU_HYPRE_REAL_MAX LDBL_MAX
#define NALU_HYPRE_REAL_MIN LDBL_MIN
#define NALU_HYPRE_REAL_EPSILON LDBL_EPSILON
#define NALU_HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define NALU_HYPRE_MPI_REAL MPI_LONG_DOUBLE

#else /* default */
typedef double NALU_HYPRE_Real;
#define NALU_HYPRE_REAL_MAX DBL_MAX
#define NALU_HYPRE_REAL_MIN DBL_MIN
#define NALU_HYPRE_REAL_EPSILON DBL_EPSILON
#define NALU_HYPRE_REAL_MIN_EXP DBL_MIN_EXP
#define NALU_HYPRE_MPI_REAL MPI_DOUBLE
#endif

#if defined(NALU_HYPRE_COMPLEX)
typedef double _Complex NALU_HYPRE_Complex;
#define NALU_HYPRE_MPI_COMPLEX MPI_C_DOUBLE_COMPLEX  /* or MPI_LONG_DOUBLE ? */

#else  /* default */
typedef NALU_HYPRE_Real NALU_HYPRE_Complex;
#define NALU_HYPRE_MPI_COMPLEX NALU_HYPRE_MPI_REAL
#endif

/*--------------------------------------------------------------------------
 * Sequential MPI stuff
 *--------------------------------------------------------------------------*/

#ifdef NALU_HYPRE_SEQUENTIAL
typedef NALU_HYPRE_Int MPI_Comm;
#endif

/*--------------------------------------------------------------------------
 * HYPRE error codes
 *--------------------------------------------------------------------------*/

#define NALU_HYPRE_ERROR_GENERIC         1   /* generic error */
#define NALU_HYPRE_ERROR_MEMORY          2   /* unable to allocate memory */
#define NALU_HYPRE_ERROR_ARG             4   /* argument error */
/* bits 4-8 are reserved for the index of the argument error */
#define NALU_HYPRE_ERROR_CONV          256   /* method did not converge as expected */
#define NALU_HYPRE_MAX_FILE_NAME_LEN  1024   /* longest filename length used in hypre */

/*--------------------------------------------------------------------------
 * HYPRE init/finalize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_Init();
NALU_HYPRE_Int NALU_HYPRE_Finalize();

/*--------------------------------------------------------------------------
 * HYPRE error user functions
 *--------------------------------------------------------------------------*/

/* Return the current hypre error flag */
NALU_HYPRE_Int NALU_HYPRE_GetError();

/* Check if the given error flag contains the given error code */
NALU_HYPRE_Int NALU_HYPRE_CheckError(NALU_HYPRE_Int hypre_ierr, NALU_HYPRE_Int hypre_error_code);

/* Return the index of the argument (counting from 1) where
   argument error (NALU_HYPRE_ERROR_ARG) has occured */
NALU_HYPRE_Int NALU_HYPRE_GetErrorArg();

/* Describe the given error flag in the given string */
void NALU_HYPRE_DescribeError(NALU_HYPRE_Int hypre_ierr, char *descr);

/* Clears the hypre error flag */
NALU_HYPRE_Int NALU_HYPRE_ClearAllErrors();

/* Clears the given error code from the hypre error flag */
NALU_HYPRE_Int NALU_HYPRE_ClearError(NALU_HYPRE_Int hypre_error_code);

/* Print GPU information */
NALU_HYPRE_Int NALU_HYPRE_PrintDeviceInfo();

/*--------------------------------------------------------------------------
 * HYPRE Version routines
 *--------------------------------------------------------------------------*/

/* RDF: This macro is used by the FEI code.  Want to eventually remove. */
#define NALU_HYPRE_VERSION "NALU_HYPRE_RELEASE_NAME Date Compiled: " __DATE__ " " __TIME__

/**
 * Allocates and returns a string with version number information in it.
 **/
NALU_HYPRE_Int
NALU_HYPRE_Version( char **version_ptr );

/**
 * Returns version number information in integer form.  Use 'NULL' for values
 * not needed.  The argument {\tt single} is a single sortable integer
 * representation of the release number.
 **/
NALU_HYPRE_Int
NALU_HYPRE_VersionNumber( NALU_HYPRE_Int  *major_ptr,
                     NALU_HYPRE_Int  *minor_ptr,
                     NALU_HYPRE_Int  *patch_ptr,
                     NALU_HYPRE_Int  *single_ptr );

/*--------------------------------------------------------------------------
 * HYPRE AP user functions
 *--------------------------------------------------------------------------*/

/*Checks whether the AP is on */
NALU_HYPRE_Int NALU_HYPRE_AssumedPartitionCheck();

/*--------------------------------------------------------------------------
 * HYPRE memory location
 *--------------------------------------------------------------------------*/

typedef enum _NALU_HYPRE_MemoryLocation
{
   NALU_HYPRE_MEMORY_UNDEFINED = -1,
   NALU_HYPRE_MEMORY_HOST,
   NALU_HYPRE_MEMORY_DEVICE
} NALU_HYPRE_MemoryLocation;

NALU_HYPRE_Int NALU_HYPRE_SetMemoryLocation(NALU_HYPRE_MemoryLocation memory_location);
NALU_HYPRE_Int NALU_HYPRE_GetMemoryLocation(NALU_HYPRE_MemoryLocation *memory_location);

#include <stdlib.h>

/*--------------------------------------------------------------------------
 * HYPRE execution policy
 *--------------------------------------------------------------------------*/

typedef enum _NALU_HYPRE_ExecutionPolicy
{
   NALU_HYPRE_EXEC_UNDEFINED = -1,
   NALU_HYPRE_EXEC_HOST,
   NALU_HYPRE_EXEC_DEVICE
} NALU_HYPRE_ExecutionPolicy;

NALU_HYPRE_Int NALU_HYPRE_SetExecutionPolicy(NALU_HYPRE_ExecutionPolicy exec_policy);
NALU_HYPRE_Int NALU_HYPRE_GetExecutionPolicy(NALU_HYPRE_ExecutionPolicy *exec_policy);

/*--------------------------------------------------------------------------
 * HYPRE UMPIRE
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_SetUmpireDevicePoolSize(size_t nbytes);
NALU_HYPRE_Int NALU_HYPRE_SetUmpireUMPoolSize(size_t nbytes);
NALU_HYPRE_Int NALU_HYPRE_SetUmpireHostPoolSize(size_t nbytes);
NALU_HYPRE_Int NALU_HYPRE_SetUmpirePinnedPoolSize(size_t nbytes);
NALU_HYPRE_Int NALU_HYPRE_SetUmpireDevicePoolName(const char *pool_name);
NALU_HYPRE_Int NALU_HYPRE_SetUmpireUMPoolName(const char *pool_name);
NALU_HYPRE_Int NALU_HYPRE_SetUmpireHostPoolName(const char *pool_name);
NALU_HYPRE_Int NALU_HYPRE_SetUmpirePinnedPoolName(const char *pool_name);

/*--------------------------------------------------------------------------
 * HYPRE GPU memory pool
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_SetGPUMemoryPoolSize(NALU_HYPRE_Int bin_growth, NALU_HYPRE_Int min_bin, NALU_HYPRE_Int max_bin,
                                     size_t max_cached_bytes);

/*--------------------------------------------------------------------------
 * HYPRE handle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_SetSpTransUseVendor( NALU_HYPRE_Int use_vendor );
NALU_HYPRE_Int NALU_HYPRE_SetSpMVUseVendor( NALU_HYPRE_Int use_vendor );
/* Backwards compatibility with NALU_HYPRE_SetSpGemmUseCusparse() */
#define NALU_HYPRE_SetSpGemmUseCusparse(use_vendor) NALU_HYPRE_SetSpGemmUseVendor(use_vendor)
NALU_HYPRE_Int NALU_HYPRE_SetSpGemmUseVendor( NALU_HYPRE_Int use_vendor );
NALU_HYPRE_Int NALU_HYPRE_SetUseGpuRand( NALU_HYPRE_Int use_curand );

#ifdef __cplusplus
}
#endif

#endif
