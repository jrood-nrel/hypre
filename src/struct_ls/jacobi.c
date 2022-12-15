/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

typedef struct
{
   void  *relax_data;

} nalu_hypre_JacobiData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_JacobiCreate( MPI_Comm  comm )
{
   nalu_hypre_JacobiData *jacobi_data;
   void              *relax_data;
   nalu_hypre_Index       stride;
   nalu_hypre_Index       indices[1];

   jacobi_data = nalu_hypre_CTAlloc(nalu_hypre_JacobiData,  1, NALU_HYPRE_MEMORY_HOST);
   relax_data = nalu_hypre_PointRelaxCreate(comm);
   nalu_hypre_PointRelaxSetNumPointsets(relax_data, 1);
   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   nalu_hypre_SetIndex3(indices[0], 0, 0, 0);
   nalu_hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
   nalu_hypre_PointRelaxSetTol(relax_data, 1.0e-6);
   (jacobi_data -> relax_data) = relax_data;

   return (void *) jacobi_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiDestroy( void *jacobi_vdata )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   if (jacobi_data)
   {
      nalu_hypre_PointRelaxDestroy(jacobi_data -> relax_data);
      nalu_hypre_TFree(jacobi_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSetup( void               *jacobi_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *x            )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxSetup((jacobi_data -> relax_data), A, b, x);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSolve( void               *jacobi_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *x            )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelax((jacobi_data -> relax_data), A, b, x);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSetTol( void   *jacobi_vdata,
                    NALU_HYPRE_Real  tol          )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxSetTol((jacobi_data -> relax_data), tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiGetTol( void   *jacobi_vdata,
                    NALU_HYPRE_Real *tol          )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxGetTol((jacobi_data -> relax_data), tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSetMaxIter( void  *jacobi_vdata,
                        NALU_HYPRE_Int    max_iter     )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxSetMaxIter((jacobi_data -> relax_data), max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiGetMaxIter( void  *jacobi_vdata,
                        NALU_HYPRE_Int  * max_iter     )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxGetMaxIter((jacobi_data -> relax_data), max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSetZeroGuess( void  *jacobi_vdata,
                          NALU_HYPRE_Int    zero_guess   )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxSetZeroGuess((jacobi_data -> relax_data), zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiGetZeroGuess( void  *jacobi_vdata,
                          NALU_HYPRE_Int  * zero_guess   )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxGetZeroGuess((jacobi_data -> relax_data), zero_guess);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiGetNumIterations( void  *jacobi_vdata,
                              NALU_HYPRE_Int  * num_iterations   )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxGetNumIterations((jacobi_data -> relax_data), num_iterations );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_JacobiSetTempVec( void               *jacobi_vdata,
                        nalu_hypre_StructVector *t            )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;

   nalu_hypre_PointRelaxSetTempVec((jacobi_data -> relax_data), t);

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_JacobiGetFinalRelativeResidualNorm( void * jacobi_vdata,
                                                    NALU_HYPRE_Real * norm )
{
   nalu_hypre_JacobiData *jacobi_data = (nalu_hypre_JacobiData *)jacobi_vdata;
   void *relax_data = jacobi_data -> relax_data;

   return nalu_hypre_PointRelaxGetFinalRelativeResidualNorm( relax_data, norm );
}
