/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_StructKrylovCAlloc( size_t               count,
                          size_t               elt_size,
                          NALU_HYPRE_MemoryLocation location)
{
   return ( (void*) nalu_hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovFree( void *ptr )
{
   nalu_hypre_TFree( ptr, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_StructKrylovCreateVector( void *vvector )
{
   nalu_hypre_StructVector *vector = (nalu_hypre_StructVector *)vvector;
   nalu_hypre_StructVector *new_vector;
   NALU_HYPRE_Int          *num_ghost = nalu_hypre_StructVectorNumGhost(vector);

   new_vector = nalu_hypre_StructVectorCreate( nalu_hypre_StructVectorComm(vector),
                                          nalu_hypre_StructVectorGrid(vector) );
   nalu_hypre_StructVectorSetNumGhost(new_vector, num_ghost);
   nalu_hypre_StructVectorInitialize(new_vector);
   nalu_hypre_StructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_StructKrylovCreateVectorArray(NALU_HYPRE_Int n, void *vvector )
{
   nalu_hypre_StructVector *vector = (nalu_hypre_StructVector *)vvector;
   nalu_hypre_StructVector **new_vector;
   NALU_HYPRE_Int          *num_ghost = nalu_hypre_StructVectorNumGhost(vector);
   NALU_HYPRE_Int i;

   new_vector = nalu_hypre_CTAlloc(nalu_hypre_StructVector*, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      NALU_HYPRE_StructVectorCreate(nalu_hypre_StructVectorComm(vector),
                               nalu_hypre_StructVectorGrid(vector),
                               (NALU_HYPRE_StructVector *) &new_vector[i] );
      nalu_hypre_StructVectorSetNumGhost(new_vector[i], num_ghost);
      NALU_HYPRE_StructVectorInitialize((NALU_HYPRE_StructVector) new_vector[i]);
      NALU_HYPRE_StructVectorAssemble((NALU_HYPRE_StructVector) new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovDestroyVector( void *vvector )
{
   nalu_hypre_StructVector *vector = (nalu_hypre_StructVector *)vvector;

   return ( nalu_hypre_StructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_StructKrylovMatvecCreate( void   *A,
                                void   *x )
{
   void *matvec_data;

   matvec_data = nalu_hypre_StructMatvecCreate();
   nalu_hypre_StructMatvecSetup(matvec_data, (nalu_hypre_StructMatrix *)A, (nalu_hypre_StructVector *)x);

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovMatvec( void   *matvec_data,
                          NALU_HYPRE_Complex  alpha,
                          void   *A,
                          void   *x,
                          NALU_HYPRE_Complex  beta,
                          void   *y           )
{
   return ( nalu_hypre_StructMatvecCompute( matvec_data,
                                       alpha,
                                       (nalu_hypre_StructMatrix *) A,
                                       (nalu_hypre_StructVector *) x,
                                       beta,
                                       (nalu_hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovMatvecDestroy( void *matvec_data )
{
   return ( nalu_hypre_StructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_StructKrylovInnerProd( void *x,
                             void *y )
{
   return ( nalu_hypre_StructInnerProd( (nalu_hypre_StructVector *) x,
                                   (nalu_hypre_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovCopyVector( void *x,
                              void *y )
{
   return ( nalu_hypre_StructCopy( (nalu_hypre_StructVector *) x,
                              (nalu_hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovClearVector( void *x )
{
   return ( nalu_hypre_StructVectorSetConstantValues( (nalu_hypre_StructVector *) x,
                                                 0.0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovScaleVector( NALU_HYPRE_Complex  alpha,
                               void   *x     )
{
   return ( nalu_hypre_StructScale( alpha, (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovAxpy( NALU_HYPRE_Complex alpha,
                        void   *x,
                        void   *y )
{
   return ( nalu_hypre_StructAxpy( alpha, (nalu_hypre_StructVector *) x,
                              (nalu_hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovIdentitySetup( void *vdata,
                                 void *A,
                                 void *b,
                                 void *x     )

{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovIdentity( void *vdata,
                            void *A,
                            void *b,
                            void *x     )

{
   return ( nalu_hypre_StructKrylovCopyVector( b, x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructKrylovCommInfo( void  *A,
                            NALU_HYPRE_Int   *my_id,
                            NALU_HYPRE_Int   *num_procs )
{
   MPI_Comm comm = nalu_hypre_StructMatrixComm((nalu_hypre_StructMatrix *) A);
   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);
   return nalu_hypre_error_flag;
}

