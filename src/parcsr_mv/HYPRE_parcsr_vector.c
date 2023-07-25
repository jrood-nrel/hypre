/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParVector interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorCreate( MPI_Comm         comm,
                       NALU_HYPRE_BigInt     global_size,
                       NALU_HYPRE_BigInt    *partitioning,
                       NALU_HYPRE_ParVector *vector )
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             nalu_hypre_ParVectorCreate(comm, global_size, partitioning) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParMultiVectorCreate( MPI_Comm         comm,
                            NALU_HYPRE_BigInt     global_size,
                            NALU_HYPRE_BigInt    *partitioning,
                            NALU_HYPRE_Int        number_vectors,
                            NALU_HYPRE_ParVector *vector )
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(5);
      return nalu_hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             nalu_hypre_ParMultiVectorCreate( comm, global_size, partitioning, number_vectors );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorDestroy( NALU_HYPRE_ParVector vector )
{
   return ( nalu_hypre_ParVectorDestroy( (nalu_hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorInitialize( NALU_HYPRE_ParVector vector )
{
   return ( nalu_hypre_ParVectorInitialize( (nalu_hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorRead( MPI_Comm         comm,
                     const char      *file_name,
                     NALU_HYPRE_ParVector *vector)
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector) nalu_hypre_ParVectorRead( comm, file_name ) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorPrint( NALU_HYPRE_ParVector  vector,
                      const char      *file_name )
{
   return ( nalu_hypre_ParVectorPrint( (nalu_hypre_ParVector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorSetConstantValues( NALU_HYPRE_ParVector  vector,
                                  NALU_HYPRE_Complex    value )
{
   return ( nalu_hypre_ParVectorSetConstantValues( (nalu_hypre_ParVector *) vector,
                                              value ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorSetRandomValues( NALU_HYPRE_ParVector  vector,
                                NALU_HYPRE_Int        seed  )
{
   return ( nalu_hypre_ParVectorSetRandomValues( (nalu_hypre_ParVector *) vector,
                                            seed ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorCopy( NALU_HYPRE_ParVector x,
                     NALU_HYPRE_ParVector y )
{
   return ( nalu_hypre_ParVectorCopy( (nalu_hypre_ParVector *) x,
                                 (nalu_hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParVector
NALU_HYPRE_ParVectorCloneShallow( NALU_HYPRE_ParVector x )
{
   return ( (NALU_HYPRE_ParVector)
            nalu_hypre_ParVectorCloneShallow( (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorScale( NALU_HYPRE_Complex   value,
                      NALU_HYPRE_ParVector x)
{
   return ( nalu_hypre_ParVectorScale( value, (nalu_hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorAxpy( NALU_HYPRE_Complex   alpha,
                     NALU_HYPRE_ParVector x,
                     NALU_HYPRE_ParVector y )
{
   return nalu_hypre_ParVectorAxpy( alpha, (nalu_hypre_ParVector *)x, (nalu_hypre_ParVector *)y );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorInnerProd( NALU_HYPRE_ParVector x,
                          NALU_HYPRE_ParVector y,
                          NALU_HYPRE_Real     *prod)
{
   if (!x)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (!y)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   *prod = nalu_hypre_ParVectorInnerProd( (nalu_hypre_ParVector *) x,
                                     (nalu_hypre_ParVector *) y) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorToParVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorToParVector( MPI_Comm         comm,
                         NALU_HYPRE_Vector     b,
                         NALU_HYPRE_BigInt    *partitioning,
                         NALU_HYPRE_ParVector *vector)
{
   if (!vector)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             nalu_hypre_VectorToParVector (comm, (nalu_hypre_Vector *) b, partitioning);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorGetValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorGetValues( NALU_HYPRE_ParVector vector,
                          NALU_HYPRE_Int       num_values,
                          NALU_HYPRE_BigInt   *indices,
                          NALU_HYPRE_Complex  *values)
{
   nalu_hypre_ParVector *par_vector = (nalu_hypre_ParVector *) vector;

   if (!par_vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_values < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (!values)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParVectorGetValues(par_vector, num_values, indices, values);
   return nalu_hypre_error_flag;
}
