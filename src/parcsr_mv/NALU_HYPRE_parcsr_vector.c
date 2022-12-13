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

#include "_hypre_parcsr_mv.h"

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
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             hypre_ParVectorCreate(comm, global_size, partitioning) ;
   return hypre_error_flag;
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
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             hypre_ParMultiVectorCreate( comm, global_size, partitioning, number_vectors );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorDestroy( NALU_HYPRE_ParVector vector )
{
   return ( hypre_ParVectorDestroy( (hypre_ParVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorInitialize( NALU_HYPRE_ParVector vector )
{
   return ( hypre_ParVectorInitialize( (hypre_ParVector *) vector ) );
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
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector) hypre_ParVectorRead( comm, file_name ) ;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorPrint( NALU_HYPRE_ParVector  vector,
                      const char      *file_name )
{
   return ( hypre_ParVectorPrint( (hypre_ParVector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorSetConstantValues( NALU_HYPRE_ParVector  vector,
                                  NALU_HYPRE_Complex    value )
{
   return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) vector,
                                              value ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorSetRandomValues( NALU_HYPRE_ParVector  vector,
                                NALU_HYPRE_Int        seed  )
{
   return ( hypre_ParVectorSetRandomValues( (hypre_ParVector *) vector,
                                            seed ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorCopy( NALU_HYPRE_ParVector x,
                     NALU_HYPRE_ParVector y )
{
   return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
                                 (hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_ParVector
NALU_HYPRE_ParVectorCloneShallow( NALU_HYPRE_ParVector x )
{
   return ( (NALU_HYPRE_ParVector)
            hypre_ParVectorCloneShallow( (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParVectorScale( NALU_HYPRE_Complex   value,
                      NALU_HYPRE_ParVector x)
{
   return ( hypre_ParVectorScale( value, (hypre_ParVector *) x) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_ParVectorAxpy( NALU_HYPRE_Complex   alpha,
                     NALU_HYPRE_ParVector x,
                     NALU_HYPRE_ParVector y )
{
   return hypre_ParVectorAxpy( alpha, (hypre_ParVector *)x, (hypre_ParVector *)y );
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
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (!y)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   *prod = hypre_ParVectorInnerProd( (hypre_ParVector *) x,
                                     (hypre_ParVector *) y) ;
   return hypre_error_flag;
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
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }
   *vector = (NALU_HYPRE_ParVector)
             hypre_VectorToParVector (comm, (hypre_Vector *) b, partitioning);
   return hypre_error_flag;
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
   hypre_ParVector *par_vector = (hypre_ParVector *) vector;

   if (!par_vector)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (num_values < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   if (!values)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   hypre_ParVectorGetValues(par_vector, num_values, indices, values);
   return hypre_error_flag;
}
