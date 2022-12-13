/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct matrix-vector implementation of Krylov interface routines.
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCAlloc( size_t count,
                           size_t elt_size,
                           NALU_HYPRE_MemoryLocation location )
{
   return ( (void*) hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovFree( void *ptr )
{
   hypre_TFree( ptr, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVector( void *vvector )
{
   hypre_SStructVector  *vector = (hypre_SStructVector  *)vvector;
   hypre_SStructVector  *new_vector;
   NALU_HYPRE_Int             object_type;

   NALU_HYPRE_Int             nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;
   hypre_SStructPVector *new_pvector;
   hypre_StructVector   *new_svector;
   NALU_HYPRE_Int            *num_ghost;

   NALU_HYPRE_Int    part;
   NALU_HYPRE_Int    nvars, var;

   object_type = hypre_SStructVectorObjectType(vector);

   NALU_HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                             hypre_SStructVectorGrid(vector),
                             &new_vector);
   NALU_HYPRE_SStructVectorSetObjectType(new_vector, object_type);

   if (object_type == NALU_HYPRE_SSTRUCT || object_type == NALU_HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pvector    = hypre_SStructVectorPVector(vector, part);
         new_pvector = hypre_SStructVectorPVector(new_vector, part);
         nvars      = hypre_SStructPVectorNVars(pvector);

         for (var = 0; var < nvars; var++)
         {
            svector = hypre_SStructPVectorSVector(pvector, var);
            num_ghost = hypre_StructVectorNumGhost(svector);

            new_svector = hypre_SStructPVectorSVector(new_pvector, var);
            hypre_StructVectorSetNumGhost(new_svector, num_ghost);
         }
      }
   }

   NALU_HYPRE_SStructVectorInitialize(new_vector);
   NALU_HYPRE_SStructVectorAssemble(new_vector);

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovCreateVectorArray(NALU_HYPRE_Int n, void *vvector )
{
   hypre_SStructVector  *vector = (hypre_SStructVector  *)vvector;
   hypre_SStructVector  **new_vector;
   NALU_HYPRE_Int             object_type;

   NALU_HYPRE_Int             nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   hypre_StructVector   *svector;
   hypre_SStructPVector *new_pvector;
   hypre_StructVector   *new_svector;
   NALU_HYPRE_Int            *num_ghost;

   NALU_HYPRE_Int    part;
   NALU_HYPRE_Int    nvars, var;

   NALU_HYPRE_Int i;

   object_type = hypre_SStructVectorObjectType(vector);

   new_vector = hypre_CTAlloc(hypre_SStructVector*, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      NALU_HYPRE_SStructVectorCreate(hypre_SStructVectorComm(vector),
                                hypre_SStructVectorGrid(vector),
                                &new_vector[i]);
      NALU_HYPRE_SStructVectorSetObjectType(new_vector[i], object_type);

      if (object_type == NALU_HYPRE_SSTRUCT || object_type == NALU_HYPRE_STRUCT)
      {
         for (part = 0; part < nparts; part++)
         {
            pvector    = hypre_SStructVectorPVector(vector, part);
            new_pvector = hypre_SStructVectorPVector(new_vector[i], part);
            nvars      = hypre_SStructPVectorNVars(pvector);

            for (var = 0; var < nvars; var++)
            {
               svector = hypre_SStructPVectorSVector(pvector, var);
               num_ghost = hypre_StructVectorNumGhost(svector);

               new_svector = hypre_SStructPVectorSVector(new_pvector, var);
               hypre_StructVectorSetNumGhost(new_svector, num_ghost);
            }
         }
      }

      NALU_HYPRE_SStructVectorInitialize(new_vector[i]);
      NALU_HYPRE_SStructVectorAssemble(new_vector[i]);
   }

   return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovDestroyVector( void *vvector )
{
   hypre_SStructVector *vector = (hypre_SStructVector  *)vvector;

   return ( NALU_HYPRE_SStructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SStructKrylovMatvecCreate( void   *A,
                                 void   *x )
{
   void *matvec_data;

   hypre_SStructMatvecCreate( &matvec_data );
   hypre_SStructMatvecSetup( matvec_data,
                             (hypre_SStructMatrix *) A,
                             (hypre_SStructVector *) x );

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovMatvec( void   *matvec_data,
                           NALU_HYPRE_Complex  alpha,
                           void   *A,
                           void   *x,
                           NALU_HYPRE_Complex  beta,
                           void   *y )
{
   return ( hypre_SStructMatvec( alpha,
                                 (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) x,
                                 beta,
                                 (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovMatvecDestroy( void *matvec_data )
{
   return ( hypre_SStructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
hypre_SStructKrylovInnerProd( void *x,
                              void *y )
{
   NALU_HYPRE_Real result;

   hypre_SStructInnerProd( (hypre_SStructVector *) x,
                           (hypre_SStructVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovCopyVector( void *x,
                               void *y )
{
   return ( hypre_SStructCopy( (hypre_SStructVector *) x,
                               (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovClearVector( void *x )
{
   return ( hypre_SStructVectorSetConstantValues( (hypre_SStructVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovScaleVector( NALU_HYPRE_Complex  alpha,
                                void   *x )
{
   return ( hypre_SStructScale( alpha, (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovAxpy( NALU_HYPRE_Complex alpha,
                         void   *x,
                         void   *y )
{
   return ( hypre_SStructAxpy( alpha, (hypre_SStructVector *) x,
                               (hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_SStructKrylovCommInfo( void  *A,
                             NALU_HYPRE_Int   *my_id,
                             NALU_HYPRE_Int   *num_procs )
{
   MPI_Comm comm = hypre_SStructMatrixComm((hypre_SStructMatrix *) A);
   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);
   return hypre_error_flag;
}

