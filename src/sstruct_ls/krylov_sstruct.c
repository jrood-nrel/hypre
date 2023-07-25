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

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SStructKrylovCAlloc( size_t count,
                           size_t elt_size,
                           NALU_HYPRE_MemoryLocation location )
{
   return ( (void*) nalu_hypre_CTAlloc(char, count * elt_size, location) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovFree( void *ptr )
{
   nalu_hypre_TFree( ptr, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SStructKrylovCreateVector( void *vvector )
{
   nalu_hypre_SStructVector  *vector = (nalu_hypre_SStructVector  *)vvector;
   nalu_hypre_SStructVector  *new_vector;
   NALU_HYPRE_Int             object_type;

   NALU_HYPRE_Int             nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *svector;
   nalu_hypre_SStructPVector *new_pvector;
   nalu_hypre_StructVector   *new_svector;
   NALU_HYPRE_Int            *num_ghost;

   NALU_HYPRE_Int    part;
   NALU_HYPRE_Int    nvars, var;

   object_type = nalu_hypre_SStructVectorObjectType(vector);

   NALU_HYPRE_SStructVectorCreate(nalu_hypre_SStructVectorComm(vector),
                             nalu_hypre_SStructVectorGrid(vector),
                             &new_vector);
   NALU_HYPRE_SStructVectorSetObjectType(new_vector, object_type);

   if (object_type == NALU_HYPRE_SSTRUCT || object_type == NALU_HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pvector    = nalu_hypre_SStructVectorPVector(vector, part);
         new_pvector = nalu_hypre_SStructVectorPVector(new_vector, part);
         nvars      = nalu_hypre_SStructPVectorNVars(pvector);

         for (var = 0; var < nvars; var++)
         {
            svector = nalu_hypre_SStructPVectorSVector(pvector, var);
            num_ghost = nalu_hypre_StructVectorNumGhost(svector);

            new_svector = nalu_hypre_SStructPVectorSVector(new_pvector, var);
            nalu_hypre_StructVectorSetNumGhost(new_svector, num_ghost);
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
nalu_hypre_SStructKrylovCreateVectorArray(NALU_HYPRE_Int n, void *vvector )
{
   nalu_hypre_SStructVector  *vector = (nalu_hypre_SStructVector  *)vvector;
   nalu_hypre_SStructVector  **new_vector;
   NALU_HYPRE_Int             object_type;

   NALU_HYPRE_Int             nparts = nalu_hypre_SStructVectorNParts(vector);
   nalu_hypre_SStructPVector *pvector;
   nalu_hypre_StructVector   *svector;
   nalu_hypre_SStructPVector *new_pvector;
   nalu_hypre_StructVector   *new_svector;
   NALU_HYPRE_Int            *num_ghost;

   NALU_HYPRE_Int    part;
   NALU_HYPRE_Int    nvars, var;

   NALU_HYPRE_Int i;

   object_type = nalu_hypre_SStructVectorObjectType(vector);

   new_vector = nalu_hypre_CTAlloc(nalu_hypre_SStructVector*, n, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      NALU_HYPRE_SStructVectorCreate(nalu_hypre_SStructVectorComm(vector),
                                nalu_hypre_SStructVectorGrid(vector),
                                &new_vector[i]);
      NALU_HYPRE_SStructVectorSetObjectType(new_vector[i], object_type);

      if (object_type == NALU_HYPRE_SSTRUCT || object_type == NALU_HYPRE_STRUCT)
      {
         for (part = 0; part < nparts; part++)
         {
            pvector    = nalu_hypre_SStructVectorPVector(vector, part);
            new_pvector = nalu_hypre_SStructVectorPVector(new_vector[i], part);
            nvars      = nalu_hypre_SStructPVectorNVars(pvector);

            for (var = 0; var < nvars; var++)
            {
               svector = nalu_hypre_SStructPVectorSVector(pvector, var);
               num_ghost = nalu_hypre_StructVectorNumGhost(svector);

               new_svector = nalu_hypre_SStructPVectorSVector(new_pvector, var);
               nalu_hypre_StructVectorSetNumGhost(new_svector, num_ghost);
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
nalu_hypre_SStructKrylovDestroyVector( void *vvector )
{
   nalu_hypre_SStructVector *vector = (nalu_hypre_SStructVector  *)vvector;

   return ( NALU_HYPRE_SStructVectorDestroy( vector ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SStructKrylovMatvecCreate( void   *A,
                                 void   *x )
{
   void *matvec_data;

   nalu_hypre_SStructMatvecCreate( &matvec_data );
   nalu_hypre_SStructMatvecSetup( matvec_data,
                             (nalu_hypre_SStructMatrix *) A,
                             (nalu_hypre_SStructVector *) x );

   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovMatvec( void   *matvec_data,
                           NALU_HYPRE_Complex  alpha,
                           void   *A,
                           void   *x,
                           NALU_HYPRE_Complex  beta,
                           void   *y )
{
   return ( nalu_hypre_SStructMatvec( alpha,
                                 (nalu_hypre_SStructMatrix *) A,
                                 (nalu_hypre_SStructVector *) x,
                                 beta,
                                 (nalu_hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovMatvecDestroy( void *matvec_data )
{
   return ( nalu_hypre_SStructMatvecDestroy( matvec_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_SStructKrylovInnerProd( void *x,
                              void *y )
{
   NALU_HYPRE_Real result;

   nalu_hypre_SStructInnerProd( (nalu_hypre_SStructVector *) x,
                           (nalu_hypre_SStructVector *) y, &result );

   return result;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovCopyVector( void *x,
                               void *y )
{
   return ( nalu_hypre_SStructCopy( (nalu_hypre_SStructVector *) x,
                               (nalu_hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovClearVector( void *x )
{
   return ( nalu_hypre_SStructVectorSetConstantValues( (nalu_hypre_SStructVector *) x,
                                                  0.0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovScaleVector( NALU_HYPRE_Complex  alpha,
                                void   *x )
{
   return ( nalu_hypre_SStructScale( alpha, (nalu_hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovAxpy( NALU_HYPRE_Complex alpha,
                         void   *x,
                         void   *y )
{
   return ( nalu_hypre_SStructAxpy( alpha, (nalu_hypre_SStructVector *) x,
                               (nalu_hypre_SStructVector *) y ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructKrylovCommInfo( void  *A,
                             NALU_HYPRE_Int   *my_id,
                             NALU_HYPRE_Int   *num_procs )
{
   MPI_Comm comm = nalu_hypre_SStructMatrixComm((nalu_hypre_SStructMatrix *) A);
   nalu_hypre_MPI_Comm_size(comm, num_procs);
   nalu_hypre_MPI_Comm_rank(comm, my_id);
   return nalu_hypre_error_flag;
}

