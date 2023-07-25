/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_IJVector interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"

#include "../NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorCreate( MPI_Comm        comm,
                      NALU_HYPRE_BigInt    jlower,
                      NALU_HYPRE_BigInt    jupper,
                      NALU_HYPRE_IJVector *vector )
{
   nalu_hypre_IJVector *vec;
   NALU_HYPRE_Int       num_procs, my_id;
   NALU_HYPRE_BigInt    row0, rowN;

   vec = nalu_hypre_CTAlloc(nalu_hypre_IJVector,  1, NALU_HYPRE_MEMORY_HOST);

   if (!vec)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_MEMORY);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (jlower > jupper + 1 || jlower < 0)
   {
      nalu_hypre_error_in_arg(2);
      nalu_hypre_TFree(vec, NALU_HYPRE_MEMORY_HOST);
      return nalu_hypre_error_flag;
   }
   if (jupper < -1)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   /* now we need the global number of rows as well
      as the global first row index */

   /* proc 0 has the first row  */
   if (my_id == 0)
   {
      row0 = jlower;
   }
   nalu_hypre_MPI_Bcast(&row0, 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   /* proc (num_procs-1) has the last row  */
   if (my_id == (num_procs - 1))
   {
      rowN = jupper;
   }
   nalu_hypre_MPI_Bcast(&rowN, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   nalu_hypre_IJVectorGlobalFirstRow(vec) = row0;
   nalu_hypre_IJVectorGlobalNumRows(vec) = rowN - row0 + 1;

   nalu_hypre_IJVectorComm(vec)            = comm;
   nalu_hypre_IJVectorNumComponents(vec)   = 1;
   nalu_hypre_IJVectorObjectType(vec)      = NALU_HYPRE_UNITIALIZED;
   nalu_hypre_IJVectorObject(vec)          = NULL;
   nalu_hypre_IJVectorTranslator(vec)      = NULL;
   nalu_hypre_IJVectorAssumedPart(vec)     = NULL;
   nalu_hypre_IJVectorPrintLevel(vec)      = 0;
   nalu_hypre_IJVectorPartitioning(vec)[0] = jlower;
   nalu_hypre_IJVectorPartitioning(vec)[1] = jupper + 1;

   *vector = (NALU_HYPRE_IJVector) vec;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetNumComponents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetNumComponents( NALU_HYPRE_IJVector vector,
                                NALU_HYPRE_Int      num_components )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (num_components < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJVectorNumComponents(vector) = num_components;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetComponent
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetComponent( NALU_HYPRE_IJVector vector,
                            NALU_HYPRE_Int      component )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_IJVectorSetComponentPar(vector, component);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorDestroy( NALU_HYPRE_IJVector vector )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_IJVectorAssumedPart(vec))
   {
      nalu_hypre_AssumedPartitionDestroy((nalu_hypre_IJAssumedPart*)nalu_hypre_IJVectorAssumedPart(vec));
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJVectorDestroyPar(vec);
      if (nalu_hypre_IJVectorTranslator(vec))
      {
         nalu_hypre_AuxParVectorDestroy((nalu_hypre_AuxParVector *)
                                   (nalu_hypre_IJVectorTranslator(vec)));
      }
   }
   else if ( nalu_hypre_IJVectorObjectType(vec) != -1 )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_TFree(vec, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorInitialize( NALU_HYPRE_IJVector vector )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      if (!nalu_hypre_IJVectorObject(vec))
      {
         nalu_hypre_IJVectorCreatePar(vec, nalu_hypre_IJVectorPartitioning(vec));
      }

      nalu_hypre_IJVectorInitializePar(vec);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_IJVectorInitialize_v2( NALU_HYPRE_IJVector vector, NALU_HYPRE_MemoryLocation memory_location )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      if (!nalu_hypre_IJVectorObject(vec))
      {
         nalu_hypre_IJVectorCreatePar(vec, nalu_hypre_IJVectorPartitioning(vec));
      }

      nalu_hypre_IJVectorInitializePar_v2(vec, memory_location);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetPrintLevel( NALU_HYPRE_IJVector vector,
                             NALU_HYPRE_Int print_level )
{
   nalu_hypre_IJVector *ijvector = (nalu_hypre_IJVector *) vector;

   if (!ijvector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJVectorPrintLevel(ijvector) = 1;
   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetValues( NALU_HYPRE_IJVector        vector,
                         NALU_HYPRE_Int             nvalues,
                         const NALU_HYPRE_BigInt   *indices,
                         const NALU_HYPRE_Complex  *values   )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (nvalues == 0) { return nalu_hypre_error_flag; }

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nvalues < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJVectorMemoryLocation(vector) );

      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         return ( nalu_hypre_IJVectorSetAddValuesParDevice(vec, nvalues, indices, values, "set") );
      }
      else
#endif
      {
         return ( nalu_hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );
      }
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorAddToValues( NALU_HYPRE_IJVector        vector,
                           NALU_HYPRE_Int             nvalues,
                           const NALU_HYPRE_BigInt   *indices,
                           const NALU_HYPRE_Complex  *values )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (nvalues == 0) { return nalu_hypre_error_flag; }

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nvalues < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJVectorMemoryLocation(vector) );

      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         return ( nalu_hypre_IJVectorSetAddValuesParDevice(vec, nvalues, indices, values, "add") );
      }
      else
#endif
      {
         return ( nalu_hypre_IJVectorAddToValuesPar(vec, nvalues, indices, values) );
      }
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorAssemble( NALU_HYPRE_IJVector vector )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJVectorMemoryLocation(vector) );

      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         return ( nalu_hypre_IJVectorAssembleParDevice(vec) );
      }
      else
#endif
      {
         return ( nalu_hypre_IJVectorAssemblePar(vec) );
      }
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorUpdateValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorUpdateValues( NALU_HYPRE_IJVector        vector,
                            NALU_HYPRE_Int             nvalues,
                            const NALU_HYPRE_BigInt   *indices,
                            const NALU_HYPRE_Complex  *values,
                            NALU_HYPRE_Int             action )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (nvalues == 0) { return nalu_hypre_error_flag; }

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nvalues < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
#if defined(NALU_HYPRE_USING_GPU)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJVectorMemoryLocation(vector) );

      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         return ( nalu_hypre_IJVectorUpdateValuesDevice(vec, nvalues, indices, values, action) );
      }
      else
#endif
      {
         if (action == 1)
         {
            return ( nalu_hypre_IJVectorSetValuesPar(vec, nvalues, indices, values) );
         }
         else
         {
            return ( nalu_hypre_IJVectorAddToValuesPar(vec, nvalues, indices, values) );
         }
      }
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorGetValues( NALU_HYPRE_IJVector      vector,
                         NALU_HYPRE_Int           nvalues,
                         const NALU_HYPRE_BigInt *indices,
                         NALU_HYPRE_Complex      *values )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (nvalues == 0) { return nalu_hypre_error_flag; }

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nvalues < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJVectorGetValuesPar(vec, nvalues, indices, values) );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetMaxOffProcElmts( NALU_HYPRE_IJVector vector,
                                  NALU_HYPRE_Int      max_off_proc_elmts )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJVectorObjectType(vec) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJVectorSetMaxOffProcElmtsPar(vec, max_off_proc_elmts));
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorSetObjectType( NALU_HYPRE_IJVector vector,
                             NALU_HYPRE_Int      type )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJVectorObjectType(vec) = type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorGetObjectType( NALU_HYPRE_IJVector  vector,
                             NALU_HYPRE_Int      *type )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *type = nalu_hypre_IJVectorObjectType(vec);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetLocalRange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorGetLocalRange( NALU_HYPRE_IJVector  vector,
                             NALU_HYPRE_BigInt   *jlower,
                             NALU_HYPRE_BigInt   *jupper )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *jlower = nalu_hypre_IJVectorPartitioning(vec)[0];
   *jupper = nalu_hypre_IJVectorPartitioning(vec)[1] - 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorGetObject( NALU_HYPRE_IJVector   vector,
                         void           **object )
{
   nalu_hypre_IJVector *vec = (nalu_hypre_IJVector *) vector;

   if (!vec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *object = nalu_hypre_IJVectorObject(vec);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorRead
 * create IJVector on host memory
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorRead( const char     *filename,
                    MPI_Comm        comm,
                    NALU_HYPRE_Int       type,
                    NALU_HYPRE_IJVector *vector_ptr )
{
   NALU_HYPRE_IJVector  vector;
   NALU_HYPRE_BigInt    jlower, jupper, j;
   NALU_HYPRE_Complex   value;
   NALU_HYPRE_Int       myid, ret;
   char            new_filename[255];
   FILE           *file;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_fscanf(file, "%b %b", &jlower, &jupper);
   NALU_HYPRE_IJVectorCreate(comm, jlower, jupper, &vector);

   NALU_HYPRE_IJVectorSetObjectType(vector, type);

   NALU_HYPRE_IJVectorInitialize_v2(vector, NALU_HYPRE_MEMORY_HOST);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  This is done with %*[ \t].  Using a
    * space here causes an input line with a single decimal value on it to be
    * read as if it were an integer followed by a decimal value. */
   while ( (ret = nalu_hypre_fscanf(file, "%b%*[ \t]%le", &j, &value)) != EOF )
   {
      if (ret != 2)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error in IJ vector input file.");
         return nalu_hypre_error_flag;
      }
      if (j < jlower || j > jupper)
      {
         NALU_HYPRE_IJVectorAddToValues(vector, 1, &j, &value);
      }
      else
      {
         NALU_HYPRE_IJVectorSetValues(vector, 1, &j, &value);
      }
   }

   NALU_HYPRE_IJVectorAssemble(vector);

   fclose(file);

   *vector_ptr = vector;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorPrint( NALU_HYPRE_IJVector  vector,
                     const char     *filename )
{
   MPI_Comm        comm;
   NALU_HYPRE_BigInt   *partitioning;
   NALU_HYPRE_BigInt    jlower, jupper, j;
   NALU_HYPRE_Complex  *h_values = NULL, *d_values = NULL, *values = NULL;
   NALU_HYPRE_Int       myid, n_local;
   char            new_filename[255];
   FILE           *file;

   if (!vector)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   comm = nalu_hypre_IJVectorComm(vector);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   partitioning = nalu_hypre_IJVectorPartitioning(vector);
   jlower = partitioning[0];
   jupper = partitioning[1] - 1;
   n_local = jupper - jlower + 1;

   nalu_hypre_fprintf(file, "%b %b\n", jlower, jupper);

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_IJVectorMemoryLocation(vector);

   d_values = nalu_hypre_TAlloc(NALU_HYPRE_Complex, n_local, memory_location);

   NALU_HYPRE_IJVectorGetValues(vector, n_local, NULL, d_values);

   if ( nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_HOST )
   {
      values = d_values;
   }
   else
   {
      h_values = nalu_hypre_TAlloc(NALU_HYPRE_Complex, n_local, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(h_values, d_values, NALU_HYPRE_Complex, n_local, NALU_HYPRE_MEMORY_HOST, memory_location);
      values = h_values;
   }

   for (j = jlower; j <= jupper; j++)
   {
      nalu_hypre_fprintf(file, "%b %.14e\n", j, values[j - jlower]);
   }

   nalu_hypre_TFree(d_values, memory_location);
   nalu_hypre_TFree(h_values, NALU_HYPRE_MEMORY_HOST);

   fclose(file);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJVectorInnerProd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJVectorInnerProd( NALU_HYPRE_IJVector  x,
                         NALU_HYPRE_IJVector  y,
                         NALU_HYPRE_Real     *prod )
{
   nalu_hypre_IJVector *xvec = (nalu_hypre_IJVector *) x;
   nalu_hypre_IJVector *yvec = (nalu_hypre_IJVector *) y;

   if (!xvec)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (!yvec)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_IJVectorObjectType(xvec) != nalu_hypre_IJVectorObjectType(yvec))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Input vectors don't have the same object type!");
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_IJVectorObjectType(xvec) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_ParVector *par_x = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(xvec);
      nalu_hypre_ParVector *par_y = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(yvec);

      NALU_HYPRE_ParVectorInnerProd(par_x, par_y, prod);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}
