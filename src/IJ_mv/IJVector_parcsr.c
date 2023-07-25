/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJVector_Par interface
 *
 *****************************************************************************/

#include "_nalu_hypre_IJ_mv.h"
#include "../NALU_HYPRE.h"

/******************************************************************************
 *
 * nalu_hypre_IJVectorCreatePar
 *
 * creates ParVector if necessary, and leaves a pointer to it as the
 * nalu_hypre_IJVector object
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorCreatePar(nalu_hypre_IJVector *vector,
                        NALU_HYPRE_BigInt   *IJpartitioning)
{
   MPI_Comm      comm = nalu_hypre_IJVectorComm(vector);

   NALU_HYPRE_BigInt  global_n, partitioning[2], jmin;
   NALU_HYPRE_Int     j;

   jmin = nalu_hypre_IJVectorGlobalFirstRow(vector);
   global_n = nalu_hypre_IJVectorGlobalNumRows(vector);

   /* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < 2; j++)
   {
      partitioning[j] = IJpartitioning[j] - jmin;
   }

   nalu_hypre_IJVectorObject(vector) = (void*) nalu_hypre_ParVectorCreate(comm, global_n, partitioning);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorDestroyPar
 *
 * frees ParVector local storage of an IJVectorPar
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorDestroyPar(nalu_hypre_IJVector *vector)
{
   return nalu_hypre_ParVectorDestroy((nalu_hypre_ParVector*)nalu_hypre_IJVectorObject(vector));
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorInitializePar
 *
 * initializes ParVector of IJVectorPar
 *
 *****************************************************************************/
NALU_HYPRE_Int
nalu_hypre_IJVectorInitializePar(nalu_hypre_IJVector *vector)
{
   return nalu_hypre_IJVectorInitializePar_v2(vector, nalu_hypre_IJVectorMemoryLocation(vector));
}

NALU_HYPRE_Int
nalu_hypre_IJVectorInitializePar_v2(nalu_hypre_IJVector *vector, NALU_HYPRE_MemoryLocation memory_location)
{
   MPI_Comm            comm         = nalu_hypre_IJVectorComm(vector);
   nalu_hypre_ParVector    *par_vector   = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_AuxParVector *aux_vector   = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);
   NALU_HYPRE_Int           print_level  = nalu_hypre_IJVectorPrintLevel(vector);
   NALU_HYPRE_Int           num_vectors  = nalu_hypre_IJVectorNumComponents(vector);

   NALU_HYPRE_BigInt       *partitioning = nalu_hypre_ParVectorPartitioning(par_vector);
   nalu_hypre_Vector       *local_vector = nalu_hypre_ParVectorLocalVector(par_vector);

   NALU_HYPRE_Int           my_id;

   NALU_HYPRE_MemoryLocation memory_location_aux =
      nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_HOST ? NALU_HYPRE_MEMORY_HOST : NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (!partitioning)
   {
      if (print_level)
      {
         nalu_hypre_printf("No ParVector partitioning for initialization -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorInitializePar\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_VectorNumVectors(local_vector) = num_vectors;
   nalu_hypre_VectorSize(local_vector) = (NALU_HYPRE_Int)(partitioning[1] - partitioning[0]);

   nalu_hypre_ParVectorInitialize_v2(par_vector, memory_location);

   if (!aux_vector)
   {
      nalu_hypre_AuxParVectorCreate(&aux_vector);
      nalu_hypre_IJVectorTranslator(vector) = aux_vector;
   }
   nalu_hypre_AuxParVectorInitialize_v2(aux_vector, memory_location_aux);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorSetMaxOffProcElmtsPar
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorSetMaxOffProcElmtsPar(nalu_hypre_IJVector *vector,
                                    NALU_HYPRE_Int       max_off_proc_elmts)
{
   nalu_hypre_AuxParVector *aux_vector;

   aux_vector = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);
   if (!aux_vector)
   {
      nalu_hypre_AuxParVectorCreate(&aux_vector);
      nalu_hypre_IJVectorTranslator(vector) = aux_vector;
   }
   nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_AuxParVectorUsrOffProcElmts(aux_vector) = max_off_proc_elmts;
#endif

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorDistributePar
 *
 * takes an IJVector generated for one processor and distributes it
 * across many processors according to vec_starts,
 * if vec_starts is NULL, it distributes them evenly?
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorDistributePar(nalu_hypre_IJVector  *vector,
                            const NALU_HYPRE_Int *vec_starts)
{
   nalu_hypre_ParVector *old_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_ParVector *par_vector;
   NALU_HYPRE_Int print_level = nalu_hypre_IJVectorPrintLevel(vector);

   if (!old_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("old_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorDistributePar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   par_vector = nalu_hypre_VectorToParVector(nalu_hypre_ParVectorComm(old_vector),
                                        nalu_hypre_ParVectorLocalVector(old_vector),
                                        (NALU_HYPRE_BigInt *)vec_starts);
   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorDistributePar\n");
         nalu_hypre_printf("**** Vector storage is unallocated ****\n");
      }
      nalu_hypre_error_in_arg(1);
   }

   nalu_hypre_ParVectorDestroy(old_vector);

   nalu_hypre_IJVectorObject(vector) = par_vector;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorZeroValuesPar
 *
 * zeroes all local components of an IJVectorPar
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorZeroValuesPar(nalu_hypre_IJVector *vector)
{
   NALU_HYPRE_Int my_id;
   NALU_HYPRE_BigInt vec_start, vec_stop;

   nalu_hypre_ParVector *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   MPI_Comm comm = nalu_hypre_IJVectorComm(vector);
   NALU_HYPRE_BigInt *partitioning;
   nalu_hypre_Vector *local_vector;
   NALU_HYPRE_Int print_level = nalu_hypre_IJVectorPrintLevel(vector);

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorZeroValuesPar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   partitioning = nalu_hypre_ParVectorPartitioning(par_vector);
   local_vector = nalu_hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("local_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorZeroValuesPar\n");
         nalu_hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   vec_start = partitioning[0];
   vec_stop  = partitioning[1];

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         nalu_hypre_printf("vec_start > vec_stop -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorZeroValuesPar\n");
         nalu_hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_assert(nalu_hypre_VectorSize(local_vector) == (NALU_HYPRE_Int)(vec_stop - vec_start));

   nalu_hypre_SeqVectorSetConstantValues(local_vector, 0.0);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorSetComponentPar
 *
 * Set the component identifier of a vector with multiple components
 * (multivector)
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorSetComponentPar(nalu_hypre_IJVector *vector,
                              NALU_HYPRE_Int       component)
{
   NALU_HYPRE_Int        print_level = nalu_hypre_IJVectorPrintLevel(vector);
   nalu_hypre_ParVector *par_vector  = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   NALU_HYPRE_Int        num_vectors = nalu_hypre_ParVectorNumVectors(par_vector);

   if (component < 0 || component > num_vectors)
   {
      if (print_level)
      {
         nalu_hypre_printf("component < 0 || component > num_vectors -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorSetComponentPar\n");
      }
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   else
   {
      nalu_hypre_ParVectorSetComponent(par_vector, component);
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorSetValuesPar
 *
 * sets a potentially noncontiguous set of components of an IJVectorPar
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorSetValuesPar(nalu_hypre_IJVector       *vector,
                           NALU_HYPRE_Int             num_values,
                           const NALU_HYPRE_BigInt   *indices,
                           const NALU_HYPRE_Complex  *values)
{
   NALU_HYPRE_Int my_id;
   NALU_HYPRE_Int j, k;
   NALU_HYPRE_BigInt i, vec_start, vec_stop;
   NALU_HYPRE_Complex *data;
   NALU_HYPRE_Int print_level = nalu_hypre_IJVectorPrintLevel(vector);

   NALU_HYPRE_BigInt *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   nalu_hypre_ParVector *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   MPI_Comm comm = nalu_hypre_IJVectorComm(vector);
   NALU_HYPRE_Int component;
   nalu_hypre_Vector *local_vector;
   NALU_HYPRE_Int vecoffset;
   NALU_HYPRE_Int vecstride;
   NALU_HYPRE_Int idxstride;

   /* If no components are to be set, perform no checking and return */
   if (num_values < 1) { return 0; }

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorSetValuesPar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   local_vector = nalu_hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("local_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorSetValuesPar\n");
         nalu_hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1] - 1;

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         nalu_hypre_printf("vec_start > vec_stop -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorSetValuesPar\n");
         nalu_hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* Determine whether indices points to local indices only, and if not, store
      indices and values in auxiliary vector structure.  If indices == NULL,
      assume that num_values components are to be set in a block starting at
      vec_start.  NOTE: If indices == NULL off proc values are ignored!!! */

   data = nalu_hypre_VectorData(local_vector);
   component = nalu_hypre_VectorComponent(local_vector);
   vecstride = nalu_hypre_VectorVectorStride(local_vector);
   idxstride = nalu_hypre_VectorIndexStride(local_vector);
   vecoffset = component * vecstride;
   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         i = indices[j];
         if (vec_start <= i && i <= vec_stop)
         {
            k = (NALU_HYPRE_Int)(i - vec_start);
            data[vecoffset + k * idxstride] = values[j];
         }
      }
   }
   else
   {
      if (num_values > (NALU_HYPRE_Int)(vec_stop - vec_start) + 1)
      {
         if (print_level)
         {
            nalu_hypre_printf("Warning! Indices beyond local range  not identified!\n ");
            nalu_hypre_printf("Off processor values have been ignored!\n");
         }
         num_values = (NALU_HYPRE_Int)(vec_stop - vec_start) + 1;
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_values; j++)
      {
         data[vecoffset + j * idxstride] = values[j];
      }
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorAddToValuesPar
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorAddToValuesPar(nalu_hypre_IJVector       *vector,
                             NALU_HYPRE_Int             num_values,
                             const NALU_HYPRE_BigInt   *indices,
                             const NALU_HYPRE_Complex  *values)
{
   MPI_Comm            comm = nalu_hypre_IJVectorComm(vector);
   nalu_hypre_ParVector    *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_AuxParVector *aux_vector = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);
   NALU_HYPRE_BigInt       *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   NALU_HYPRE_Int           print_level = nalu_hypre_IJVectorPrintLevel(vector);

   nalu_hypre_Vector       *local_vector;
   NALU_HYPRE_Int           idxstride, vecstride;
   NALU_HYPRE_Int           component, vecoffset;
   NALU_HYPRE_Int           num_vectors;
   NALU_HYPRE_Int           my_id;
   NALU_HYPRE_Int           i, j, vec_start, vec_stop;
   NALU_HYPRE_Complex      *data;

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorAddToValuesPar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   local_vector = nalu_hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("local_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorAddToValuesPar\n");
         nalu_hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1] - 1;

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         nalu_hypre_printf("vec_start > vec_stop -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorAddToValuesPar\n");
         nalu_hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   data = nalu_hypre_VectorData(local_vector);
   num_vectors = nalu_hypre_VectorNumVectors(local_vector);
   component   = nalu_hypre_VectorComponent(local_vector);
   vecstride   = nalu_hypre_VectorVectorStride(local_vector);
   idxstride   = nalu_hypre_VectorIndexStride(local_vector);
   vecoffset   = component * vecstride;

   if (indices)
   {
      NALU_HYPRE_Int current_num_elmts
         = nalu_hypre_AuxParVectorCurrentOffProcElmts(aux_vector);
      NALU_HYPRE_Int max_off_proc_elmts
         = nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      NALU_HYPRE_BigInt *off_proc_i = nalu_hypre_AuxParVectorOffProcI(aux_vector);
      NALU_HYPRE_Complex *off_proc_data = nalu_hypre_AuxParVectorOffProcData(aux_vector);
      NALU_HYPRE_Int k;

      for (j = 0; j < num_values; j++)
      {
         i = indices[j];
         if (i < vec_start || i > vec_stop)
         {
            /* if elements outside processor boundaries, store in off processor
               stash */
            if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector) =
                  max_off_proc_elmts;
               nalu_hypre_AuxParVectorOffProcI(aux_vector)
                  = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, max_off_proc_elmts, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_AuxParVectorOffProcData(aux_vector)
                  = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max_off_proc_elmts, NALU_HYPRE_MEMORY_HOST);
               off_proc_i = nalu_hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = nalu_hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = nalu_hypre_TReAlloc(off_proc_i, NALU_HYPRE_BigInt, max_off_proc_elmts, NALU_HYPRE_MEMORY_HOST);
               off_proc_data = nalu_hypre_TReAlloc(off_proc_data, NALU_HYPRE_Complex,
                                              max_off_proc_elmts, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector)
                  = max_off_proc_elmts;
               nalu_hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               nalu_hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = i;
            off_proc_data[current_num_elmts++] = values[j];
            nalu_hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = current_num_elmts;
         }
         else /* local values are added to the vector */
         {
            k = (NALU_HYPRE_Int)(i - vec_start);
            data[vecoffset + k * idxstride] += values[j];
         }
      }

      if (current_num_elmts > 0 && num_vectors > 1)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "Off processor AddToValues not implemented for multivectors!\n");
         return nalu_hypre_error_flag;
      }
   }
   else
   {
      if (num_values > (NALU_HYPRE_Int)(vec_stop - vec_start) + 1)
      {
         if (print_level)
         {
            nalu_hypre_printf("Warning! Indices beyond local range  not identified!\n ");
            nalu_hypre_printf("Off processor values have been ignored!\n");
         }
         num_values = (NALU_HYPRE_Int)(vec_stop - vec_start) + 1;
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < num_values; j++)
      {
         data[vecoffset + j * idxstride] += values[j];
      }
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorAssemblePar
 *
 * currently tests existence of of ParVector object and its partitioning
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorAssemblePar(nalu_hypre_IJVector *vector)
{
   nalu_hypre_ParVector     *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   nalu_hypre_AuxParVector  *aux_vector = (nalu_hypre_AuxParVector*) nalu_hypre_IJVectorTranslator(vector);
   MPI_Comm             comm = nalu_hypre_IJVectorComm(vector);
   NALU_HYPRE_Int            print_level = nalu_hypre_IJVectorPrintLevel(vector);

   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorAssemblePar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
   }

   if (aux_vector)
   {
      NALU_HYPRE_Int off_proc_elmts, current_num_elmts;
      NALU_HYPRE_Int max_off_proc_elmts;
      NALU_HYPRE_BigInt *off_proc_i;
      NALU_HYPRE_Complex *off_proc_data;
      current_num_elmts = nalu_hypre_AuxParVectorCurrentOffProcElmts(aux_vector);
      nalu_hypre_MPI_Allreduce(&current_num_elmts, &off_proc_elmts, 1, NALU_HYPRE_MPI_INT,
                          nalu_hypre_MPI_SUM, comm);
      if (off_proc_elmts)
      {
         max_off_proc_elmts = nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector);
         off_proc_i = nalu_hypre_AuxParVectorOffProcI(aux_vector);
         off_proc_data = nalu_hypre_AuxParVectorOffProcData(aux_vector);
         nalu_hypre_IJVectorAssembleOffProcValsPar(vector, max_off_proc_elmts,
                                              current_num_elmts, NALU_HYPRE_MEMORY_HOST,
                                              off_proc_i, off_proc_data);
         nalu_hypre_TFree(nalu_hypre_AuxParVectorOffProcI(aux_vector), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_AuxParVectorOffProcData(aux_vector), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_AuxParVectorMaxOffProcElmts(aux_vector) = 0;
         nalu_hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = 0;
      }
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 *
 * nalu_hypre_IJVectorGetValuesPar
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorGetValuesPar(nalu_hypre_IJVector      *vector,
                           NALU_HYPRE_Int            num_values,
                           const NALU_HYPRE_BigInt  *indices,
                           NALU_HYPRE_Complex       *values)
{
   NALU_HYPRE_Int        my_id;
   MPI_Comm         comm           = nalu_hypre_IJVectorComm(vector);
   NALU_HYPRE_BigInt    *IJpartitioning = nalu_hypre_IJVectorPartitioning(vector);
   NALU_HYPRE_BigInt     vec_start;
   NALU_HYPRE_BigInt     vec_stop;
   NALU_HYPRE_BigInt     jmin           = nalu_hypre_IJVectorGlobalFirstRow(vector);
   nalu_hypre_ParVector *par_vector     = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);
   NALU_HYPRE_Int        print_level    = nalu_hypre_IJVectorPrintLevel(vector);

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1)
   {
      return 0;
   }

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("par_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorGetValuesPar\n");
         nalu_hypre_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_Vector *local_vector = nalu_hypre_ParVectorLocalVector(par_vector);
   if (!local_vector)
   {
      if (print_level)
      {
         nalu_hypre_printf("local_vector == NULL -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorGetValuesPar\n");
         nalu_hypre_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1];

   if (vec_start > vec_stop)
   {
      if (print_level)
      {
         nalu_hypre_printf("vec_start > vec_stop -- ");
         nalu_hypre_printf("nalu_hypre_IJVectorGetValuesPar\n");
         nalu_hypre_printf("**** This vector partitioning should not occur ****\n");
      }
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParVectorGetValues2(par_vector, num_values, (NALU_HYPRE_BigInt *) indices, jmin, values);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * nalu_hypre_IJVectorAssembleOffProcValsPar
 *
 * This is for handling set and get values calls to off-proc. entries - it is
 * called from assemble.  There is an alternate version for when the assumed
 * partition is being used.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_IJVectorAssembleOffProcValsPar( nalu_hypre_IJVector       *vector,
                                      NALU_HYPRE_Int             max_off_proc_elmts,
                                      NALU_HYPRE_Int             current_num_elmts,
                                      NALU_HYPRE_MemoryLocation  memory_location,
                                      NALU_HYPRE_BigInt         *off_proc_i,
                                      NALU_HYPRE_Complex        *off_proc_data)
{
   NALU_HYPRE_Int myid;
   NALU_HYPRE_BigInt global_first_row, global_num_rows;
   NALU_HYPRE_Int i, j, in, k;
   NALU_HYPRE_Int proc_id, last_proc, prev_id, tmp_id;
   NALU_HYPRE_Int max_response_size;
   NALU_HYPRE_Int ex_num_contacts = 0;
   NALU_HYPRE_BigInt range_start, range_end;
   NALU_HYPRE_Int storage;
   NALU_HYPRE_Int indx;
   NALU_HYPRE_BigInt row;
   NALU_HYPRE_Int num_ranges, row_count;
   NALU_HYPRE_Int num_recvs;
   NALU_HYPRE_Int counter;
   NALU_HYPRE_BigInt upper_bound;
   NALU_HYPRE_Int num_real_procs;

   NALU_HYPRE_BigInt *row_list = NULL;
   NALU_HYPRE_Int *a_proc_id = NULL, *orig_order = NULL;
   NALU_HYPRE_Int *real_proc_id = NULL, *us_real_proc_id = NULL;
   NALU_HYPRE_Int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   NALU_HYPRE_Int *recv_starts = NULL;
   NALU_HYPRE_BigInt *response_buf = NULL;
   NALU_HYPRE_Int *response_buf_starts = NULL;
   NALU_HYPRE_Int *num_rows_per_proc = NULL;
   NALU_HYPRE_Int  tmp_int;
   NALU_HYPRE_Int  obj_size_bytes, big_int_size, complex_size;
   NALU_HYPRE_Int  first_index;

   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;

   NALU_HYPRE_Complex tmp_complex;
   NALU_HYPRE_BigInt *ex_contact_buf = NULL;
   NALU_HYPRE_Complex *vector_data;
   NALU_HYPRE_Complex value;

   nalu_hypre_DataExchangeResponse      response_obj1, response_obj2;
   nalu_hypre_ProcListElements          send_proc_obj;

   MPI_Comm comm = nalu_hypre_IJVectorComm(vector);
   nalu_hypre_ParVector *par_vector = (nalu_hypre_ParVector*) nalu_hypre_IJVectorObject(vector);

   nalu_hypre_IJAssumedPart   *apart;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   global_num_rows = nalu_hypre_IJVectorGlobalNumRows(vector);
   global_first_row = nalu_hypre_IJVectorGlobalFirstRow(vector);

   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      NALU_HYPRE_BigInt  *off_proc_i_h    = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Complex *off_proc_data_h = nalu_hypre_TAlloc(NALU_HYPRE_Complex, current_num_elmts, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TMemcpy(off_proc_i_h,    off_proc_i,    NALU_HYPRE_BigInt,  current_num_elmts, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(off_proc_data_h, off_proc_data, NALU_HYPRE_Complex, current_num_elmts, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);

      off_proc_i    = off_proc_i_h;
      off_proc_data = off_proc_data_h;
   }

   /* call nalu_hypre_IJVectorAddToValuesParCSR directly inside this function
    * with one chunk of data */
   NALU_HYPRE_Int      off_proc_nelm_recv_cur = 0;
   NALU_HYPRE_Int      off_proc_nelm_recv_max = 0;
   NALU_HYPRE_BigInt  *off_proc_i_recv = NULL;
   NALU_HYPRE_Complex *off_proc_data_recv = NULL;
   NALU_HYPRE_BigInt  *off_proc_i_recv_d = NULL;
   NALU_HYPRE_Complex *off_proc_data_recv_d = NULL;

   /* verify that we have created the assumed partition */
   if  (nalu_hypre_IJVectorAssumedPart(vector) == NULL)
   {
      nalu_hypre_IJVectorCreateAssumedPartition(vector);
   }

   apart = (nalu_hypre_IJAssumedPart*) nalu_hypre_IJVectorAssumedPart(vector);

   /* get the assumed processor id for each row */
   a_proc_id = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);
   orig_order =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);
   real_proc_id = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);
   row_list =   nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);

   if (current_num_elmts > 0)
   {
      for (i = 0; i < current_num_elmts; i++)
      {
         row = off_proc_i[i];
         row_list[i] = row;
         nalu_hypre_GetAssumedPartitionProcFromRow(comm, row, global_first_row,
                                              global_num_rows, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/

      nalu_hypre_BigQsortb2i(row_list, a_proc_id, orig_order, 0, current_num_elmts - 1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i = 1; i < current_num_elmts; i++)
      {
         if (a_proc_id[i] > last_proc)
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }

   }

   /* now we will go through a create a contact list - need to contact
      assumed processors and find out who the actual row owner is - we
      will contact with a range (2 numbers) */

   ex_contact_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ex_num_contacts, NALU_HYPRE_MEMORY_HOST);
   ex_contact_vec_starts =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ex_num_contacts + 1, NALU_HYPRE_MEMORY_HOST);
   ex_contact_buf =  nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  ex_num_contacts * 2, NALU_HYPRE_MEMORY_HOST);

   counter = 0;
   range_end = -1;
   for (i = 0; i < current_num_elmts; i++)
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0) { ex_contact_buf[counter * 2 - 1] = row_list[i - 1]; }

         /*start new range*/
         ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter * 2;
         ex_contact_buf[counter * 2] =  row_list[i];
         counter++;

         nalu_hypre_GetAssumedPartitionRowRange(comm, proc_id, global_first_row,
                                           global_num_rows, &range_start, &range_end);
      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter * 2;
   /*finish the last range*/
   if (counter > 0)
   {
      ex_contact_buf[counter * 2 - 1] = row_list[current_num_elmts - 1];
   }

   /* create response object - can use same fill response as used in the commpkg
      routine */
   response_obj1.fill_response = nalu_hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/
   response_obj1.data2 = NULL;

   max_response_size = 6;  /* 6 means we can fit 3 ranges*/

   nalu_hypre_DataExchangeList(ex_num_contacts, ex_contact_procs,
                          ex_contact_buf, ex_contact_vec_starts, sizeof(NALU_HYPRE_BigInt),
                          sizeof(NALU_HYPRE_BigInt), &response_obj1, max_response_size, 4,
                          comm, (void**) &response_buf, &response_buf_starts);

   /* now response_buf contains a proc_id followed by an upper bound for the
      range.  */

   nalu_hypre_TFree(ex_contact_procs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ex_contact_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ex_contact_vec_starts, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(a_proc_id, NALU_HYPRE_MEMORY_HOST);
   a_proc_id = NULL;

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];
   num_ranges = num_ranges / 2;

   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;

   /* loop through ranges - create a list of actual processor ids*/
   for (i = 0; i < num_ranges; i++)
   {
      upper_bound = response_buf[i * 2 + 1];
      counter = 0;
      tmp_id = (NALU_HYPRE_Int)response_buf[i * 2];

      /* loop through row_list entries - counting how many are in the range */
      while (j < current_num_elmts && row_list[j] <= upper_bound)
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;
      }
      if (counter > 0 && tmp_id != prev_id)
      {
         num_real_procs++;
      }
      prev_id = tmp_id;
   }

   /* now we have the list of real procesors ids (real_proc_id) - and the number
      of distinct ones - so now we can set up data to be sent - we have
      NALU_HYPRE_Int and NALU_HYPRE_Complex data.  (row number and value) - we will send
      everything as a void since we may not know the rel sizes of ints and
      doubles */

   /* first find out how many elements to send per proc - so we can do
      storage */

   complex_size = sizeof(NALU_HYPRE_Complex);
   big_int_size = sizeof(NALU_HYPRE_BigInt);

   obj_size_bytes = nalu_hypre_max(big_int_size, complex_size);

   ex_contact_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_real_procs, NALU_HYPRE_MEMORY_HOST);
   num_rows_per_proc = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_real_procs, NALU_HYPRE_MEMORY_HOST);

   counter = 0;

   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;

      /* loop through real procs - these are sorted (row_list is sorted also)*/
      for (i = 1; i < current_num_elmts; i++)
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
         }
      }
   }

   /* calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_real_procs + 1, NALU_HYPRE_MEMORY_HOST);
   ex_contact_vec_starts[0] = -1;

   for (i = 0; i < num_real_procs; i++)
   {
      storage += 1 + 2 *  num_rows_per_proc[i];
      ex_contact_vec_starts[i + 1] = -storage - 1; /* need negative for next loop */
   }

   /*void_contact_buf = nalu_hypre_TAlloc(char, storage*obj_size_bytes);*/
   void_contact_buf = nalu_hypre_CTAlloc(char, storage * obj_size_bytes, NALU_HYPRE_MEMORY_HOST);
   index_ptr = void_contact_buf; /* step through with this index */

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf_d contains #rows, row #, data, etc. */

   /* un-sort real_proc_id  - we want to access data arrays in order */

   us_real_proc_id =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  current_num_elmts, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < current_num_elmts; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   nalu_hypre_TFree(real_proc_id, NALU_HYPRE_MEMORY_HOST);

   prev_id = -1;
   for (i = 0; i < current_num_elmts; i++)
   {
      proc_id = us_real_proc_id[i];
      /* can't use row list[i] - you loose the negative signs that differentiate
         add/set values */
      row = off_proc_i[i];
      /* find position of this processor */
      indx = nalu_hypre_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in =  ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in * obj_size_bytes);

      /* first time for this processor - add the number of rows to the buffer */
      if (in < 0)
      {
         in = -in - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in * obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         nalu_hypre_TMemcpy( index_ptr,  &tmp_int,  NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in++;
      }
      /* add row # */
      nalu_hypre_TMemcpy( index_ptr,  &row,  NALU_HYPRE_BigInt, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      /* add value */
      tmp_complex = off_proc_data[i];
      nalu_hypre_TMemcpy( index_ptr,  &tmp_complex, NALU_HYPRE_Complex, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      /* increment the indexes to keep track of where we are - fix later */
      ex_contact_vec_starts[indx] = in;
   }

   /* some clean up */

   nalu_hypre_TFree(response_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(response_buf_starts, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(us_real_proc_id, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(orig_order, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(row_list, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_rows_per_proc, NALU_HYPRE_MEMORY_HOST);

   for (i = num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i - 1];
   }

   ex_contact_vec_starts[0] = 0;

   /* now send the data */

   /***********************************/
   /* now get the info in send_proc_obj_d */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */

   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = NULL; /* don't care who sent it to us */
   send_proc_obj.vec_starts =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_proc_obj.storage_length + 1, NALU_HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements =
      nalu_hypre_TAlloc(char, obj_size_bytes * send_proc_obj.element_storage_length, NALU_HYPRE_MEMORY_HOST);

   response_obj2.fill_response = nalu_hypre_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   nalu_hypre_DataExchangeList(num_real_procs, ex_contact_procs,
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 5,
                          comm,  (void **) &response_buf, &response_buf_starts);

   /***********************************/

   nalu_hypre_TFree(response_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(response_buf_starts, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(ex_contact_procs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(void_contact_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ex_contact_vec_starts, NALU_HYPRE_MEMORY_HOST);

   /* Now we can unpack the send_proc_objects and either set or add to the
      vector data */

   num_recvs = send_proc_obj.length;

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;

   vector_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(par_vector));
   first_index =  nalu_hypre_ParVectorFirstIndex(par_vector);

   for (i = 0; i < num_recvs; i++)
   {
      indx = recv_starts[i];

      /* get the number of rows for  this recv */
      nalu_hypre_TMemcpy( &row_count,  recv_data_ptr, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;

      for (j = 0; j < row_count; j++) /* for each row: unpack info */
      {
         /* row # */
         nalu_hypre_TMemcpy( &row,  recv_data_ptr, NALU_HYPRE_BigInt, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* value */
         nalu_hypre_TMemcpy( &value,  recv_data_ptr, NALU_HYPRE_Complex, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         if (memory_location == NALU_HYPRE_MEMORY_HOST)
         {
            k = (NALU_HYPRE_Int)(row - first_index - global_first_row);
            vector_data[k] += value;
         }
         else
         {
            if (off_proc_nelm_recv_cur >= off_proc_nelm_recv_max)
            {
               off_proc_nelm_recv_max = 2 * (off_proc_nelm_recv_cur + 1);
               off_proc_i_recv    = nalu_hypre_TReAlloc(off_proc_i_recv,    NALU_HYPRE_BigInt,  off_proc_nelm_recv_max,
                                                   NALU_HYPRE_MEMORY_HOST);
               off_proc_data_recv = nalu_hypre_TReAlloc(off_proc_data_recv, NALU_HYPRE_Complex, off_proc_nelm_recv_max,
                                                   NALU_HYPRE_MEMORY_HOST);
            }
            off_proc_i_recv[off_proc_nelm_recv_cur] = row;
            off_proc_data_recv[off_proc_nelm_recv_cur] = value;
            off_proc_nelm_recv_cur ++;
         }
      }
   }

   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      off_proc_i_recv_d    = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  off_proc_nelm_recv_cur, NALU_HYPRE_MEMORY_DEVICE);
      off_proc_data_recv_d = nalu_hypre_TAlloc(NALU_HYPRE_Complex, off_proc_nelm_recv_cur, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TMemcpy(off_proc_i_recv_d,    off_proc_i_recv,    NALU_HYPRE_BigInt,  off_proc_nelm_recv_cur,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(off_proc_data_recv_d, off_proc_data_recv, NALU_HYPRE_Complex, off_proc_nelm_recv_cur,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_GPU)
      nalu_hypre_IJVectorSetAddValuesParDevice(vector, off_proc_nelm_recv_cur, off_proc_i_recv_d,
                                          off_proc_data_recv_d, "add");
#endif
   }

   nalu_hypre_TFree(send_proc_obj.v_elements, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_proc_obj.vec_starts, NALU_HYPRE_MEMORY_HOST);

   if (memory_location == NALU_HYPRE_MEMORY_DEVICE)
   {
      nalu_hypre_TFree(off_proc_i,    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(off_proc_data, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(off_proc_i_recv,    NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(off_proc_data_recv, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(off_proc_i_recv_d,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(off_proc_data_recv_d, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}
