/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Functions for scanning and printing "box-dimensioned" data.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_PrintBoxArrayData
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PrintBoxArrayData( FILE            *file,
                         nalu_hypre_BoxArray  *box_array,
                         nalu_hypre_BoxArray  *data_space,
                         NALU_HYPRE_Int        num_values,
                         NALU_HYPRE_Int        dim,
                         NALU_HYPRE_Complex   *data       )
{
   nalu_hypre_Box       *box;
   nalu_hypre_Box       *data_box;

   NALU_HYPRE_Int        data_box_volume;

   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      stride;
   nalu_hypre_Index      index;

   NALU_HYPRE_Int        i, j, d;
   NALU_HYPRE_Complex    value;

   /* Print data from the host */
   nalu_hypre_SetIndex(stride, 1);
   nalu_hypre_ForBoxI(i, box_array)
   {
      box      = nalu_hypre_BoxArrayBox(box_array, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      start = nalu_hypre_BoxIMin(box);
      data_box_volume = nalu_hypre_BoxVolume(data_box);

      nalu_hypre_BoxGetSize(box, loop_size);

      nalu_hypre_SerialBoxLoop1Begin(dim, loop_size,
                                data_box, start, stride, datai);
      {
         /* Print lines of the form: "%d: (%d, %d, %d; %d) %.14e\n" */
         zypre_BoxLoopGetIndex(index);
         for (j = 0; j < num_values; j++)
         {
            nalu_hypre_fprintf(file, "%d: (%d",
                          i, nalu_hypre_IndexD(start, 0) + nalu_hypre_IndexD(index, 0));
            for (d = 1; d < dim; d++)
            {
               nalu_hypre_fprintf(file, ", %d",
                             nalu_hypre_IndexD(start, d) + nalu_hypre_IndexD(index, d));
            }
            value = data[datai + j * data_box_volume];
#ifdef NALU_HYPRE_COMPLEX
            nalu_hypre_fprintf(file, "; %d) %.14e , %.14e\n",
                          j, nalu_hypre_creal(value), nalu_hypre_cimag(value));
#else
            nalu_hypre_fprintf(file, "; %d) %.14e\n", j, value);
#endif
         }
      }
      nalu_hypre_SerialBoxLoop1End(datai);

      data += num_values * data_box_volume;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PrintCCVDBoxArrayData
 *
 * Note that the the stencil loop (j) is _outside_ the space index loop
 * (datai), unlike nalu_hypre_PrintBoxArrayData (there is no j loop in
 * nalu_hypre_PrintCCBoxArrayData)
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PrintCCVDBoxArrayData( FILE            *file,
                             nalu_hypre_BoxArray  *box_array,
                             nalu_hypre_BoxArray  *data_space,
                             NALU_HYPRE_Int        num_values,
                             NALU_HYPRE_Int        center_rank,
                             NALU_HYPRE_Int        stencil_size,
                             NALU_HYPRE_Int       *symm_elements,
                             NALU_HYPRE_Int        dim,
                             NALU_HYPRE_Complex   *data       )
{
   nalu_hypre_Box       *box;
   nalu_hypre_Box       *data_box;

   NALU_HYPRE_Int        data_box_volume;

   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      stride;
   nalu_hypre_Index      index;

   NALU_HYPRE_Int        i, j, d;
   NALU_HYPRE_Complex    value;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   nalu_hypre_SetIndex(stride, 1);

   /* First is the constant, off-diagonal, part of the matrix: */
   for (j = 0; j < stencil_size; j++)
   {
      if (symm_elements[j] < 0 && j != center_rank)
      {
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf( file, "*: (*, *, *; %d) %.14e , %.14e\n",
                        j, nalu_hypre_creal(data[0]), nalu_hypre_cimag(data[0]));
#else
         nalu_hypre_fprintf( file, "*: (*, *, *; %d) %.14e\n",
                        j, data[0] );
#endif
      }
      ++data;
   }

   /* Then each box has a variable, diagonal, part of the matrix: */
   nalu_hypre_ForBoxI(i, box_array)
   {
      box      = nalu_hypre_BoxArrayBox(box_array, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      start = nalu_hypre_BoxIMin(box);
      data_box_volume = nalu_hypre_BoxVolume(data_box);

      nalu_hypre_BoxGetSize(box, loop_size);

      nalu_hypre_SerialBoxLoop1Begin(dim, loop_size,
                                data_box, start, stride, datai);
      {
         /* Print line of the form: "%d: (%d, %d, %d; %d) %.14e\n" */
         zypre_BoxLoopGetIndex(index);
         nalu_hypre_fprintf(file, "%d: (%d",
                       i, nalu_hypre_IndexD(start, 0) + nalu_hypre_IndexD(index, 0));
         for (d = 1; d < dim; d++)
         {
            nalu_hypre_fprintf(file, ", %d",
                          nalu_hypre_IndexD(start, d) + nalu_hypre_IndexD(index, d));
         }
         value = data[datai];
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf(file, "; %d) %.14e , %.14e\n",
                       center_rank, nalu_hypre_creal(value), nalu_hypre_cimag(value));
#else
         nalu_hypre_fprintf(file, "; %d) %.14e\n", center_rank, value);
#endif
      }
      nalu_hypre_SerialBoxLoop1End(datai);
      data += data_box_volume;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PrintCCBoxArrayData
 *
 * same as nalu_hypre_PrintBoxArrayData but for constant coefficients
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PrintCCBoxArrayData( FILE            *file,
                           nalu_hypre_BoxArray  *box_array,
                           nalu_hypre_BoxArray  *data_space,
                           NALU_HYPRE_Int        num_values,
                           NALU_HYPRE_Complex   *data       )
{
   NALU_HYPRE_Int        datai;

   NALU_HYPRE_Int        i, j;
   NALU_HYPRE_Complex    value;

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   nalu_hypre_ForBoxI(i, box_array)
   {
      datai = nalu_hypre_CCBoxIndexRank_noargs();

      for (j = 0; j < num_values; j++)
      {
         value = data[datai + j];
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf(file, "*: (*, *, *; %d) %.14e , %.14e\n",
                       j, nalu_hypre_creal(value), nalu_hypre_cimag(value));
#else
         nalu_hypre_fprintf(file, "*: (*, *, *; %d) %.14e\n", j, value);
#endif
      }

      data += num_values;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ReadBoxArrayData  (for non-constant coefficients)
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ReadBoxArrayData( FILE            *file,
                        nalu_hypre_BoxArray  *box_array,
                        nalu_hypre_BoxArray  *data_space,
                        NALU_HYPRE_Int        num_values,
                        NALU_HYPRE_Int        dim,
                        NALU_HYPRE_Complex   *data       )
{
   nalu_hypre_Box       *box;
   nalu_hypre_Box       *data_box;

   NALU_HYPRE_Int        data_box_volume;

   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      stride;

   NALU_HYPRE_Int        i, j, d, idummy;

   /* Read data on the host */
   nalu_hypre_SetIndex(stride, 1);
   nalu_hypre_ForBoxI(i, box_array)
   {
      box      = nalu_hypre_BoxArrayBox(box_array, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      start = nalu_hypre_BoxIMin(box);
      data_box_volume = nalu_hypre_BoxVolume(data_box);

      nalu_hypre_BoxGetSize(box, loop_size);

      nalu_hypre_SerialBoxLoop1Begin(dim, loop_size,
                                data_box, start, stride, datai);
      {
         /* Read lines of the form: "%d: (%d, %d, %d; %d) %le\n" */
         for (j = 0; j < num_values; j++)
         {
            nalu_hypre_fscanf(file, "%d: (%d", &idummy, &idummy);
            for (d = 1; d < dim; d++)
            {
               nalu_hypre_fscanf(file, ", %d", &idummy);
            }
            nalu_hypre_fscanf(file, "; %d) %le\n",
                         &idummy, &data[datai + j * data_box_volume]);
         }
      }
      nalu_hypre_SerialBoxLoop1End(datai);

      data += num_values * data_box_volume;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ReadBoxArrayData_CC  (for when there are some constant coefficients)
 *
 * Note: data array is expected to live on the host memory.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ReadBoxArrayData_CC( FILE            *file,
                           nalu_hypre_BoxArray  *box_array,
                           nalu_hypre_BoxArray  *data_space,
                           NALU_HYPRE_Int        stencil_size,
                           NALU_HYPRE_Int        real_stencil_size,
                           NALU_HYPRE_Int        constant_coefficient,
                           NALU_HYPRE_Int        dim,
                           NALU_HYPRE_Complex   *data       )
{
   nalu_hypre_Box       *box;
   nalu_hypre_Box       *data_box;

   NALU_HYPRE_Int        data_box_volume, constant_stencil_size;

   nalu_hypre_Index      loop_size;
   nalu_hypre_IndexRef   start;
   nalu_hypre_Index      stride;

   NALU_HYPRE_Int        i, j, d, idummy;

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   if (constant_coefficient == 1) { constant_stencil_size = stencil_size; }
   if (constant_coefficient == 2) { constant_stencil_size = stencil_size - 1; }

   nalu_hypre_SetIndex(stride, 1);

   nalu_hypre_ForBoxI(i, box_array)
   {
      box      = nalu_hypre_BoxArrayBox(box_array, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      start = nalu_hypre_BoxIMin(box);
      data_box_volume = nalu_hypre_BoxVolume(data_box);

      nalu_hypre_BoxGetSize(box, loop_size);

      /* First entries will be the constant part of the matrix.
         There is one entry for each constant stencil element,
         excluding ones which are redundant due to symmetry.*/
      for (j = 0; j < constant_stencil_size; j++)
      {
         nalu_hypre_fscanf(file, "*: (*, *, *; %d) %le\n", &idummy, &data[j]);
      }

      /* Next entries, if any, will be for a variable diagonal: */
      data += real_stencil_size;

      if (constant_coefficient == 2)
      {
         nalu_hypre_SerialBoxLoop1Begin(dim, loop_size,
                                   data_box, start, stride, datai);
         {
            /* Read line of the form: "%d: (%d, %d, %d; %d) %.14e\n" */
            nalu_hypre_fscanf(file, "%d: (%d", &idummy, &idummy);
            for (d = 1; d < dim; d++)
            {
               nalu_hypre_fscanf(file, ", %d", &idummy);
            }
            nalu_hypre_fscanf(file, "; %d) %le\n", &idummy, &data[datai]);
         }
         nalu_hypre_SerialBoxLoop1End(datai);
         data += data_box_volume;
      }
   }

   return nalu_hypre_error_flag;
}
