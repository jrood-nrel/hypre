/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * nalu_hypre_IJMatrix interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"

#include "../NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

NALU_HYPRE_Int
nalu_hypre_IJMatrixGetRowPartitioning( NALU_HYPRE_IJMatrix matrix,
                                  NALU_HYPRE_BigInt **row_partitioning )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Variable ijmatrix is NULL -- nalu_hypre_IJMatrixGetRowPartitioning\n");
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixRowPartitioning(ijmatrix))
   {
      *row_partitioning = nalu_hypre_IJMatrixRowPartitioning(ijmatrix);
   }
   else
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

NALU_HYPRE_Int
nalu_hypre_IJMatrixGetColPartitioning( NALU_HYPRE_IJMatrix matrix,
                                  NALU_HYPRE_BigInt **col_partitioning )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Variable ijmatrix is NULL -- nalu_hypre_IJMatrixGetColPartitioning\n");
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixColPartitioning(ijmatrix))
   {
      *col_partitioning = nalu_hypre_IJMatrixColPartitioning(ijmatrix);
   }
   else
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJMatrixSetObject( NALU_HYPRE_IJMatrix  matrix,
                         void           *object )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nalu_hypre_IJMatrixObject(ijmatrix) != NULL)
   {
      /*nalu_hypre_printf("Referencing a new IJMatrix object can orphan an old -- ");
      nalu_hypre_printf("nalu_hypre_IJMatrixSetObject\n");*/
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJMatrixObject(ijmatrix) = object;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrixRead: Read from file, HYPRE's IJ format or MM format
 * create IJMatrix on host memory
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    NALU_HYPRE_Int       type,
                    NALU_HYPRE_IJMatrix *matrix_ptr,
                    NALU_HYPRE_Int       is_mm       /* if is a Matrix-Market file */)
{
   NALU_HYPRE_IJMatrix  matrix;
   NALU_HYPRE_BigInt    ilower, iupper, jlower, jupper;
   NALU_HYPRE_BigInt    I, J;
   NALU_HYPRE_Int       ncols;
   NALU_HYPRE_Complex   value;
   NALU_HYPRE_Int       myid, ret;
   NALU_HYPRE_Int       isSym = 0;
   char            new_filename[255];
   FILE           *file;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   if (is_mm)
   {
      nalu_hypre_sprintf(new_filename, "%s", filename);
   }
   else
   {
      nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   }

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (is_mm)
   {
      MM_typecode matcode;
      NALU_HYPRE_Int nrow, ncol, nnz;

      if (nalu_hypre_mm_read_banner(file, &matcode) != 0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Could not process Matrix Market banner.");
         return nalu_hypre_error_flag;
      }

      if (!nalu_hypre_mm_is_valid(matcode))
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Invalid Matrix Market file.");
         return nalu_hypre_error_flag;
      }

      if ( !( (nalu_hypre_mm_is_real(matcode) || nalu_hypre_mm_is_integer(matcode)) &&
              nalu_hypre_mm_is_coordinate(matcode) && nalu_hypre_mm_is_sparse(matcode) ) )
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "Only sparse real-valued/integer coordinate matrices are supported");
         return nalu_hypre_error_flag;
      }

      if (nalu_hypre_mm_is_symmetric(matcode))
      {
         isSym = 1;
      }

      if (nalu_hypre_mm_read_mtx_crd_size(file, &nrow, &ncol, &nnz) != 0)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "MM read size error !");
         return nalu_hypre_error_flag;
      }

      ilower = 0;
      iupper = ilower + nrow - 1;
      jlower = 0;
      jupper = jlower + ncol - 1;
   }
   else
   {
      nalu_hypre_fscanf(file, "%b %b %b %b", &ilower, &iupper, &jlower, &jupper);
   }

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);

   NALU_HYPRE_IJMatrixSetObjectType(matrix, type);

   NALU_HYPRE_IJMatrixInitialize_v2(matrix, NALU_HYPRE_MEMORY_HOST);

   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  See comments in IJVectorRead(). */
   ncols = 1;
   while ( (ret = nalu_hypre_fscanf(file, "%b %b%*[ \t]%le", &I, &J, &value)) != EOF )
   {
      if (ret != 3)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error in IJ matrix input file.");
         return nalu_hypre_error_flag;
      }

      if (is_mm)
      {
         I --;
         J --;
      }

      if (I < ilower || I > iupper)
      {
         NALU_HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &I, &J, &value);
      }
      else
      {
         NALU_HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
      }

      if (isSym && I != J)
      {
         if (J < ilower || J > iupper)
         {
            NALU_HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &J, &I, &value);
         }
         else
         {
            NALU_HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &J, &I, &value);
         }
      }
   }

   NALU_HYPRE_IJMatrixAssemble(matrix);

   fclose(file);

   *matrix_ptr = matrix;

   return nalu_hypre_error_flag;
}

