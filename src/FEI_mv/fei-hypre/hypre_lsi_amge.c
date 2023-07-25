/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LSI_AMGE interface
 *
 *****************************************************************************/

#ifdef HAVE_AMGE

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "seq_ls/amge/AMGe_matrix_topology.h"
#include "seq_mv/csr_matrix.h"

extern int nalu_hypre_AMGeMatrixTopologySetup(nalu_hypre_AMGeMatrixTopology ***A,
                 int *level, int *i_element_node_0, int *j_element_node_0,
                 int num_elements, int num_nodes, int Max_level);
extern int nalu_hypre_AMGeCoarsenodeSetup(nalu_hypre_AMGeMatrixTopology **A, int *level,
                 int **i_node_neighbor_coarsenode, int **j_node_neighbor_coarsenode,
                 int **i_node_coarsenode, int **j_node_coarsenode,
                 int **i_block_node, int **j_block_node, int *Num_blocks,
                 int *Num_elements, int *Num_nodes);

/* ********************************************************************* */
/* local variables to this module                                        */
/* ********************************************************************* */

int    rowLeng=0;
int    *i_element_node_0;
int    *j_element_node_0;
int    num_nodes, num_elements;
int    *i_dof_on_boundary;
int    system_size=1, num_dofs;
int    element_count=0;
int    temp_elemat_cnt;
int    **temp_elem_node, *temp_elem_node_cnt;
double **temp_elem_data;

/* ********************************************************************* */
/* constructor                                                           */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeCreate()
{
   printf("LSI_AMGe constructor\n");
   i_element_node_0   = NULL;
   j_element_node_0   = NULL;
   num_nodes          = 0;
   num_elements       = 0;
   system_size        = 1;
   num_dofs           = 0;
   element_count      = 0;
   temp_elemat_cnt    = 0;
   temp_elem_node     = NULL;
   temp_elem_node_cnt = NULL;
   temp_elem_data     = NULL;
   i_dof_on_boundary  = NULL;
   return 0;
}

/* ********************************************************************* */
/* destructor                                                            */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeDestroy()
{
   int i;

   printf("LSI_AMGe destructor\n");
   nalu_hypre_TFree(i_element_node_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_element_node_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_dof_on_boundary, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(temp_elem_node_cnt, NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < num_elements; i++ )
   {
      nalu_hypre_TFree(temp_elem_node[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(temp_elem_data[i], NALU_HYPRE_MEMORY_HOST);
   }
   temp_elem_node     = NULL;
   temp_elem_node_cnt = NULL;
   temp_elem_data     = NULL;
   return 0;
}

/* ********************************************************************* */
/* set the number of nodes in the finest grid                            */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeSetNNodes(int nNodes)
{
   int i;

   printf("LSI_AMGe NNodes = %d\n", nNodes);
   num_nodes = nNodes;
   return 0;
}

/* ********************************************************************* */
/* set the number of elements in the finest grid                         */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeSetNElements(int nElems)
{
   int i, nbytes;

   printf("LSI_AMGe NElements = %d\n", nElems);
   num_elements = nElems;
   nbytes = num_elements * sizeof(double*);
   temp_elem_data = nalu_hypre_TAlloc( nbytes ,NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < num_elements; i++ ) temp_elem_data[i] = NULL;
   nbytes = num_elements * sizeof(int*);
   temp_elem_node = nalu_hypre_TAlloc( nbytes ,NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < num_elements; i++ ) temp_elem_node[i] = NULL;
   nbytes = num_elements * sizeof(int);
   temp_elem_node_cnt = nalu_hypre_TAlloc( nbytes ,NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/* ********************************************************************* */
/* set system size                                                       */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeSetSystemSize(int size)
{
   printf("LSI_AMGe SystemSize = %d\n", size);
   system_size = size;
   return 0;
}

/* ********************************************************************* */
/* set boundary condition                                                */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeSetBoundary(int size, int *list)
{
   int i;

   printf("LSI_AMGe SetBoundary = %d\n", size);

   if ( i_dof_on_boundary == NULL )
      i_dof_on_boundary = nalu_hypre_TAlloc(int, num_nodes * system_size , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < num_nodes*system_size; i++ ) i_dof_on_boundary[i] = -1;

   for ( i = 0; i < size; i++ )
   {
      if (list[i] >= 0 && list[i] < num_nodes*system_size)
         i_dof_on_boundary[list[i]] = 0;
      else printf("AMGeSetBoundary ERROR : %d(%d)\n", list[i],num_nodes*system_size);
   }
   return 0;
}

/* ********************************************************************* */
/* load a row into this module                                           */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGePutRow(int row, int length, const double *colVal,
                          const int *colInd)
{
   int i, nbytes;

   if ( rowLeng == 0 )
   {
      if ( element_count % 100 == 0 )
         printf("LSI_AMGe PutRow %d\n", element_count);
      if ( element_count < 0 || element_count >= num_elements )
         printf("ERROR : element count too large %d\n",element_count);

      temp_elem_node_cnt[element_count] = length / system_size;
      nbytes = length / system_size * sizeof(int);
      temp_elem_node[element_count] = nalu_hypre_TAlloc( nbytes ,NALU_HYPRE_MEMORY_HOST);
      for ( i = 0; i < length; i+=system_size )
         temp_elem_node[element_count][i/system_size] = (colInd[i]-1)/system_size;
      nbytes = length * length * sizeof(double);
      temp_elem_data[element_count] = nalu_hypre_TAlloc(nbytes,NALU_HYPRE_MEMORY_HOST);
      temp_elemat_cnt = 0;
      rowLeng = length;
   }
   for ( i = 0; i < length; i++ )
      temp_elem_data[element_count][temp_elemat_cnt++] = colVal[i];
   if ( temp_elemat_cnt == rowLeng * rowLeng )
   {
      element_count++;
      rowLeng = 0;
   }
   return 0;
}

/* ********************************************************************* */
/* Solve                                                                 */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeSolve(double *rhs, double *x)
{
   int    i, j, l, counter, ierr, total_length;
   int    *Num_nodes, *Num_elements, *Num_dofs, level;
   int    max_level, Max_level;
   int    multiplier;

   /* coarsenode information and coarsenode neighborhood information */

   int **i_node_coarsenode, **j_node_coarsenode;
   int **i_node_neighbor_coarsenode, **j_node_neighbor_coarsenode;

   /* PDEsystem information: --------------------------------------- */

   int *i_dof_node_0, *j_dof_node_0;
   int *i_node_dof_0, *j_node_dof_0;

   int *i_element_dof_0, *j_element_dof_0;
   double *element_data;

   int **i_node_dof, **j_node_dof;

   /* Dirichlet boundary conditions information: ------------------- */

   /* int *i_dof_on_boundary; */

   /* nested dissection blocks: ------------------------------------ */

   int **i_block_node, **j_block_node;
   int *Num_blocks;

   /* nested dissection ILU(1) smoother: --------------------------- */
   /* internal format: --------------------------------------------- */

   int **i_ILUdof_to_dof;
   int **i_ILUdof_ILUdof_t, **j_ILUdof_ILUdof_t,
       **i_ILUdof_ILUdof, **j_ILUdof_ILUdof;
   double **LD_data, **U_data;

   /* -------------------------------------------------------------- */
   /*  PCG & V_cycle arrays:                                         */
   /* -------------------------------------------------------------- */

   double *r, *v, **w, **d, *aux, *v_coarse, *w_coarse;
   double *d_coarse, *v_fine, *w_fine, *d_fine;
   int max_iter = 1000;
   int coarse_level;
   int nu = 1;  /* not used ---------------------------------------- */

   double reduction_factor;

   /* Interpolation P and stiffness matrices Matrix; --------------- */

   nalu_hypre_CSRMatrix     **P;
   nalu_hypre_CSRMatrix     **Matrix;
   nalu_hypre_AMGeMatrixTopology **A;

   /* element matrices information: -------------------------------- */

   int *i_element_chord_0, *j_element_chord_0;
   double *a_element_chord_0;
   int *i_chord_dof_0, *j_chord_dof_0;
   int *Num_chords;

   /* auxiliary arrays for enforcing Dirichlet boundary conditions:  */

   int *i_dof_dof_a, *j_dof_dof_a;
   double *a_dof_dof;

   /* ===============================================================*/
   /* set num_nodes, num_elements                                    */
   /* fill up element_data                                           */
   /* fill up i_element_node_0 and j_element_node_0                  */
   /* fill up i_dof_on_boundary (0 - boundary, 1 - otherwise)        */
   /* ===============================================================*/

   num_elements = element_count;
   if ( num_nodes == 0 || num_elements == 0 )
   {
      printf("NALU_HYPRE_LSI_AMGe ERROR : num_nodes or num_elements not set.\n");
      exit(1);
   }
   total_length = 0;
   for ( i = 0; i < num_elements; i++ )
   {
      multiplier = temp_elem_node_cnt[i] * system_size;
      total_length += (multiplier * multiplier);
   }
   element_data = nalu_hypre_TAlloc(double, total_length , NALU_HYPRE_MEMORY_HOST);
   counter = 0;
   for ( i = 0; i < num_elements; i++ )
   {
      multiplier = temp_elem_node_cnt[i] * system_size;
      multiplier *= multiplier;
      for ( j = 0; j < multiplier; j++ )
         element_data[counter++] = temp_elem_data[i][j];
      nalu_hypre_TFree(temp_elem_data[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(temp_elem_data, NALU_HYPRE_MEMORY_HOST);
   temp_elem_data = NULL;

   total_length = 0;
   for (i = 0; i < num_elements; i++) total_length += temp_elem_node_cnt[i];
   i_element_node_0 = nalu_hypre_TAlloc(int, (num_elements + 1) , NALU_HYPRE_MEMORY_HOST);
   j_element_node_0 = nalu_hypre_TAlloc(int, total_length , NALU_HYPRE_MEMORY_HOST);
   counter = 0;
   for (i = 0; i < num_elements; i++)
   {
      i_element_node_0[i] = counter;
      for (j = 0; j < temp_elem_node_cnt[i]; j++)
         j_element_node_0[counter++] = temp_elem_node[i][j];
      nalu_hypre_TFree(temp_elem_node[i], NALU_HYPRE_MEMORY_HOST);
   }
   i_element_node_0[num_elements] = counter;
   nalu_hypre_TFree(temp_elem_node, NALU_HYPRE_MEMORY_HOST);
   temp_elem_node = NULL;

   /* -------------------------------------------------------------- */
   /* initialization                                                 */
   /* -------------------------------------------------------------- */

   Max_level    = 25;
   Num_chords   = nalu_hypre_CTAlloc(int,  Max_level, NALU_HYPRE_MEMORY_HOST);
   Num_elements = nalu_hypre_CTAlloc(int,  Max_level, NALU_HYPRE_MEMORY_HOST);
   Num_nodes    = nalu_hypre_CTAlloc(int,  Max_level, NALU_HYPRE_MEMORY_HOST);
   Num_dofs     = nalu_hypre_CTAlloc(int,  Max_level, NALU_HYPRE_MEMORY_HOST);
   Num_blocks   = nalu_hypre_CTAlloc(int,  Max_level, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < Max_level; i++)
   {
      Num_dofs[i] = 0;
      Num_elements[i] = 0;
   }

   Num_nodes[0] = num_nodes;
   Num_elements[0] = num_elements;

   /* -------------------------------------------------------------- */
   /* set up matrix topology for the fine matrix                     */
   /* input : i_element_node_0, j_element_node_0, num_elements,      */
   /*         num_nodes, Max_level                                   */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up topology \n");
   ierr = nalu_hypre_AMGeMatrixTopologySetup(&A, &level, i_element_node_0,
                j_element_node_0, num_elements, num_nodes, Max_level);

   max_level = level;

   /* -------------------------------------------------------------- */
   /* set up matrix topology for the coarse grids                    */
   /* input : A, Num_elements[0], Num_nodes[0]                       */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up coarse grids \n");
   ierr = nalu_hypre_AMGeCoarsenodeSetup(A, &level, &i_node_neighbor_coarsenode,
                &j_node_neighbor_coarsenode, &i_node_coarsenode,
                &j_node_coarsenode, &i_block_node, &j_block_node,
                Num_blocks, Num_elements, Num_nodes);

   /* -------------------------------------------------------------- */
   /* set up dof arrays based on system size                         */
   /* output : i_dof_node_0, j_dof_node_0, num_dofs                  */
   /* -------------------------------------------------------------- */

   ierr = compute_dof_node(&i_dof_node_0, &j_dof_node_0,
                           Num_nodes[0], system_size, &num_dofs);

   Num_dofs[0] = num_dofs;

   /*
   if (system_size == 1) i_dof_on_boundary = i_node_on_boundary;
   else
   {
      ierr = compute_dof_on_boundary(&i_dof_on_boundary, i_node_on_boundary,
                                     Num_nodes[0], system_size);
      nalu_hypre_TFree(i_node_on_boundary, NALU_HYPRE_MEMORY_HOST);
      i_node_on_boundary = NULL;
   }
   */

   /* -------------------------------------------------------------- */
   /* get element_dof information                                    */
   /* -------------------------------------------------------------- */

   ierr = transpose_matrix_create(&i_node_dof_0, &j_node_dof_0,
                   i_dof_node_0, j_dof_node_0, Num_dofs[0], Num_nodes[0]);

   if (system_size == 1)
   {
      i_element_dof_0 = i_element_node_0;
      j_element_dof_0 = j_element_node_0;
   }
   else
      ierr = matrix_matrix_product(&i_element_dof_0, &j_element_dof_0,
                i_element_node_0,j_element_node_0,i_node_dof_0,j_node_dof_0,
                Num_elements[0], Num_nodes[0], Num_dofs[0]);

   /* -------------------------------------------------------------- */
   /* store element matrices in element_chord format                 */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up element dof relations \n");
   ierr = nalu_hypre_AMGeElementMatrixDof(i_element_dof_0, j_element_dof_0,
                element_data, &i_element_chord_0, &j_element_chord_0,
                &a_element_chord_0, &i_chord_dof_0, &j_chord_dof_0,
                &Num_chords[0], Num_elements[0], Num_dofs[0]);

   printf("LSI_AMGe Solve : Setting up interpolation \n");
   ierr = nalu_hypre_AMGeInterpolationSetup(&P, &Matrix, A, &level,
                /* ------ fine-grid element matrices ----- */
                i_element_chord_0, j_element_chord_0, a_element_chord_0,
                i_chord_dof_0, j_chord_dof_0,

                /* nnz: of the assembled matrices -------*/
                Num_chords,

                /* ----- coarse node information  ------ */
                i_node_neighbor_coarsenode, j_node_neighbor_coarsenode,
                i_node_coarsenode, j_node_coarsenode,

                /* --------- Dirichlet b.c. ----------- */
                i_dof_on_boundary,

                /* -------- PDEsystem information -------- */
                system_size, i_dof_node_0, j_dof_node_0,
                i_node_dof_0, j_node_dof_0, &i_node_dof, &j_node_dof,

                Num_elements, Num_nodes, Num_dofs);

   nalu_hypre_TFree(i_dof_on_boundary, NALU_HYPRE_MEMORY_HOST);
   i_dof_on_boundary = NULL;
   nalu_hypre_TFree(i_dof_node_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_dof_node_0, NALU_HYPRE_MEMORY_HOST);

   printf("LSI_AMGe Solve : Setting up smoother \n");
   ierr = nalu_hypre_AMGeSmootherSetup(&i_ILUdof_to_dof, &i_ILUdof_ILUdof,
                &j_ILUdof_ILUdof, &LD_data, &i_ILUdof_ILUdof_t,
                &j_ILUdof_ILUdof_t, &U_data, Matrix, &level,
                i_block_node, j_block_node, i_node_dof, j_node_dof,
                Num_blocks, Num_nodes, Num_dofs);

   nalu_hypre_TFree(i_node_dof_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_dof_0, NALU_HYPRE_MEMORY_HOST);

   for (l=0; l < level+1; l++)
   {
      nalu_hypre_TFree(i_block_node[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_block_node[l], NALU_HYPRE_MEMORY_HOST);
   }

   for (l=1; l < level+1; l++)
   {
      nalu_hypre_TFree(i_node_dof[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_node_dof[l], NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(i_node_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_block_node, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_block_node, NALU_HYPRE_MEMORY_HOST);

   /* ===================================================================== */
   /* =================== S O L U T I O N   P A R T: ====================== */
   /* ===================================================================== */

   /* one V(1,1) --cycle as preconditioner in PCG: ======================== */
   /* ILU solve pre--smoothing, ILU solve post--smoothing; ================ */

   w = nalu_hypre_CTAlloc(double*,  level+1, NALU_HYPRE_MEMORY_HOST);
   d = nalu_hypre_CTAlloc(double*,  level+1, NALU_HYPRE_MEMORY_HOST);

   for (l=0; l < level+1; l++)
   {
      Num_dofs[l] = Num_nodes[l] * system_size;
      if (Num_dofs[l] > 0)
      {
	  w[l] = nalu_hypre_CTAlloc(double,  Num_dofs[l], NALU_HYPRE_MEMORY_HOST);
	  d[l] = nalu_hypre_CTAlloc(double,  Num_dofs[l], NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
	  level = l-1;
	  break;
      }
   }

   num_dofs = Num_dofs[0];

   /*x = nalu_hypre_CTAlloc(double, num_dofs);  */
   /*rhs = nalu_hypre_CTAlloc(double, num_dofs);*/

   r = nalu_hypre_CTAlloc(double,  num_dofs, NALU_HYPRE_MEMORY_HOST);
   aux = nalu_hypre_CTAlloc(double,  num_dofs, NALU_HYPRE_MEMORY_HOST);
   v_fine = nalu_hypre_CTAlloc(double,  num_dofs, NALU_HYPRE_MEMORY_HOST);
   w_fine = nalu_hypre_CTAlloc(double,  num_dofs, NALU_HYPRE_MEMORY_HOST);
   d_fine = nalu_hypre_CTAlloc(double,  num_dofs, NALU_HYPRE_MEMORY_HOST);

   coarse_level = level;
   v_coarse = nalu_hypre_CTAlloc(double,  Num_dofs[coarse_level], NALU_HYPRE_MEMORY_HOST);
   w_coarse = nalu_hypre_CTAlloc(double,  Num_dofs[coarse_level], NALU_HYPRE_MEMORY_HOST);
   d_coarse = nalu_hypre_CTAlloc(double,  Num_dofs[coarse_level], NALU_HYPRE_MEMORY_HOST);

   for (l=0; l < level; l++)
   {
      printf("\n\n=======================================================\n");
      printf("             Testing level[%d] PCG solve:                  \n",l);
      printf("===========================================================\n");

      for (i=0; i < Num_dofs[l]; i++) x[i] = 0.e0;

      /* for (i=0; i < Num_dofs[l]; i++) rhs[i] = rand(); */

      i_dof_dof_a = nalu_hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof_a = nalu_hypre_CSRMatrixJ(Matrix[l]);
      a_dof_dof   = nalu_hypre_CSRMatrixData(Matrix[l]);

      ierr = nalu_hypre_ILUsolve(x, i_ILUdof_to_dof[l], i_ILUdof_ILUdof[l],
	           j_ILUdof_ILUdof[l], LD_data[l], i_ILUdof_ILUdof_t[l],
                   j_ILUdof_ILUdof_t[l], U_data[l], rhs, Num_dofs[l]);

      ierr = nalu_hypre_ILUpcg(x, rhs, a_dof_dof, i_dof_dof_a, j_dof_dof_a,
                   i_ILUdof_to_dof[l], i_ILUdof_ILUdof[l], j_ILUdof_ILUdof[l],
                   LD_data[l], i_ILUdof_ILUdof_t[l], j_ILUdof_ILUdof_t[l],
                   U_data[l], v_fine, w_fine, d_fine, max_iter, Num_dofs[l]);

      printf("\n\n=======================================================\n");
      printf("             END test PCG solve:                           \n");
      printf("===========================================================\n");

   }

   printf("\n\n===============================================================\n");
   printf(" ------- V_cycle & nested dissection ILU(1) smoothing: --------\n");
   printf("================================================================\n");

   num_dofs = Num_dofs[0];

   /* for (i=0; i < num_dofs; i++) rhs[i] = rand(); */

   ierr = nalu_hypre_VcycleILUpcg(x, rhs, w, d, &reduction_factor, Matrix,
                i_ILUdof_to_dof, i_ILUdof_ILUdof, j_ILUdof_ILUdof, LD_data,
                i_ILUdof_ILUdof_t, j_ILUdof_ILUdof_t, U_data, P, aux, r,
                v_fine, w_fine, d_fine, max_iter, v_coarse, w_coarse, d_coarse,
                nu, level, coarse_level, Num_dofs);

   /* nalu_hypre_TFree(x);   */
   /* nalu_hypre_TFree(rhs); */

   nalu_hypre_TFree(r, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(aux, NALU_HYPRE_MEMORY_HOST);

   for (l=0; l < level+1; l++)
      if (Num_dofs[l] > 0)
      {
 nalu_hypre_TFree(w[l], NALU_HYPRE_MEMORY_HOST);
 nalu_hypre_TFree(d[l], NALU_HYPRE_MEMORY_HOST);
	nalu_hypre_CSRMatrixDestroy(Matrix[l]);
      }

   for (l=0; l < max_level; l++)
   {
      nalu_hypre_TFree(i_node_coarsenode[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_node_coarsenode[l], NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(i_node_neighbor_coarsenode[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_node_neighbor_coarsenode[l], NALU_HYPRE_MEMORY_HOST);

      if (system_size == 1 &&Num_dofs[l+1] > 0)
      {
	  nalu_hypre_CSRMatrixI(P[l]) = NULL;
	  nalu_hypre_CSRMatrixJ(P[l]) = NULL;
      }

   }
   for (l=0; l < level; l++)
   {
      nalu_hypre_TFree(i_ILUdof_to_dof[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_ILUdof_ILUdof[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_ILUdof_ILUdof[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(LD_data[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(i_ILUdof_ILUdof_t[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(j_ILUdof_ILUdof_t[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(U_data[l], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixDestroy(P[l]);

   }

   nalu_hypre_TFree(v_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(d_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(d, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(v_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(d_coarse, NALU_HYPRE_MEMORY_HOST);

   for (l=0; l < max_level+1; l++)
      nalu_hypre_DestroyAMGeMatrixTopology(A[l]);

   nalu_hypre_TFree(Num_nodes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Num_elements, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Num_dofs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Num_blocks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Num_chords, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_chord_dof_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_chord_dof_0, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_element_chord_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_element_chord_0, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(a_element_chord_0, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(P, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Matrix, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(A, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_ILUdof_to_dof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(i_ILUdof_ILUdof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_ILUdof_ILUdof, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(LD_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_ILUdof_ILUdof_t, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_ILUdof_ILUdof_t, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(U_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_node_coarsenode, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_coarsenode, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(i_node_neighbor_coarsenode, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(j_node_neighbor_coarsenode, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(element_data, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/* ********************************************************************* */
/* local variables to this module                                        */
/* ********************************************************************* */

int NALU_HYPRE_LSI_AMGeWriteToFile()
{
   int  i, j, k, length;
   FILE *fp;

   fp = fopen("elem_mat", "w");

   for ( i = 0; i < element_count; i++ )
   {
      length = temp_elem_node_cnt[i] * system_size;
      for ( j = 0; j < length; j++ )
      {
         for ( k = 0; k < length; k++ )
            fprintf(fp, "%13.6e ", temp_elem_data[i][j*length+k]);
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
   }
   fclose(fp);

   fp = fopen("elem_node", "w");

   fprintf(fp, "%d %d\n", element_count, num_nodes);
   for (i = 0; i < element_count; i++)
   {
      for (j = 0; j < temp_elem_node_cnt[i]; j++)
         fprintf(fp, "%d ", temp_elem_node[i][j]+1);
      fprintf(fp,"\n");
   }

   fclose(fp);

   fp = fopen("node_bc", "w");

   for (i = 0; i < num_nodes*system_size; i++)
   {
      fprintf(fp, "%d\n", i_dof_on_boundary[i]);
   }
   fclose(fp);

   return 0;
}

#else

/* this is used only to eliminate compiler warnings */
int nalu_hypre_empty4;

#endif

