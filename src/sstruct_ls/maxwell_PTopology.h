/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

typedef struct
{
   nalu_hypre_IJMatrix    *Face_iedge;
   nalu_hypre_IJMatrix    *Element_iedge;
   nalu_hypre_IJMatrix    *Edge_iedge;

   nalu_hypre_IJMatrix    *Element_Face;
   nalu_hypre_IJMatrix    *Element_Edge;

} nalu_hypre_PTopology;

