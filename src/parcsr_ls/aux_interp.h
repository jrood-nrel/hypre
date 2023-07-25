/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void
nalu_hypre_ParCSRCommExtendA(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int newoff, NALU_HYPRE_Int *found,
                        NALU_HYPRE_Int *p_num_recvs, NALU_HYPRE_Int **p_recv_procs,
                        NALU_HYPRE_Int **p_recv_vec_starts, NALU_HYPRE_Int *p_num_sends,
                        NALU_HYPRE_Int **p_send_procs, NALU_HYPRE_Int **p_send_map_starts,
                        NALU_HYPRE_Int **p_send_map_elmts, NALU_HYPRE_Int **p_node_add);

NALU_HYPRE_Int alt_insert_new_nodes(nalu_hypre_ParCSRCommPkg *comm_pkg,
                               nalu_hypre_ParCSRCommPkg *extend_comm_pkg,
                               NALU_HYPRE_Int *IN_marker,
                               NALU_HYPRE_Int full_off_procNodes,
                               NALU_HYPRE_Int *OUT_marker);

NALU_HYPRE_Int nalu_hypre_ssort(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n);
NALU_HYPRE_Int index_of_minimum(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n);
void swap_int(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int a, NALU_HYPRE_Int b);
void initialize_vecs(NALU_HYPRE_Int diag_n, NALU_HYPRE_Int offd_n, NALU_HYPRE_Int *diag_ftc, NALU_HYPRE_Int *offd_ftc,
                     NALU_HYPRE_Int *diag_pm, NALU_HYPRE_Int *offd_pm, NALU_HYPRE_Int *tmp_CF);

NALU_HYPRE_Int exchange_interp_data(
   NALU_HYPRE_Int **CF_marker_offd,
   NALU_HYPRE_Int **dof_func_offd,
   nalu_hypre_CSRMatrix **A_ext,
   NALU_HYPRE_Int *full_off_procNodes,
   nalu_hypre_CSRMatrix **Sop,
   nalu_hypre_ParCSRCommPkg **extend_comm_pkg,
   nalu_hypre_ParCSRMatrix *A,
   NALU_HYPRE_Int *CF_marker,
   nalu_hypre_ParCSRMatrix *S,
   NALU_HYPRE_Int num_functions,
   NALU_HYPRE_Int *dof_func,
   NALU_HYPRE_Int skip_fine_or_same_sign);

void build_interp_colmap(nalu_hypre_ParCSRMatrix *P, NALU_HYPRE_Int full_off_procNodes,
                         NALU_HYPRE_Int *tmp_CF_marker_offd, NALU_HYPRE_Int *fine_to_coarse_offd);

#ifdef __cplusplus
}
#endif
