!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!-----------------------------------------------------------------------
! Test driver for unstructured matrix interface (structured storage)
!-----------------------------------------------------------------------
 
!-----------------------------------------------------------------------
! Standard 7-point laplacian in 3D with grid and anisotropy determined
! as user settings.
!-----------------------------------------------------------------------

      program test

      implicit none

      include 'mpif.h'

      integer MAXZONS, MAXBLKS, MAXDIM, MAXLEVELS
      integer NALU_HYPRE_PARCSR

      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      parameter (MAXDIM=3)
      parameter (MAXLEVELS=25)
      parameter (NALU_HYPRE_PARCSR=5555)

      integer             num_procs, myid

      integer             dim
      integer             nx, ny, nz
      integer             Px, Py, Pz
      integer             bx, by, bz
      double precision    cx, cy, cz
      integer             n_pre, n_post
      integer             solver_id
      integer             precond_id

      integer             setup_type
      integer             debug_flag, ioutdat, k_dim
      integer             nlevels

      integer             zero, one
      parameter           (zero = 0, one = 1)
      integer             maxiter, num_iterations
      integer             generate_matrix, generate_rhs
      character           matfile(32), vecfile(32)
      character*31        matfile_str, vecfile_str

      double precision    tol, pc_tol, convtol
      parameter           (pc_tol = 0.0)
      double precision    final_res_norm
                     
! parameters for BoomerAMG
      integer             hybrid, coarsen_type, measure_type
      integer             cycle_type
      integer             smooth_num_sweep
      integer*8           num_grid_sweeps
      integer*8           num_grid_sweeps2(4)
      integer*8           grid_relax_type
      integer*8           grid_relax_points
      integer*8           relax_weights
      double precision    strong_threshold, trunc_factor, drop_tol
      double precision    max_row_sum
      data                max_row_sum /1.0/

! parameters for ParaSails
      double precision    sai_threshold
      double precision    sai_filter

      integer*8           A, A_storage
      integer*8           b, b_storage
      integer*8           x, x_storage

      integer*8           solver
      integer*8           precond
      integer*8           precond_gotten
      integer*8           row_starts

      double precision    values(4)

      integer             p, q, r

      integer             ierr

      integer             i
      integer             first_local_row, last_local_row
      integer             first_local_col, last_local_col
      integer             indices(MAXZONS)
      double precision    vals(MAXZONS)

      integer             dof_func(1000), j

!-----------------------------------------------------------------------
!     Initialize MPI
!-----------------------------------------------------------------------

      call MPI_INIT(ierr)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)

!-----------------------------------------------------------------------
!     Set defaults
!-----------------------------------------------------------------------

      dim = 3

      nx = 10
      ny = 10
      nz = 10

      Px  = num_procs
      Py  = 1
      Pz  = 1

      bx = 1
      by = 1
      bz = 1

      cx = 1.0
      cy = 1.0
      cz = 1.0

      n_pre  = 1
      n_post = 1

      solver_id = 3

!-----------------------------------------------------------------------
!     Read options
!-----------------------------------------------------------------------
 
!     open( 5, file='parcsr_linear_solver.in', status='old')
!
!     read( 5, *) dim
!
!     read( 5, *) nx
!     read( 5, *) ny
!     read( 5, *) nz
!
!     read( 5, *) Px
!     read( 5, *) Py
!     read( 5, *) Pz
!
!     read( 5, *) bx
!     read( 5, *) by
!     read( 5, *) bz
!
!     read( 5, *) cx
!     read( 5, *) cy
!     read( 5, *) cz
!
!     read( 5, *) n_pre
!     read( 5, *) n_post
!
!     write(6,*) 'Generate matrix? !0 yes, 0 no (from file)'
      read(5,*) generate_matrix

      if (generate_matrix .eq. 0) then
!       write(6,*) 'What file to use for matrix (<= 31 chars)?'
        read(5,*) matfile_str
        i = 1
  100   if (matfile_str(i:i) .ne. ' ') then
          matfile(i) = matfile_str(i:i)
        else
          goto 200
        endif
        i = i + 1
        goto 100
  200   matfile(i) = char(0)
      endif

!     write(6,*) 'Generate right-hand side? !0 yes, 0 no (from file)'
      read(5,*) generate_rhs

      if (generate_rhs .eq. 0) then
!       write(6,*)
!    &    'What file to use for right-hand side (<= 31 chars)?'
        read(5,*) vecfile_str
        i = 1
  300   if (vecfile_str(i:i) .ne. ' ') then
          vecfile(i) = vecfile_str(i:i)
        else
          goto 400
        endif
        i = i + 1
        goto 300
  400   vecfile(i) = char(0)
      endif

!     write(6,*) 'What solver_id?'
!     write(6,*) '0 AMG, 1 AMG-PCG, 2 DS-PCG, 3 AMG-GMRES, 4 DS-GMRES,'
!     write(6,*) '5 AMG-CGNR, 6 DS-CGNR, 7 PILUT-GMRES, 8 ParaSails-GMRES,'
!     write(6,*) '9 AMG-BiCGSTAB, 10 DS-BiCGSTAB'
      read(5,*) solver_id

      if (solver_id .eq. 7) then
!       write(6,*) 'What drop tolerance?  <0 do not drop'
        read(5,*) drop_tol
      endif
 
!     write(6,*) 'What relative residual norm tolerance?'
      read(5,*) tol

!     close( 5 )

!-----------------------------------------------------------------------
!     Check a few things
!-----------------------------------------------------------------------

      if ((Px*Py*Pz) .ne. num_procs) then
         print *, 'Error: Invalid number of processors or topology'
         stop
      endif

      if ((dim .lt. 1) .or. (dim .gt. 3)) then
         print *, 'Error: Invalid problem dimension'
         stop
      endif

      if ((nx*ny*nz) .gt. MAXZONS) then
         print *, 'Error: Invalid number of zones'
         stop
      endif

      if ((bx*by*bz) .gt. MAXBLKS) then
         print *, 'Error: Invalid number of blocks'
         stop
      endif

!-----------------------------------------------------------------------
!     Print driver parameters
!-----------------------------------------------------------------------

      if (myid .eq. 0) then
         print *, 'Running with these driver parameters:'
         print *, '  (nx, ny, nz)    = (', nx, ',', ny, ',', nz, ')'
         print *, '  (Px, Py, Pz)    = (',  Px, ',',  Py, ',',  Pz, ')'
         print *, '  (bx, by, bz)    = (', bx, ',', by, ',', bz, ')'
         print *, '  (cx, cy, cz)    = (', cx, ',', cy, ',', cz, ')'
         print *, '  (n_pre, n_post) = (', n_pre, ',', n_post, ')'
         print *, '  dim             = ', dim
      endif

!-----------------------------------------------------------------------
!     Compute some grid and processor information
!-----------------------------------------------------------------------

      if (dim .eq. 1) then

!        compute p from Px and myid
         p = mod(myid,Px)

      elseif (dim .eq. 2) then

!        compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)

      elseif (dim .eq. 3) then

!        compute p,q,r from Px,Py,Pz and myid
         p = mod(myid,Px)
         q = mod((( myid - p)/Px),Py)
         r = (myid - (p + Px*q))/(Px*Py)

      endif

!----------------------------------------------------------------------
!     Set up the matrix
!-----------------------------------------------------------------------

      values(2) = -cx
      values(3) = -cy
      values(4) = -cz

      values(1) = 0.0
      if (nx .gt. 1) values(1) = values(1) + 2d0*cx
      if (ny .gt. 1) values(1) = values(1) + 2d0*cy
      if (nz .gt. 1) values(1) = values(1) + 2d0*cz

! Generate a Dirichlet Laplacian
      if (generate_matrix .eq. 0) then

        call NALU_HYPRE_IJMatrixRead(matfile, MPI_COMM_WORLD,
     &                          NALU_HYPRE_PARCSR, A, ierr)

        call NALU_HYPRE_IJMatrixGetObject(A, A_storage, ierr)

        call NALU_HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, ierr)

      else

        call NALU_HYPRE_GenerateLaplacian(MPI_COMM_WORLD, nx, ny, nz,
     &                               Px, Py, Pz, p, q, r, values,
     &                               A_storage, ierr)

        call NALU_HYPRE_ParCSRMatrixGetLocalRange(A_storage,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, ierr)

        call NALU_HYPRE_IJMatrixCreate(MPI_COMM_WORLD,
     &            first_local_row, last_local_row,
     &            first_local_col, last_local_col, A, ierr)

        call NALU_HYPRE_IJMatrixSetObject(A, A_storage, ierr)

        call NALU_HYPRE_IJMatrixSetObjectType(A, NALU_HYPRE_PARCSR, ierr)

      endif

      matfile(1)  = 'd'
      matfile(2)  = 'r'
      matfile(3)  = 'i'
      matfile(4)  = 'v'
      matfile(5)  = 'e'
      matfile(6)  = 'r'
      matfile(7)  = '.'
      matfile(8)  = 'o'
      matfile(9)  = 'u'
      matfile(10) = 't'
      matfile(11) = '.'
      matfile(12) = 'A'
      matfile(13) = char(0)
   
      call NALU_HYPRE_IJMatrixPrint(A, matfile, ierr)

      call hypre_ParCSRMatrixRowStarts(A_storage, row_starts, ierr)

!-----------------------------------------------------------------------
!     Set up the rhs and initial guess
!-----------------------------------------------------------------------

      if (generate_rhs .eq. 0) then

        call NALU_HYPRE_IJVectorRead(vecfile, MPI_COMM_WORLD,
     &                          NALU_HYPRE_PARCSR, b, ierr)

        call NALU_HYPRE_IJVectorGetObject(b, b_storage, ierr)

      else

        call NALU_HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &                            last_local_col, b, ierr)
        call NALU_HYPRE_IJVectorSetObjectType(b, NALU_HYPRE_PARCSR, ierr)
        call NALU_HYPRE_IJVectorInitialize(b, ierr)

! Set up a Dirichlet 0 problem
        do i = 1, last_local_col - first_local_col + 1
          indices(i) = first_local_col - 1 + i
          vals(i) = 1.
        enddo

        call NALU_HYPRE_IJVectorSetValues(b,
     &    last_local_col - first_local_col + 1, indices, vals, ierr)

        call NALU_HYPRE_IJVectorGetObject(b, b_storage, ierr)

        vecfile(1)  = 'd'
        vecfile(2)  = 'r'
        vecfile(3)  = 'i'
        vecfile(4)  = 'v'
        vecfile(5)  = 'e'
        vecfile(6)  = 'r'
        vecfile(7)  = '.'
        vecfile(8)  = 'o'
        vecfile(9)  = 'u'
        vecfile(10) = 't'
        vecfile(11) = '.'
        vecfile(12) = 'b'
        vecfile(13) = char(0)
   
        call NALU_HYPRE_IJVectorPrint(b, vecfile, ierr)

      endif

      call NALU_HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col,
     &                          last_local_col, x, ierr)
      call NALU_HYPRE_IJVectorSetObjectType(x, NALU_HYPRE_PARCSR, ierr)
      call NALU_HYPRE_IJVectorInitialize(x, ierr)
      do i = 1, last_local_col - first_local_col + 1
          indices(i) = first_local_col - 1 + i
          vals(i) = 0.
      enddo
      call NALU_HYPRE_IJVectorSetValues(x,
     &  last_local_col - first_local_col + 1, indices, vals, ierr)

! Choose a nonzero initial guess
      call NALU_HYPRE_IJVectorGetObject(x, x_storage, ierr)

      vecfile(1)  = 'd'
      vecfile(2)  = 'r'
      vecfile(3)  = 'i'
      vecfile(4)  = 'v'
      vecfile(5)  = 'e'
      vecfile(6)  = 'r'
      vecfile(7)  = '.'
      vecfile(8)  = 'o'
      vecfile(9)  = 'u'
      vecfile(10) = 't'
      vecfile(11) = '.'
      vecfile(12) = 'x'
      vecfile(13) = '0'
      vecfile(14) = char(0)
   
      call NALU_HYPRE_IJVectorPrint(x, vecfile, ierr)

!-----------------------------------------------------------------------
!     Solve the linear system
!-----------------------------------------------------------------------

!     General solver parameters, passing hard coded constants
!     will break the interface.

      maxiter = 100
      convtol = 0.9
      debug_flag = 0
      ioutdat = 1

      if (solver_id .eq. 0) then

! Set defaults for BoomerAMG
        maxiter = 500
        coarsen_type = 6
        hybrid = 1
        measure_type = 0
        strong_threshold = 0.25
        trunc_factor = 0.0
        cycle_type = 1
        smooth_num_sweep = 1

        print *, 'Solver: AMG'

        call NALU_HYPRE_BoomerAMGCreate(solver, ierr)
        call NALU_HYPRE_BoomerAMGSetCoarsenType(solver,
     &                                  (hybrid*coarsen_type), ierr)
        call NALU_HYPRE_BoomerAMGSetMeasureType(solver, measure_type, ierr)
        call NALU_HYPRE_BoomerAMGSetTol(solver, tol, ierr)
        call NALU_HYPRE_BoomerAMGSetStrongThrshld(solver,
     &                                      strong_threshold, ierr)
        call NALU_HYPRE_BoomerAMGSetTruncFactor(solver, trunc_factor, ierr)
        call NALU_HYPRE_BoomerAMGSetPrintLevel(solver, ioutdat,ierr)
        call NALU_HYPRE_BoomerAMGSetPrintFileName(solver,"test.out.log",ierr)
        call NALU_HYPRE_BoomerAMGSetMaxIter(solver, maxiter, ierr)
        call NALU_HYPRE_BoomerAMGSetCycleType(solver, cycle_type, ierr)

! RDF: Used this to test the fortran interface for SetDofFunc
!        do i = 1, 1000/2
!           j = 2*i-1
!           dof_func(j) = 0
!           j = j + 1
!           dof_func(j) = 1
!        enddo
!        call NALU_HYPRE_BoomerAMGSetNumFunctions(solver, 2, ierr)
!        call NALU_HYPRE_BoomerAMGSetDofFunc(solver, dof_func, ierr)

!        call NALU_HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
!     &                                      grid_relax_type,
!     &                                      grid_relax_points,
!     &                                      coarsen_type,
!     &                                      relax_weights,
!     &                                      MAXLEVELS,ierr)
!        num_grid_sweeps2(1) = 1
!        num_grid_sweeps2(2) = 1
!        num_grid_sweeps2(3) = 1
!        num_grid_sweeps2(4) = 1
!        call NALU_HYPRE_BoomerAMGSetNumGridSweeps(solver,
!     &                                       num_grid_sweeps2, ierr)
!        call NALU_HYPRE_BoomerAMGSetGridRelaxType(solver,
!     &                                       grid_relax_type, ierr)
!        call NALU_HYPRE_BoomerAMGSetRelaxWeight(solver,
!     &                                     relax_weights, ierr)
!       call NALU_HYPRE_BoomerAMGSetSmoothOption(solver, smooth_option,
!    &                                      ierr)
!       call NALU_HYPRE_BoomerAMGSetSmoothNumSwp(solver, smooth_num_sweep,
!    &                                      ierr)
!        call NALU_HYPRE_BoomerAMGSetGridRelaxPnts(solver,
!     &                                       grid_relax_points,
!     &                                       ierr)
        call NALU_HYPRE_BoomerAMGSetMaxLevels(solver, MAXLEVELS, ierr)
        call NALU_HYPRE_BoomerAMGSetMaxRowSum(solver, max_row_sum,
     &                                   ierr)
        call NALU_HYPRE_BoomerAMGSetDebugFlag(solver, debug_flag, ierr)
        call NALU_HYPRE_BoomerAMGSetup(solver, A_storage, b_storage,
     &                         x_storage, ierr)
        call NALU_HYPRE_BoomerAMGSolve(solver, A_storage, b_storage,
     &                         x_storage, ierr)
        call NALU_HYPRE_BoomerAMGGetNumIterations(solver, num_iterations, 
     &						ierr)
        call NALU_HYPRE_BoomerAMGGetFinalReltvRes(solver, final_res_norm,
     &                                       ierr)
        call NALU_HYPRE_BoomerAMGDestroy(solver, ierr)

      endif

      if (solver_id .eq. 3 .or. solver_id .eq. 4 .or.
     &    solver_id .eq. 7 .or. solver_id .eq. 8) then

        maxiter = 100
        k_dim = 5

!       Solve the system using preconditioned GMRES

        call NALU_HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, solver, ierr)
        call NALU_HYPRE_ParCSRGMRESSetKDim(solver, k_dim, ierr)
        call NALU_HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter, ierr)
        call NALU_HYPRE_ParCSRGMRESSetTol(solver, tol, ierr)
        call NALU_HYPRE_ParCSRGMRESSetLogging(solver, one, ierr)

        if (solver_id .eq. 4) then

          print *, 'Solver: DS-GMRES'

          precond_id = 1
          precond = 0

          call NALU_HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

        else if (solver_id .eq. 3) then

          print *, 'Solver: AMG-GMRES'

          precond_id = 2

! Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call NALU_HYPRE_BoomerAMGCreate(precond, ierr)
          call NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call NALU_HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                    (hybrid*coarsen_type), ierr)
          call NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &						ierr)
          call NALU_HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                        strong_threshold, ierr)
          call NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat, ierr)
          call NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call NALU_HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                        grid_relax_type,
     &                                        grid_relax_points,
     &                                        coarsen_type,
     &                                        relax_weights,
     &                                        MAXLEVELS,ierr)
          call NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call NALU_HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
!    &                                        ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
!    &                                        ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                        grid_relax_points, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxLevels(precond,
     &                                  MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)
          call NALU_HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          call NALU_HYPRE_BoomerAMGSetSetupType(precond,setup_type,ierr)
          
        else if (solver_id .eq. 7) then

          print *, 'Solver: Pilut-GMRES'

          precond_id = 3

          call NALU_HYPRE_ParCSRPilutCreate(MPI_COMM_WORLD,
     &                                 precond, ierr) 

          if (ierr .ne. 0) write(6,*) 'ParCSRPilutCreate error'

          call NALU_HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          if (drop_tol .ge. 0.)
     &        call NALU_HYPRE_ParCSRPilutSetDropToleran(precond,
     &                                              drop_tol, ierr)

        else if (solver_id .eq. 8) then

          print *, 'Solver: ParaSails-GMRES'

          precond_id = 4

          call NALU_HYPRE_ParaSailsCreate(MPI_COMM_WORLD, precond,
     &                               ierr)
          call NALU_HYPRE_ParCSRGMRESSetPrecond(solver, precond_id,
     &                                     precond, ierr)

          sai_threshold = 0.1
          nlevels       = 1
          sai_filter    = 0.1

          call NALU_HYPRE_ParaSailsSetParams(precond, sai_threshold,
     &                                  nlevels, ierr)
          call NALU_HYPRE_ParaSailsSetFilter(precond, sai_filter, ierr)
          call NALU_HYPRE_ParaSailsSetLogging(precond, ioutdat, ierr)

        endif

        call NALU_HYPRE_ParCSRGMRESGetPrecond(solver, precond_gotten,
     &                                   ierr)

        if (precond_gotten .ne. precond) then
          print *, 'NALU_HYPRE_ParCSRGMRESGetPrecond got bad precond'
          stop
        else
          print *, 'NALU_HYPRE_ParCSRGMRESGetPrecond got good precond'
        endif

        call NALU_HYPRE_ParCSRGMRESSetup(solver, A_storage, b_storage,
     &                              x_storage, ierr)
        call NALU_HYPRE_ParCSRGMRESSolve(solver, A_storage, b_storage,
     &                              x_storage, ierr)
        call NALU_HYPRE_ParCSRGMRESGetNumIteratio(solver,
     &                                       num_iterations, ierr)
        call NALU_HYPRE_ParCSRGMRESGetFinalRelati(solver,
     &                                       final_res_norm, ierr)

        if (solver_id .eq. 3) then
           call NALU_HYPRE_BoomerAMGDestroy(precond, ierr)
        else if (solver_id .eq. 7) then
           call NALU_HYPRE_ParCSRPilutDestroy(precond, ierr)
        else if (solver_id .eq. 8) then
           call NALU_HYPRE_ParaSailsDestroy(precond, ierr)
        endif

        call NALU_HYPRE_ParCSRGMRESDestroy(solver, ierr)

      endif

      if (solver_id .eq. 1 .or. solver_id .eq. 2) then

        maxiter = 500

        call NALU_HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, solver, ierr)
        call NALU_HYPRE_ParCSRPCGSetMaxIter(solver, maxiter, ierr)
        call NALU_HYPRE_ParCSRPCGSetTol(solver, tol, ierr)
        call NALU_HYPRE_ParCSRPCGSetTwoNorm(solver, one, ierr)
        call NALU_HYPRE_ParCSRPCGSetRelChange(solver, zero, ierr)
        call NALU_HYPRE_ParCSRPCGSetPrintLevel(solver, one, ierr)
  
        if (solver_id .eq. 2) then

          print *, 'Solver: DS-PCG'

          precond_id = 1
          precond = 0

          call NALU_HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     &                                   precond, ierr)

        else if (solver_id .eq. 1) then

          print *, 'Solver: AMG-PCG'

          precond_id = 2

! Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call NALU_HYPRE_BoomerAMGCreate(precond, ierr)
          call NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call NALU_HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold,
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call NALU_HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call NALU_HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
!    &                                        ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothNumSwp(precond,
!    &                                        smooth_num_sweep,
!    &                                        ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call NALU_HYPRE_ParCSRPCGSetPrecond(solver, precond_id,
     &                                   precond, ierr)

        endif

        call NALU_HYPRE_ParCSRPCGGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'NALU_HYPRE_ParCSRPCGGetPrecond got bad precond'
          stop
        else
          print *, 'NALU_HYPRE_ParCSRPCGGetPrecond got good precond'
        endif

        call NALU_HYPRE_ParCSRPCGSetup(solver, A_storage, b_storage,
     &                            x_storage, ierr)
        call NALU_HYPRE_ParCSRPCGSolve(solver, A_storage, b_storage,
     &                            x_storage, ierr)
        call NALU_HYPRE_ParCSRPCGGetNumIterations(solver, num_iterations,
     &                                       ierr)
        call NALU_HYPRE_ParCSRPCGGetFinalRelative(solver, final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 1) then
          call NALU_HYPRE_BoomerAMGDestroy(precond, ierr)
        endif

        call NALU_HYPRE_ParCSRPCGDestroy(solver, ierr)

      endif

      if (solver_id .eq. 5 .or. solver_id .eq. 6) then

        maxiter = 1000

        call NALU_HYPRE_ParCSRCGNRCreate(MPI_COMM_WORLD, solver, ierr)
        call NALU_HYPRE_ParCSRCGNRSetMaxIter(solver, maxiter, ierr)
        call NALU_HYPRE_ParCSRCGNRSetTol(solver, tol, ierr)
        call NALU_HYPRE_ParCSRCGNRSetLogging(solver, one, ierr)

        if (solver_id .eq. 6) then

          print *, 'Solver: DS-CGNR'

          precond_id = 1
          precond = 0

          call NALU_HYPRE_ParCSRCGNRSetPrecond(solver, precond_id,
     &                                    precond, ierr)

        else if (solver_id .eq. 5) then 

          print *, 'Solver: AMG-CGNR'

          precond_id = 2

! Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call NALU_HYPRE_BoomerAMGCreate(precond, ierr)
          call NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call NALU_HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold, ierr)
          call NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call NALU_HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS,ierr)
          call NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call NALU_HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
!    &                                        ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
!    &                                        ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points,
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call NALU_HYPRE_ParCSRCGNRSetPrecond(solver, precond_id, precond,
     &                                    ierr)
        endif

        call NALU_HYPRE_ParCSRCGNRGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'NALU_HYPRE_ParCSRCGNRGetPrecond got bad precond'
          stop
        else
          print *, 'NALU_HYPRE_ParCSRCGNRGetPrecond got good precond'
        endif

        call NALU_HYPRE_ParCSRCGNRSetup(solver, A_storage, b_storage,
     &                             x_storage, ierr)
        call NALU_HYPRE_ParCSRCGNRSolve(solver, A_storage, b_storage,
     &                             x_storage, ierr)
        call NALU_HYPRE_ParCSRCGNRGetNumIteration(solver, num_iterations,
     &                                      ierr)
        call NALU_HYPRE_ParCSRCGNRGetFinalRelativ(solver, final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 5) then
          call NALU_HYPRE_BoomerAMGDestroy(precond, ierr)
        endif 

        call NALU_HYPRE_ParCSRCGNRDestroy(solver, ierr)

      endif

      if (solver_id .eq. 9 .or. solver_id .eq. 10) then

        maxiter = 1000

        call NALU_HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, solver, ierr)
        call NALU_HYPRE_ParCSRBiCGSTABSetMaxIter(solver, maxiter, ierr)
        call NALU_HYPRE_ParCSRBiCGSTABSetTol(solver, tol, ierr)
        call NALU_HYPRE_ParCSRBiCGSTABSetLogging(solver, one, ierr)

        if (solver_id .eq. 10) then

          print *, 'Solver: DS-BiCGSTAB'

          precond_id = 1
          precond = 0

          call NALU_HYPRE_ParCSRBiCGSTABSetPrecond(solver, precond_id,
     &                                        precond, ierr)

        else if (solver_id .eq. 9) then

          print *, 'Solver: AMG-BiCGSTAB'

          precond_id = 2

! Set defaults for BoomerAMG
          maxiter = 1
          coarsen_type = 6
          hybrid = 1
          measure_type = 0
          setup_type = 1
          strong_threshold = 0.25
          trunc_factor = 0.0
          cycle_type = 1
          smooth_num_sweep = 1

          call NALU_HYPRE_BoomerAMGCreate(precond, ierr)
          call NALU_HYPRE_BoomerAMGSetTol(precond, pc_tol, ierr)
          call NALU_HYPRE_BoomerAMGSetCoarsenType(precond,
     &                                       (hybrid*coarsen_type),
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetMeasureType(precond, measure_type, 
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetStrongThrshld(precond,
     &                                         strong_threshold,
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor,
     &                                       ierr)
          call NALU_HYPRE_BoomerAMGSetPrintLevel(precond, ioutdat,ierr)
          call NALU_HYPRE_BoomerAMGSetPrintFileName(precond, "test.out.log",
     &                                         ierr)
          call NALU_HYPRE_BoomerAMGSetMaxIter(precond, maxiter, ierr)
          call NALU_HYPRE_BoomerAMGSetCycleType(precond, cycle_type, ierr)
          call NALU_HYPRE_BoomerAMGInitGridRelaxatn(num_grid_sweeps,
     &                                         grid_relax_type,
     &                                         grid_relax_points,
     &                                         coarsen_type,
     &                                         relax_weights,
     &                                         MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetNumGridSweeps(precond,
     &                                         num_grid_sweeps, ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxType(precond,
     &                                         grid_relax_type, ierr)
          call NALU_HYPRE_BoomerAMGSetRelaxWeight(precond,
     &                                       relax_weights, ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothOption(precond, smooth_option,
!    &                                        ierr)
!         call NALU_HYPRE_BoomerAMGSetSmoothNumSwp(precond, smooth_num_sweep,
!    &                                        ierr)
          call NALU_HYPRE_BoomerAMGSetGridRelaxPnts(precond,
     &                                         grid_relax_points, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxLevels(precond, MAXLEVELS, ierr)
          call NALU_HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum,
     &                                     ierr)

          call NALU_HYPRE_ParCSRBiCGSTABSetPrecond(solver, precond_id,
     &                                        precond,
     &                                        ierr)

        endif

        call NALU_HYPRE_ParCSRBiCGSTABGetPrecond(solver,precond_gotten,ierr)

        if (precond_gotten .ne. precond) then
          print *, 'NALU_HYPRE_ParCSRBiCGSTABGetPrecond got bad precond'
          stop
        else
          print *, 'NALU_HYPRE_ParCSRBiCGSTABGetPrecond got good precond'
        endif

        call NALU_HYPRE_ParCSRBiCGSTABSetup(solver, A_storage, b_storage,
     &                                 x_storage, ierr)
        call NALU_HYPRE_ParCSRBiCGSTABSolve(solver, A_storage, b_storage,
     &                                 x_storage, ierr)
        call NALU_HYPRE_ParCSRBiCGSTABGetNumIter(solver,
     &                                      num_iterations,
     &                                      ierr)
        call NALU_HYPRE_ParCSRBiCGSTABGetFinalRel(solver,
     &                                       final_res_norm,
     &                                       ierr)

        if (solver_id .eq. 9) then
          call NALU_HYPRE_BoomerAMGDestroy(precond, ierr)
        endif

        call NALU_HYPRE_ParCSRBiCGSTABDestroy(solver, ierr)

      endif

!-----------------------------------------------------------------------
!     Print the solution and other info
!-----------------------------------------------------------------------

      vecfile(1)  = 'd'
      vecfile(2)  = 'r'
      vecfile(3)  = 'i'
      vecfile(4)  = 'v'
      vecfile(5)  = 'e'
      vecfile(6)  = 'r'
      vecfile(7)  = '.'
      vecfile(8)  = 'o'
      vecfile(9)  = 'u'
      vecfile(10) = 't'
      vecfile(11) = '.'
      vecfile(12) = 'x'
      vecfile(13) = char(0)
   
      call NALU_HYPRE_IJVectorPrint(x, vecfile, ierr)

      if (myid .eq. 0) then
         print *, 'Iterations = ', num_iterations
         print *, 'Final Residual Norm = ', final_res_norm
      endif

!-----------------------------------------------------------------------
!     Finalize things
!-----------------------------------------------------------------------

      call NALU_HYPRE_ParCSRMatrixDestroy(A_storage, ierr)
      call NALU_HYPRE_IJVectorDestroy(b, ierr)
      call NALU_HYPRE_IJVectorDestroy(x, ierr)

!     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end
