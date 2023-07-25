!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!-----------------------------------------------------------------------
! Test driver for structured matrix interface (structured storage)
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
! Standard 7-point laplacian in 3D with grid and anisotropy determined
! as user settings.
!-----------------------------------------------------------------------

      program test

      implicit none

      include 'mpif.h'

      integer   MAXZONS
      integer   MAXBLKS
      integer   MAXDIM

      parameter (MAXZONS=4194304)
      parameter (MAXBLKS=32)
      parameter (MAXDIM=3)

      integer             num_procs, myid
      logical             file_exists

      integer             dim
      integer             nx, ny, nz
      integer             Px, Py, Pz
      integer             bx, by, bz
      double precision    cx, cy, cz
      integer             n_pre, n_post
      integer             slvr_id
      integer             prec_id

      integer             zero, one
      integer             prec_iter
      integer             slvr_iter
      integer             dscg_iter
      double precision    slvr_tol
      double precision    prec_tol
      double precision    convtol

      integer             num_iterations
      double precision    final_res_norm

!     NALU_HYPRE_StructMatrix  A
!     NALU_HYPRE_StructVector  b
!     NALU_HYPRE_StructVector  x

      integer*8           A
      integer*8           b
      integer*8           x

!     NALU_HYPRE_StructSolver  solver
!     NALU_HYPRE_StructSolver  precond

      integer*8           solver
      integer*8           precond

!     NALU_HYPRE_StructGrid    grid
!     NALU_HYPRE_StructStencil stencil

      integer*8           grid
      integer*8           stencil

      double precision    dxyz(3)

      integer             A_num_ghost(6)

      integer             iupper(3,MAXBLKS), ilower(3,MAXBLKS)

      integer             istart(3)

      integer             offsets(3,MAXDIM+1)

      integer             stencil_indices(MAXDIM+1)
      double precision    values((MAXDIM+1)*MAXZONS)

      integer             p, q, r
      integer             nblocks, volume

      integer             i, s, d
      integer             ix, iy, iz, ib
      integer             ierr

      data A_num_ghost   / 0, 0, 0, 0, 0, 0 /
      data zero          / 0 /
      data one           / 1 /

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

      slvr_id = 1

      istart(1) = -3
      istart(2) = -3
      istart(3) = -3

!-----------------------------------------------------------------------
!     Read options
!-----------------------------------------------------------------------
      inquire(file='struct_linear_solver.in', exist=file_exists)
      if (file_exists) then
         open(5, file='struct_linear_solver.in', status='old')

         read(5, *) dim

         read(5, *) nx
         read(5, *) ny
         read(5, *) nz

         read(5, *) Px
         read(5, *) Py
         read(5, *) Pz

         read(5, *) bx
         read(5, *) by
         read(5, *) bz

         read(5, *) cx
         read(5, *) cy
         read(5, *) cz

         read(5, *) n_pre
         read(5, *) n_post
         read(5, *) slvr_id

         close(5)
      endif

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
         write(*,*)
         write(*,'(a)') 'Running with these driver parameters:'
         write(*,'(3x,a,i0,a,i0,a,i0,a)') '(nx, ny, nz)    = (',
     &                                      nx, ', ', ny, ', ', nz, ')'
         write(*,'(3x,a,i0,a,i0,a,i0,a)') '(Px, Py, Pz)    = (',
     &                                      Px, ', ', Py, ', ', Pz, ')'
         write(*,'(3x,a,i0,a,i0,a,i0,a)') '(bx, by, bz)    = (',
     &                                      bx, ', ', by, ', ', bz, ')'
         write(*,'(3x,a,f0.1,a,f0.1,a,f0.1,a)') '(cx, cy, cz)    = (',
     &                                      cx, ', ', cy, ', ', cz, ')'
         write(*,'(3x,a,i0,a,i0,a)') '(n_pre, n_post) = (',
     &                                n_pre, ', ', n_post, ')'
         write(*,'(3x,a,i0)') 'dim             = ', dim
         write(*,'(3x,a,i0)') 'solver ID       = ', slvr_id
         write(*,*)
      endif

!-----------------------------------------------------------------------
!     Set up dxyz for PFMG solver
!-----------------------------------------------------------------------

      dxyz(1) = dsqrt(1.d0 / cx)
      dxyz(2) = dsqrt(1.d0 / cy)
      dxyz(3) = dsqrt(1.d0 / cz)

!-----------------------------------------------------------------------
!     Compute some grid and processor information
!-----------------------------------------------------------------------

      if (dim .eq. 1) then
         volume  = nx
         nblocks = bx

!        compute p from Px and myid
         p = mod(myid,Px)
      elseif (dim .eq. 2) then
         volume  = nx*ny
         nblocks = bx*by

!        compute p,q from Px, Py and myid
         p = mod(myid,Px)
         q = mod(((myid - p)/Px),Py)
      elseif (dim .eq. 3) then
         volume  = nx*ny*nz
         nblocks = bx*by*bz

!        compute p,q,r from Px,Py,Pz and myid
         p = mod(myid,Px)
         q = mod((( myid - p)/Px),Py)
         r = (myid - (p + Px*q))/(Px*Py)
      endif

!----------------------------------------------------------------------
!    Compute ilower and iupper from (p,q,r), (bx,by,bz), and (nx,ny,nz)
!    and set up the grid structure.
!----------------------------------------------------------------------

      ib = 1
      if (dim .eq. 1) then
         do ix=0,bx-1
            ilower(1,ib) = istart(1) + nx*(bx*p+ix)
            iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
            ib = ib + 1
         enddo
      elseif (dim .eq. 2) then
         do iy=0,by-1
            do ix=0,bx-1
               ilower(1,ib) = istart(1) + nx*(bx*p+ix)
               iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
               ilower(2,ib) = istart(2) + ny*(by*q+iy)
               iupper(2,ib) = istart(2) + ny*(by*q+iy+1)-1
               ib = ib + 1
            enddo
         enddo
      elseif (dim .eq. 3) then
         do iz=0,bz-1
            do iy=0,by-1
               do ix=0,bx-1
                  ilower(1,ib) = istart(1) + nx*(bx*p+ix)
                  iupper(1,ib) = istart(1) + nx*(bx*p+ix+1)-1
                  ilower(2,ib) = istart(2) + ny*(by*q+iy)
                  iupper(2,ib) = istart(2) + ny*(by*q+iy+1)-1
                  ilower(3,ib) = istart(3) + nz*(bz*r+iz)
                  iupper(3,ib) = istart(3) + nz*(bz*r+iz+1)-1
                  ib = ib + 1
               enddo
            enddo
         enddo
      endif

      call NALU_HYPRE_StructGridCreate(MPI_COMM_WORLD, dim, grid, ierr)
      do ib=1,nblocks
         call NALU_HYPRE_StructGridSetExtents(grid, ilower(1,ib),
     & iupper(1,ib), ierr)
      enddo
      call NALU_HYPRE_StructGridAssemble(grid, ierr)

!----------------------------------------------------------------------
!     Compute the offsets and set up the stencil structure.
!----------------------------------------------------------------------

      if (dim .eq. 1) then
         offsets(1,1) = -1
         offsets(1,2) =  0
      elseif (dim .eq. 2) then
         offsets(1,1) = -1
         offsets(2,1) =  0
         offsets(1,2) =  0
         offsets(2,2) = -1
         offsets(1,3) =  0
         offsets(2,3) =  0
      elseif (dim .eq. 3) then
         offsets(1,1) = -1
         offsets(2,1) =  0
         offsets(3,1) =  0
         offsets(1,2) =  0
         offsets(2,2) = -1
         offsets(3,2) =  0
         offsets(1,3) =  0
         offsets(2,3) =  0
         offsets(3,3) = -1
         offsets(1,4) =  0
         offsets(2,4) =  0
         offsets(3,4) =  0
      endif

      call NALU_HYPRE_StructStencilCreate(dim, (dim+1), stencil, ierr)
      do s=1,dim+1
         call NALU_HYPRE_StructStencilSetElement(stencil, (s - 1),
     & offsets(1,s), ierr)
      enddo

!-----------------------------------------------------------------------
!     Set up the matrix structure
!-----------------------------------------------------------------------

      do i=1,dim
         A_num_ghost(2*i - 1) = 1
         A_num_ghost(2*i) = 1
      enddo

      call NALU_HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil,
     & A, ierr)
      call NALU_HYPRE_StructMatrixSetSymmetric(A, 1, ierr)
      call NALU_HYPRE_StructMatrixSetNumGhost(A, A_num_ghost, ierr)
      call NALU_HYPRE_StructMatrixInitialize(A, ierr)

!-----------------------------------------------------------------------
!     Set the coefficients for the grid
!-----------------------------------------------------------------------

      do s=1,(dim + 1)
         stencil_indices(s) = s - 1
      enddo

      do i=1,(dim + 1)*volume,(dim + 1)
         if (dim .eq. 1) then
            values(i  ) = -cx
            values(i+1) = 2.0*(cx)
         elseif (dim .eq. 2) then
            values(i  ) = -cx
            values(i+1) = -cy
            values(i+2) = 2.0*(cx+cy)
         elseif (dim .eq. 3) then
            values(i  ) = -cx
            values(i+1) = -cy
            values(i+2) = -cz
            values(i+3) = 2.0*(cx+cy+cz)
         endif
      enddo

      do ib=1,nblocks
         call NALU_HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
     & iupper(1,ib), (dim+1), stencil_indices, values, ierr)
      enddo

!-----------------------------------------------------------------------
!     Zero out stencils reaching to real boundary
!-----------------------------------------------------------------------

      do i=1,volume
         values(i) = 0.0
      enddo
      do d=1,dim
         do ib=1,nblocks
            if( ilower(d,ib) .eq. istart(d) ) then
               i = iupper(d,ib)
               iupper(d,ib) = istart(d)
               stencil_indices(1) = d - 1
               call NALU_HYPRE_StructMatrixSetBoxValues(A, ilower(1,ib),
     & iupper(1,ib), 1, stencil_indices, values, ierr)
               iupper(d,ib) = i
            endif
         enddo
      enddo

      call NALU_HYPRE_StructMatrixAssemble(A, ierr)
!     call NALU_HYPRE_StructMatrixPrint(A, zero, ierr)
!     call NALU_HYPRE_StructMatrixPrint("driver.out.A", A, zero, ierr)

!-----------------------------------------------------------------------
!     Set up the rhs and initial guess
!-----------------------------------------------------------------------

      call NALU_HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, b, ierr)
      call NALU_HYPRE_StructVectorInitialize(b, ierr)
      do i=1,volume
         values(i) = 1.0
      enddo
      do ib=1,nblocks
         call NALU_HYPRE_StructVectorSetBoxValues(b, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call NALU_HYPRE_StructVectorAssemble(b, ierr)
!     call NALU_HYPRE_StructVectorPrint("driver.out.b", b, zero, ierr)

      call NALU_HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, x, ierr)
      call NALU_HYPRE_StructVectorInitialize(x, ierr)
      do i=1,volume
         values(i) = 0.0
      enddo
      do ib=1,nblocks
         call NALU_HYPRE_StructVectorSetBoxValues(x, ilower(1,ib),
     & iupper(1,ib), values, ierr)
      enddo
      call NALU_HYPRE_StructVectorAssemble(x, ierr)
!     call NALU_HYPRE_StructVectorPrint(x, zero, ierr)
!     call NALU_HYPRE_StructVectorPrint("driver.out.x0", x, zero, ierr)

!-----------------------------------------------------------------------
!     Solve the linear system
!-----------------------------------------------------------------------

!     General solver parameters, passing hard coded constants
!     will break the interface.
      slvr_iter = 50
      prec_iter = 1
      dscg_iter = 100
      slvr_tol  = 0.000001
      prec_tol  = 0.0
      convtol   = 0.9
      prec_id   = -1

      if (slvr_id .eq. 0) then
!        Solve the system using SMG

         call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructSMGSetMemoryUse(solver, zero, ierr)
         call NALU_HYPRE_StructSMGSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructSMGSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructSMGSetRelChange(solver, zero, ierr)
         call NALU_HYPRE_StructSMGSetNumPreRelax(solver, n_pre, ierr)
         call NALU_HYPRE_StructSMGSetNumPostRelax(solver, n_post, ierr)
         call NALU_HYPRE_StructSMGSetLogging(solver, one, ierr)
         call NALU_HYPRE_StructSMGSetup(solver, A, b, x, ierr)

         call NALU_HYPRE_StructSMGSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructSMGGetNumIterations(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructSMGGetFinalRelative(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructSMGDestroy(solver, ierr)

      elseif (slvr_id .eq. 1) then
!        Solve the system using PFMG

         call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructPFMGSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructPFMGSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructPFMGSetRelChange(solver, zero, ierr)
!        weighted Jacobi = 1; red-black GS = 2
         call NALU_HYPRE_StructPFMGSetRelaxType(solver, one, ierr)
         call NALU_HYPRE_StructPFMGSetNumPreRelax(solver, n_pre, ierr)
         call NALU_HYPRE_StructPFMGSetNumPostRelax(solver, n_post, ierr)
!        call NALU_HYPRE_StructPFMGSetDxyz(solver, dxyz, ierr)
         call NALU_HYPRE_StructPFMGSetLogging(solver, one, ierr)
         call NALU_HYPRE_StructPFMGSetPrintLevel(solver, one, ierr)
         call NALU_HYPRE_StructPFMGSetup(solver, A, b, x, ierr)

         call NALU_HYPRE_StructPFMGSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructPFMGGetNumIteration(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructPFMGGetFinalRelativ(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructPFMGDestroy(solver, ierr)

      elseif ((slvr_id .gt. 9) .and. (slvr_id .lt. 20)) then
!        Solve the system using CG

         call NALU_HYPRE_StructPCGCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructPCGSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructPCGSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructPCGSetTwoNorm(solver, one, ierr)
         call NALU_HYPRE_StructPCGSetRelChange(solver, zero, ierr)
         call NALU_HYPRE_StructPCGSetLogging(solver, one, ierr)

         if (slvr_id .eq. 10) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 11) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 17) then
!           use 2-step jacobi as preconditioner
            prec_id = 7
            prec_iter  = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)

         elseif (slvr_id .eq. 18) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 19) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructPCGSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructPCGSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructPCGSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructPCGGetNumIterations(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructPCGGetFinalRelative(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructPCGDestroy(solver, ierr)

      elseif ((slvr_id .gt. 19) .and. (slvr_id .lt. 30)) then
!        Solve the system using Hybrid

         call NALU_HYPRE_StructHybridCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructHybridSetDSCGMaxIte(solver, dscg_iter, ierr)
         call NALU_HYPRE_StructHybridSetPCGMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructHybridSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructHybridSetConvergenc(solver, convtol, ierr)
         call NALU_HYPRE_StructHybridSetTwoNorm(solver, one, ierr)
         call NALU_HYPRE_StructHybridSetRelChange(solver, zero, ierr)
         call NALU_HYPRE_StructHybridSetLogging(solver, one, ierr)

         if (slvr_id .eq. 20) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 21) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 27) then
!           use 2-step jacobi as preconditioner
            prec_id = 7
            prec_iter  = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)

         elseif (slvr_id .eq. 28) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 29) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructHybridSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructHybridSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructHybridSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructHybridGetNumIterati(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructHybridGetFinalRelat(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructHybridDestroy(solver, ierr)

      elseif ((slvr_id .gt. 29) .and. (slvr_id .lt. 40)) then
!        Solve the system using GMRes

         call NALU_HYPRE_StructGMResCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructGMResSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructGMResSetKDim(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructGMResSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructGMResSetPrintLevel(solver, one);
         call NALU_HYPRE_StructGMResSetLogging(solver, one, ierr)

         if (slvr_id .eq. 30) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 31) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 36) then
!           use 2-step jacobi as preconditioner
            prec_id   = 6
            prec_iter = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)
            prec_id = 7

         elseif (slvr_id .eq. 38) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 39) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructGMResSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructGMResSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructGMResSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructGMResGetNumIteratio(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructGMResGetFinalRelati(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructGMResDestroy(solver, ierr)

      elseif ((slvr_id .gt. 39) .and. (slvr_id .lt. 50)) then
!        Solve the system using BiCGStab

         call NALU_HYPRE_StructBiCGStabCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructBiCGStabSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructBiCGStabSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructBiCGStabSetPrintLev(solver, one);
         call NALU_HYPRE_StructBiCGStabSetLogging(solver, one, ierr)

         if (slvr_id .eq. 40) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 41) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 47) then
!           use 2-step jacobi as preconditioner
            prec_id   = 7
            prec_iter = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)

         elseif (slvr_id .eq. 38) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 39) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructBiCGStabSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructBiCGStabSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructBiCGStabSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructBiCGStabGetNumItera(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructBiCGStabGetFinalRel(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructBiCGStabDestroy(solver, ierr)

      elseif ((slvr_id .gt. 49) .and. (slvr_id .lt. 60)) then
!        Solve the system using LGMRes

         call NALU_HYPRE_StructLGMResCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructLGMResSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructLGMResSetKDim(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructLGMResSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructLGMResSetPrintLevel(solver, one);
         call NALU_HYPRE_StructLGMResSetLogging(solver, one, ierr)

         if (slvr_id .eq. 50) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 51) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 56) then
!           use 2-step jacobi as preconditioner
            prec_id   = 6
            prec_iter = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)
            prec_id = 7

         elseif (slvr_id .eq. 58) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 59) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructLGMResSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructLGMResSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructLGMResSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructLGMResGetNumIter(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructLGMResGetFinalRel(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructLGMResDestroy(solver, ierr)

      elseif ((slvr_id .gt. 59) .and. (slvr_id .lt. 70)) then
!        Solve the system using FlexGMRes

         call NALU_HYPRE_StructFGMResCreate(MPI_COMM_WORLD, solver, ierr)
         call NALU_HYPRE_StructFGMResSetMaxIter(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructFGMResSetKDim(solver, slvr_iter, ierr)
         call NALU_HYPRE_StructFGMResSetTol(solver, slvr_tol, ierr)
         call NALU_HYPRE_StructFGMResSetPrintLevel(solver, one);
         call NALU_HYPRE_StructFGMResSetLogging(solver, one, ierr)

         if (slvr_id .eq. 60) then
!           use symmetric SMG as preconditioner
            prec_id = 0

            call NALU_HYPRE_StructSMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructSMGSetMemoryUse(precond, zero, ierr)
            call NALU_HYPRE_StructSMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructSMGSetTol(precond, prec_tol, ierr)
            call NALU_HYPRE_StructSMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructSMGSetNumPostRelax(precond, n_post, ierr)
            call NALU_HYPRE_StructSMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 61) then
!           use symmetric PFMG as preconditioner
            prec_id = 1

            call NALU_HYPRE_StructPFMGCreate(MPI_COMM_WORLD, precond,
     & ierr)
            call NALU_HYPRE_StructPFMGSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructPFMGSetTol(precond, prec_tol, ierr)
!           weighted Jacobi = 1; red-black GS = 2
            call NALU_HYPRE_StructPFMGSetRelaxType(precond, one, ierr)
            call NALU_HYPRE_StructPFMGSetNumPreRelax(precond, n_pre, ierr)
            call NALU_HYPRE_StructPFMGSetNumPostRelax(precond, n_post, ierr)
!           call NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz, ierr)
            call NALU_HYPRE_StructPFMGSetLogging(precond, zero, ierr)

         elseif (slvr_id .eq. 66) then
!           use 2-step jacobi as preconditioner
            prec_id   = 6
            prec_iter = 2

            call NALU_HYPRE_StructJacobiCreate(MPI_COMM_WORLD, precond, ierr)
            call NALU_HYPRE_StructJacobiSetMaxIter(precond, prec_iter, ierr)
            call NALU_HYPRE_StructJacobiSetTol(precond, prec_tol, ierr)
            prec_id = 7

         elseif (slvr_id .eq. 68) then
!           use diagonal scaling as preconditioner
            prec_id = 8
            precond = zero

         elseif (slvr_id .eq. 69) then
!           use no preconditioner
            prec_id = 9

         else
!           Invalid preconditioner
            if (myid .eq. 0) then
               write(*,*) 'Invalid preconditioner!'
            endif
            call MPI_ABORT(MPI_COMM_WORLD, -2, ierr)
         endif

         call NALU_HYPRE_StructFGMResSetPrecond(solver, prec_id, precond,
     & ierr)
         call NALU_HYPRE_StructFGMResSetup(solver, A, b, x, ierr)
         call NALU_HYPRE_StructFGMResSolve(solver, A, b, x, ierr)
         call NALU_HYPRE_StructFGMResGetNumIter(solver, num_iterations,
     & ierr)
         call NALU_HYPRE_StructFGMResGetFinalRel(solver, final_res_norm,
     & ierr)
         call NALU_HYPRE_StructFGMResDestroy(solver, ierr)

      else
         if (myid .eq. 0) then
            write(*,*) 'Invalid solver!'
         endif
         call MPI_ABORT(MPI_COMM_WORLD, -1, ierr)
      endif

      if (prec_id .eq. 0) then
         call NALU_HYPRE_StructSMGDestroy(precond, ierr)

      elseif (prec_id .eq. 1) then
         call NALU_HYPRE_StructPFMGDestroy(precond, ierr)

      elseif (prec_id .eq. 7) then
         call NALU_HYPRE_StructJacobiDestroy(precond, ierr)
      endif

!-----------------------------------------------------------------------
!     Print the solution and other info
!-----------------------------------------------------------------------

!  call NALU_HYPRE_StructVectorPrint("driver.out.x", x, zero, ierr)

      if (myid .eq. 0) then
         write(*,'(a, i0)') 'Number of iterations = ', num_iterations
         write(*,'(a, e15.7)') 'Final Relative Residual Norm = ',
     &                          final_res_norm
      endif

!-----------------------------------------------------------------------
!     Finalize things
!-----------------------------------------------------------------------

      call NALU_HYPRE_StructGridDestroy(grid, ierr)
      call NALU_HYPRE_StructStencilDestroy(stencil, ierr)
      call NALU_HYPRE_StructMatrixDestroy(A, ierr)
      call NALU_HYPRE_StructVectorDestroy(b, ierr)
      call NALU_HYPRE_StructVectorDestroy(x, ierr)

!     Finalize MPI

      call MPI_FINALIZE(ierr)

      stop
      end
