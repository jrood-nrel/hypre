!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!***********************************************************************
!     Routines to test struct_ls fortran interfaces
!***********************************************************************


!***********************************************************************
!             NALU_HYPRE_StructBiCGSTAB routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structbicgstabcreate
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabdestroy
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsetup
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructBiCGSTABSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsolve
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructBiCGSTABSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsettol
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructBiCGSTABSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsetprecond
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsetprecond(fsolver, fprecond_id,
     1                                           fprecond_solver)
      integer ierr
      integer*8 fsolver
      integer*8 fprecond_id
      integer*8 fprecond_solver

      call NALU_HYPRE_StructBiCGSTABSetPrecond(fsolver, fprecond_id,
     1                                    fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabsetprintle(fsolver, fprintlev)
      integer ierr
      integer fprintlev
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABSetPrintLev(fsolver, fprintlev, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabsetprintle: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabgetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabgetnumiter(fsolver, fnumiter)
      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_StructBiCGSTABGetNumItera(fsolver, fnumiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabgetnumiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabgetresidual
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabgetresidua(fsolver, fresidual)
      integer ierr
      integer*8 fsolver
      double precision fresidual

      call NALU_HYPRE_StructBiCGSTABGetResidual(fsolver, fresidual, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabgetresidua: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structbicgstabgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structbicgstabgetfinalre(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructBiCGSTABGetFinalRel(fsolver, fnorm)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structbicgstabgetfinalre: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructGMRES routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structgmrescreate
!***********************************************************************
      subroutine fnalu_hypre_structgmrescreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmrescreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmresdestroy
!***********************************************************************
      subroutine fnalu_hypre_structgmresdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmresdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressetup
!***********************************************************************
      subroutine fnalu_hypre_structgmressetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructGMRESSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressolve
!***********************************************************************
      subroutine fnalu_hypre_structgmressolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructGMRESSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressettol
!***********************************************************************
      subroutine fnalu_hypre_structgmressettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructGMRESSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structgmressetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressetprecond
!***********************************************************************
      subroutine fnalu_hypre_structgmressetprecond(fsolver, fprecond_id,
     1                                        fprecond_solver)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond_solver

      call NALU_HYPRE_StructGMRESSetPrecond(fsolver, fprecond_id,
     1                                 fprecond_solver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressetlogging
!***********************************************************************
      subroutine fnalu_hypre_structgmressetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmressetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structgmressetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESSetPrintLevel(fsolver, fprint_level, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmressetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmresgetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structgmresgetnumiterati(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call NALU_HYPRE_StructGMRESGetNumIteratio(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmresgetnumiterati: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structgmresgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structgmresgetfinalrelat(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructGMRESGetFinalRelati(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgmresgetfinalrelat: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructHybrid routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structhybridcreate
!***********************************************************************
      subroutine fnalu_hypre_structhybridcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructHybridCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybriddestroy
!***********************************************************************
      subroutine fnalu_hypre_structhybriddestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructHybridDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybriddestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetup
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructHybridSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsolve
!***********************************************************************
      subroutine fnalu_hypre_structhybridsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructHybridSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetsolvertype
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetsolvertyp(fsolver, fsolver_typ)
      integer ierr
      integer fsolver_typ
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetSolverType(fsolver, fsolver_typ, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetsolvertyp: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetstopcrit
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetstopcrit(fsolver, fstop_crit)
      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetStopCrit(fsolver, fstop_crit, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetstopcrit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetkdim
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetkdim(fsolver, fkdim)
      integer ierr
      integer fkdim
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetKDim(fsolver, fkdim, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetkdim: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsettol
!***********************************************************************
      subroutine fnalu_hypre_structhybridsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructHybridSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetconvergencetol
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetconvergen(fsolver, fcftol)
      integer ierr
      integer*8 fsolver
      double precision fcftol

      call NALU_HYPRE_StructHybridSetConvergenc(fsolver, fcftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetconvergen: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetpcgabsolutetolfactor
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetpcgabsolu(fsolver, fpcgtol)
      integer ierr
      integer*8 fsolver
      double precision fpcgtol

      call NALU_HYPRE_StructHybridSetPCGAbsolut(fsolver, fpcgtol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetpcgabsolu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetdscgmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetdscgmaxit(fsolver, fdscgmaxitr)
      integer ierr
      integer fdscgmaxitr
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetDSCGMaxIte(fsolver, fdscgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetdscgmaxit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetpcgmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetpcgmaxite(fsolver, fpcgmaxitr)
      integer ierr
      integer fpcgmaxitr
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetPCGMaxIter(fsolver, fpcgmaxitr, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetpcgmaxite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsettwonorm
!***********************************************************************
      subroutine fnalu_hypre_structhybridsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsettwonorm: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetprecond
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetprecond(fsolver, fprecond_id,
     1                                         fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_StructHybridSetPrecond(fsolver, fprecond_id, fprecond,
     1                                  ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structhybridsetprintleve(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_StructHybridSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridsetprintleve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridgetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structhybridgetnumiterat(fsolver, fnumits)
      integer ierr
      integer fnumits
      integer*8 fsolver

      call NALU_HYPRE_StructHybridGetNumIterati(fsolver, fnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridgetnumiterat: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridgetdscgnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structhybridgetdscgnumit(fsolver, fdscgnumits)
      integer ierr
      integer fdscgnumits
      integer*8 fsolver

      call NALU_HYPRE_StructHybridGetDSCGNumIte(fsolver, fdscgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridgetdscgnumit: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridgetpcgnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structhybridgetpcgnumite(fsolver, fpcgnumits)
      integer ierr
      integer fpcgnumits
      integer*8 fsolver

      call NALU_HYPRE_StructHybridGetPCGNumIter(fsolver, fpcgnumits, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridgetpcgnumite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structhybridgetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structhybridgetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructHybridGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structhybridgetfinalrela: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructInterpreter routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structvectorsetrandomvalues
!***********************************************************************
      subroutine fnalu_hypre_structvectorsetrandomvalu(fvector, fseed)
      integer ierr
      integer fseed
      integer*8 fvector

      call nalu_hypre_StructVectorSetRandomValu(fvector, fseed, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorsetrandomvalues: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fnalu_hypre_structsetrandomvalues
!***********************************************************************
      subroutine fnalu_hypre_structsetrandomvalues(fvector, fseed)
      integer ierr
      integer fseed
      integer*8 fvector

      call nalu_hypre_StructSetRandomValues(fvector, fseed, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsetrandomvalues: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fnalu_hypre_structsetupinterpreter
!***********************************************************************
      subroutine fnalu_hypre_structsetupinterpreter(fi)
      integer ierr
      integer*8 fi

      call NALU_HYPRE_StructSetupInterpreter(fi, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsetupinterpreter: err = ', ierr
      endif

      return
      end


!***********************************************************************
!     fnalu_hypre_structsetupmatvec
!***********************************************************************
      subroutine fnalu_hypre_structsetupmatvec(fmv)
      integer ierr
      integer*8 fmv

      call NALU_HYPRE_StructSetupMatvec(fmv, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsetupmatvec: err = ', ierr
      endif

      return
      end




!***********************************************************************
!             NALU_HYPRE_StructJacobi routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structjacobicreate
!***********************************************************************
      subroutine fnalu_hypre_structjacobicreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobicreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobidestroy
!***********************************************************************
      subroutine fnalu_hypre_structjacobidestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobidestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisetup
!***********************************************************************
      subroutine fnalu_hypre_structjacobisetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructJacobiSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisolve
!***********************************************************************
      subroutine fnalu_hypre_structjacobisolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructJacobiSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisettol
!***********************************************************************
      subroutine fnalu_hypre_structjacobisettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructJacobiSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobigettol
!***********************************************************************
      subroutine fnalu_hypre_structjacobigettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructJacobiGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structjacobisetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobigetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structjacobigetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structjacobisetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobigetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structjacobigetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobisetnonzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structjacobisetnonzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiSetNonZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobisetnonzerogu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobigetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structjacobigetnumiterat(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call NALU_HYPRE_StructJacobiGetNumIterati(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigetnumiterat: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structjacobigetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structjacobigetfinalrela(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructJacobiGetFinalRelat(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigetfinalrela: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructPCG routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structpcgcreate
!***********************************************************************
      subroutine fnalu_hypre_structpcgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructPCGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgdestroy
!***********************************************************************
      subroutine fnalu_hypre_structpcgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructPCGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetup
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructPCGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsolve
!***********************************************************************
      subroutine fnalu_hypre_structpcgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructPCGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsettol
!***********************************************************************
      subroutine fnalu_hypre_structpcgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructPCGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructPCGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsettwonorm
!***********************************************************************
      subroutine fnalu_hypre_structpcgsettwonorm(fsolver, ftwonorm)
      integer ierr
      integer ftwonorm
      integer*8 fsolver

      call NALU_HYPRE_StructPCGSetTwoNorm(fsolver, ftwonorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsettwonorm: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetrelchange(fsolver, frelchng)
      integer ierr
      integer frelchng
      integer*8 fsolver

      call NALU_HYPRE_StructPCGSetRelChange(fsolver, frelchng, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetprecond
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetprecond(fsolver, fprecond_id, 
     1                                      fprecond)
      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_StructPCGSetPrecond(fsolver, fprecond_id, fprecond,
     1                               ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetprecond: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetlogging(fsolver, flogging) 
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructPCGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcgsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structpcgsetprintlevel(fsolver, fprntlvl) 
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_StructPCGSetPrintLevel(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcggetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structpcggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call NALU_HYPRE_StructPCGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpcggetnumiteration: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpcggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structpcggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructPCGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structjacobigetfinalrelativ: err = ', ierr
      endif

      return
      end



!***********************************************************************
!     fnalu_hypre_structdiagscalesetup
!***********************************************************************
      subroutine fnalu_hypre_structdiagscalesetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructDiagScaleSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structdiagscalesetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structdiagscale
!***********************************************************************
      subroutine fnalu_hypre_structdiagscale(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructDiagScale(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structdiagscale: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructPFMG routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structpfmgcreate
!***********************************************************************
      subroutine fnalu_hypre_structpfmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgdestroy
!***********************************************************************
      subroutine fnalu_hypre_structpfmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetup
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructPFMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsolve
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructPFMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsettol
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructPFMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggettol
!***********************************************************************
      subroutine fnalu_hypre_structpfmggettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructPFMGGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetmaxlevels
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetmaxlevels(fsolver, fmaxlevels)
      integer ierr
      integer fmaxlevels
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetMaxLevels(fsolver, fmaxlevels, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetmaxlevels: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetmaxlevels
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetmaxlevels(fsolver, fmaxlevels)
      integer ierr
      integer fmaxlevels
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetMaxLevels(fsolver, fmaxlevels, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetmaxlevels: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetnonzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetnonzerogues(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetnonzerogues: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetnumiteratio(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetNumIteration(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetnumiteratio: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetfinalrelati(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructPFMGGetFinalRelativ(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetfinalrelati: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetskiprelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetskiprelax(fsolver, fskiprelax)
      integer ierr
      integer fskiprelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetSkipRelax(fsolver, fskiprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetskiprelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetskiprelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetskiprelax(fsolver, fskiprelax)
      integer ierr
      integer fskiprelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetSkipRelax(fsolver, fskiprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetskiprelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetrelaxtype
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetrelaxtype(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetRelaxType(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetrelaxtype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetrelaxtype
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetrelaxtype(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetRelaxType(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetrelaxtype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetraptype
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetraptype(fsolver, fraptype)
      integer ierr
      integer fraptype
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetRAPType(fsolver, fraptype, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetraptype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetraptype
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetraptype(fsolver, fraptype)
      integer ierr
      integer fraptype
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetRAPType(fsolver, fraptype, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetraptype: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetnumprerelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetnumprerelax(fsolver,
     1                                             fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetnumprerelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetnumprerelax(fsolver,
     1                                             fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetnumpostrelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetnumpostrela(fsolver,
     1                                             fnumpostrelax)
      integer ierr
      integer fnumpostrelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetNumPostRelax(fsolver, fnumpostrelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetnumpostrela: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetnumpostrelax
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetnumpostrela(fsolver,
     1                                             fnumpostrelax)
      integer ierr
      integer fnumpostrelax
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetNumPostRelax(fsolver, fnumpostrelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetnumpostrela: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetdxyz
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetdxyz(fsolver, fdxyz)
      integer ierr
      integer*8 fsolver
      double precision fdxyz

      call NALU_HYPRE_StructPFMGSetDxyz(fsolver, fdxyz, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetdxyz: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetlogging
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmgsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structpfmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structpfmggetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structpfmggetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_StructPFMGGetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structpfmggetprintlevel: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructSMG routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structsmgcreate
!***********************************************************************
      subroutine fnalu_hypre_structsmgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructSMGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgdestroy
!***********************************************************************
      subroutine fnalu_hypre_structsmgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSMGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetup
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructSMGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsolve
!***********************************************************************
      subroutine fnalu_hypre_structsmgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructSMGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetmemoryuse
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetmemoryuse(fsolver, fmemuse)
      integer ierr
      integer fmemuse
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetMemoryUse(fsolver, fmemuse, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetmemoryuse: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetmemoryuse
!***********************************************************************
      subroutine fnalu_hypre_structsmggetmemoryuse(fsolver, fmemuse)
      integer ierr
      integer fmemuse
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetMemoryUse(fsolver, fmemuse, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetmemoryuse: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsettol
!***********************************************************************
      subroutine fnalu_hypre_structsmgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructSMGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggettol
!***********************************************************************
      subroutine fnalu_hypre_structsmggettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructSMGGetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structsmggetmaxiter(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetmaxiter: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structsmggetrelchange(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetRelChange(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetrelchange: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structsmggetzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetnonzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetnonzeroguess(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetNonZeroGuess(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetnonzeroguess: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structsmggetnumiteration(fsolver, fnumiters)
      integer ierr
      integer fnumiters
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetNumIterations(fsolver, fnumiters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetnumiteration: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structsmggetfinalrelativ(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructSMGGetFinalRelative(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetfinalrelativ: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetnumprerelax
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetnumprerelax(fsolver, fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetnumprerelax
!***********************************************************************
      subroutine fnalu_hypre_structsmggetnumprerelax(fsolver, fnumprerelax)
      integer ierr
      integer fnumprerelax
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetNumPreRelax(fsolver, fnumprerelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetnumprerelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetnumpostrelax
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetnumpostrelax(fsolver, fnumpstrlx)
      integer ierr
      integer fnumpstrlx
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetNumPostRelax(fsolver, fnumpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetnumpostrelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetnumpostrelax
!***********************************************************************
      subroutine fnalu_hypre_structsmggetnumpostrelax(fsolver, fnumpstrlx)
      integer ierr
      integer fnumpstrlx
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetNumPostRelax(fsolver, fnumpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetnumpostrelax: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetlogging
!***********************************************************************
      subroutine fnalu_hypre_structsmggetlogging(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetlogging: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmgsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structsmgsetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_StructSMGSetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmgsetprintlevel: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsmggetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structsmggetprintlevel(fsolver, fprintlevel)
      integer ierr
      integer fprintlevel
      integer*8 fsolver

      call NALU_HYPRE_StructSMGGetPrintLevel(fsolver, fprintlevel, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsmggetprintlevel: err = ', ierr
      endif

      return
      end





!***********************************************************************
!             NALU_HYPRE_StructSparseMSG routines
!***********************************************************************

!***********************************************************************
!     fnalu_hypre_structsparsemsgcreate
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgcreate(fcomm, fsolver)
      integer ierr
      integer fcomm
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGCreate(fcomm, fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgcreate: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgdestroy
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgdestroy(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGDestroy(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgdestroy: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetup
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetup(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructSparseMSGSetup(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetup: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsolve
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsolve(fsolver, fA, fb, fx)
      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_StructSparseMSGSolve(fsolver, fA, fb, fx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsolve: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetjump
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetjump(fsolver, fjump)
      integer ierr
      integer fjump
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetJump(fsolver, fjump, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetjump: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsettol
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsettol(fsolver, ftol)
      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_StructSparseMSGSetTol(fsolver, ftol, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsettol: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetmaxiter
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetmaxite(fsolver, fmaxiter)
      integer ierr
      integer fmaxiter
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetMaxIter(fsolver, fmaxiter, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetmaxite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetrelchange
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetrelcha(fsolver, frelchange)
      integer ierr
      integer frelchange
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetRelChan(fsolver, frelchange, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetrelcha: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetzerogu(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetZeroGue(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetzerogu: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetnonzeroguess
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetnonzer(fsolver)
      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetNonZero(fsolver, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetnonzer: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsggetnumiterations
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsggetnumite(fsolver, fniters)
      integer ierr
      integer fniters
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGGetNumIter(fsolver, fniters, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsggetnumite: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsggetfinalrelativeresidualnorm
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsggetfinalr(fsolver, fnorm)
      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_StructSparseMSGGetFinalRe(fsolver, fnorm, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsggetfinalr: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetrelaxtype
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetrelaxt(fsolver, frelaxtype)
      integer ierr
      integer frelaxtype
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetRelaxTy(fsolver, frelaxtype, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetrelaxt: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetnumprerelax
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetnumpre(fsolver, fnprelax)
      integer ierr
      integer fnprelax
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetNumPreR(fsolver, fnprelax, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetnumpre: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetnumpostrelax
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetnumpos(fsolver, fnpstrlx)
      integer ierr
      integer fnpstrlx
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetNumPost(fsolver, fnpstrlx, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetnumpos: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetnumfinerelax
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetnumfin(fsolver, fnfine)
      integer ierr
      integer fnfine
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetNumFine(fsolver, fnfine, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetnumfin: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetlogging
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetloggin(fsolver, flogging)
      integer ierr
      integer flogging
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetLogging(fsolver, flogging, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetloggin: err = ', ierr
      endif

      return
      end

!***********************************************************************
!     fnalu_hypre_structsparsemsgsetprintlevel
!***********************************************************************
      subroutine fnalu_hypre_structsparsemsgsetprintl(fsolver, fprntlvl)
      integer ierr
      integer fprntlvl
      integer*8 fsolver

      call NALU_HYPRE_StructSparseMSGSetPrintLe(fsolver, fprntlvl, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structsparsemsgsetprintl: err = ', ierr
      endif

      return
      end
