!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!****************************************************************************
! NALU_HYPRE_SStruct_ls fortran interface routines
!****************************************************************************


!****************************************************************************
!                NALU_HYPRE_SStructBiCGSTAB routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabdestroy(fsolver)

      integer ierr
      integer*8 fsolver

       call NALU_HYPRE_SStructBiCGSTABDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

       call NALU_HYPRE_SStructBiCGSTABSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSolve
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructBiCGSTABSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructBiCGSTABSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetMinIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetminite(fsolver, fmin_iter)

      integer ierr
      integer fmin_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABSetMinIter(fsolver, fmin_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetminiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetmaxite(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetStopCrit
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetstopcr(fsolver, fstop_crit)

      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABSetStopCri(fsolver, fstop_crit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetstopcrit error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetPrecond
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetprecon(fsolver, fprecond_id,
     1                                            fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_SStructBiCGSTABSetPrecond(fsolver, fprecond_id,
     1                                     fprecond, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetloggin(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabsetprintl(fsolver, fprint)

      integer ierr
      integer fprint
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABSetPrintLe(fsolver, fprint, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabgetnumite(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructBiCGSTABGetNumIter(fsolver, fnumiter, 
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabgetnumiterations error = ', 
     1                                          ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabgetfinalr(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructBiCGSTABGetFinalRe(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabgetfinalrelative error = ',
     1                                                             ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructBiCGSTABGetResidual
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructbicgstabgetresidu(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call NALU_HYPRE_SStructBiCGSTABGetResidua(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructbicgstabgetresidual error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructGMRES routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmrescreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmrescreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmresdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmresdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructGMRESSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSolve
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructGMRESSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetKDim
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetkdim(fsolver, fkdim)

      integer ierr
      integer fkdim
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetKDim(fsolver, fkdim, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetkdim error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructGMRESSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetMinIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetminiter(fsolver, fmin_iter)

      integer ierr
      integer fmin_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetMinIter(fsolver, fmin_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetminiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetStopCrit
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetstopcrit(fsolver, fstop_crit)

      integer ierr
      integer fstop_crit
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetStopCrit(fsolver, fstop_crit, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetstopcrit error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetPrecond
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetprecond(fsolver, fprecond_id, 
     1                                         fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_SStructGMRESSetPrecond(fsolver, fprecond_id, fprecond,
     1                                  ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmressetprintleve(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmressetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmresgetnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructGMRESGetNumIterati(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmresgetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmresgetfinalrela(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructGMRESGetFinalRelat(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmresgetfinalrelative error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGMRESGetResidual
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgmresgetresidual(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call NALU_HYPRE_SStructGMRESGetResidual(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructgmresgetresidual error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructInterpreter routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpvectorsetrandomv(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call NALU_HYPRE_SStructPVectorSetRandomVa(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpvectorsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorsetrandomva(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call NALU_HYPRE_SStructVectorSetRandomVal(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructvectorsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSetRandomValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsetrandomvalues(fsolver, fseed)

      integer ierr
      integer*8 fsolver
      integer*8 fseed

      call NALU_HYPRE_SStructSetRandomValues(fsolver, fseed, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsetrandomvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSetupInterpreter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsetupinterpreter(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSetupInterpreter(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsetupinterpreter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSetupMatvec
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsetupmatvec(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSetupMatvec(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsetupmatvec error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructFAC routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfaccreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructFACCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfaccreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACDestroy2
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacdestroy2(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructFACDestroy2(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacdestroy2 error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetup2
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetup2(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructFACSetup2(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetup2 error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSolve3
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsolve3(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructFACSolve3(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsolve3 error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      integer*8 ftol

      call NALU_HYPRE_SStructFACSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetPLevels
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetplevels(fsolver, fnparts, fplevels)

      integer ierr
      integer*8 fsolver
      integer*8 fnparts
      integer*8 fplevels

      call NALU_HYPRE_SStructFACSetPLevels(fsolver, fnparts, fplevels, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetplevels error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetPRefinements
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetprefinement(fsolver, fnparts,
     1                                            frfactors)

      integer ierr
      integer*8 fsolver
      integer*8 fnparts
      integer*8 frfactors(3)

      call NALU_HYPRE_SStructFACSetPRefinements(fsolver, fnparts, frfactors,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetprefinements error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetMaxLevels
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetmaxlevels(fsolver, fmaxlevels) 

      integer ierr
      integer*8 fsolver
      integer*8 fmaxlevels

      call NALU_HYPRE_SStructFACSetMaxLevels(fsolver, fmaxlevels, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetmaxlevels error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetmaxiter(fsolver, fmaxiter) 

      integer ierr
      integer*8 fsolver
      integer*8 fmaxiter

      call NALU_HYPRE_SStructFACSetMaxIter(fsolver, fmaxiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetRelChange
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetrelchange(fsolver, frelchange) 

      integer ierr
      integer*8 fsolver
      integer*8 frelchange

      call NALU_HYPRE_SStructFACSetRelChange(fsolver, frelchange, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetrelchange error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetzeroguess(fsolver) 

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructFACSetZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetNonZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetnonzerogues(fsolver) 

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructFACSetNonZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetnonzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetRelaxType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetrelaxtype(fsolver, frelaxtype) 

      integer ierr
      integer*8 fsolver
      integer*8 frelaxtype

      call NALU_HYPRE_SStructFACSetRelaxType(fsolver, frelaxtype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetrelaxtype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetNumPreRelax
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetnumprerelax(fsolver, fnumprerelax) 

      integer ierr
      integer*8 fsolver
      integer*8 fnumprerelax

      call NALU_HYPRE_SStructFACSetNumPreRelax(fsolver, fnumprerelax, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetnumprerelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetNumPostRelax
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetnumpostrela(fsolver,
     1                                            fnumpostrelax) 

      integer ierr
      integer*8 fsolver
      integer*8 fnumpostrelax

      call NALU_HYPRE_SStructFACSetNumPostRelax(fsolver, fnumpostrelax, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetnumpostrelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetCoarseSolverType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetcoarsesolve(fsolver,
     1                                            fcsolvertype) 

      integer ierr
      integer*8 fsolver
      integer*8 fcsolvertype

      call NALU_HYPRE_SStructFACSetCoarseSolver(fsolver, fcsolvertype,
     1                                      ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetcoarsesolvertype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_SStructFACSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacgetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructFACGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacgetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructFACGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructfacgetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructFACGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructfacgetfinalrelative error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructPCG routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructPCGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSolve
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructPCGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructPCGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetTwoNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsettwonorm(fsolver, ftwo_norm)

      integer ierr
      integer ftwo_norm
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGSetTwoNorm(fsolver, ftwo_norm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsettwonorm error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetRelChange
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetrelchange(fsolver, frel_change)

      integer ierr
      integer frel_change
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGSetRelChange(fsolver, frel_change, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetrelchange error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetPrecond
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetprecond(fsolver, fprecond_id,
     1                                       fprecond)

      integer ierr
      integer fprecond_id
      integer*8 fsolver
      integer*8 fprecond

      call NALU_HYPRE_SStructPCGSetPrecond(fsolver, fprecond_id, fprecond,
     1                                ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetprecond error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcgsetprintlevel(fsolver, flevel)

      integer ierr
      integer flevel
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGSetPrintLevel(fsolver, flevel, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcgsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcggetnumiteratio(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructPCGGetNumIteration(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcggetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcggetfinalrelati(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructPCGGetFinalRelativ(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcggetfinalrelative error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructPCGGetResidual
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructpcggetresidual(fsolver, fresidual)

      integer ierr
      integer*8 fsolver
      integer*8 fresidual

      call NALU_HYPRE_SStructPCGGetResidual(fsolver, fresidual, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructpcggetresidual error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructDiagScaleSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructdiagscalesetup(fsolver, fA, fy, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fy
      integer*8 fx

      call NALU_HYPRE_SStructDiagScaleSetup(fsolver, fA, fy, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructdiagscalesetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructDiagScale
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructdiagscale(fsolver, fA, fy, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fy
      integer*8 fx

      call NALU_HYPRE_SStructDiagScale(fsolver, fA, fy, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructdiagscale error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructSplit routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructSplitSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSolve
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructSplitSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructSplitSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsetzeroguess(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitSetZeroGuess(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsetzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetNonZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsetnonzerogu(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitSetNonZeroGue(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsetnonzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitSetStructSolver
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitsetstructsol(fsolver, fssolver)

      integer ierr
      integer fssolver
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitSetStructSolv(fsolver, fssolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitsetstructsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitgetnumiterat(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructSplitGetNumIterati(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitgetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsplitgetfinalrela(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructSplitGetFinalRelat(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsplitgetfinalrelative error = ', ierr
      endif

      return
      end





!****************************************************************************
!                NALU_HYPRE_SStructSYSPFMG routines
!****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgdestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetup
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetup(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructSysPFMGSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetup error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSolve
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsolve(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructSysPFMGSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsolve error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetTol
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsettol(fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructSysPFMGSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsettol error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetMaxIter
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetmaxiter(fsolver, fmax_iter)

      integer ierr
      integer fmax_iter
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetmaxiter error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetRelChange
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetrelchang(fsolver, frel_change)

      integer ierr
      integer frel_change
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetRelChang(fsolver, frel_change, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetrelchange error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetzerogue(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetZeroGues(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetNonZeroGuess
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetnonzero(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetNonZeroG(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetnonzeroguess error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetRelaxType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetrelaxty(fsolver, frelax_type)

      integer ierr
      integer frelax_type
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetRelaxTyp(fsolver, frelax_type, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetrelaxtype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetNumPreRelax
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetnumprer(fsolver, 
     1                                            fnum_pre_relax)

      integer ierr
      integer fnum_pre_relax
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetNumPreRe(fsolver, fnum_pre_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetnumprerelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetNumPostRelax
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetnumpost(fsolver, 
     1                                            fnum_post_relax)

      integer ierr
      integer fnum_post_relax
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetNumPostR(fsolver, fnum_post_relax,
     1                                        ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetnumpostrelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetSkipRelax
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetskiprel(fsolver, fskip_relax)

      integer ierr
      integer fskip_relax
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetSkipRela(fsolver, fskip_relax, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetskiprelax error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetDxyz
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetdxyz(fsolver, fdxyz)

      integer ierr
      integer*8 fsolver
      double precision fdxyz

      call NALU_HYPRE_SStructSysPFMGSetDxyz(fsolver, fdxyz, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetdxyz error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetLogging
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetlogging(fsolver, flog)

      integer ierr
      integer flog
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetLogging(fsolver, flog, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetlogging error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGSetPrintLevel
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmgsetprintle(fsolver, fprint_level)

      integer ierr
      integer fprint_level
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGSetPrintLev(fsolver, fprint_level,
     1                                       ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmgsetprintlevel error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGGetNumIterations
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmggetnumiter(fsolver, fnumiter)

      integer ierr
      integer fnumiter
      integer*8 fsolver

      call NALU_HYPRE_SStructSysPFMGGetNumItera(fsolver, fnumiter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmggetnumiteration error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructsyspfmggetfinalre(fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructSysPFMGGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructsyspfmggetfinalrelative error = ', ierr
      endif

      return
      end


!*****************************************************************************
!*
!* NALU_HYPRE_SStructMaxwell interface
!*
!*****************************************************************************

!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellCreate
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellcreate(fcomm, fsolver)

      integer ierr
      integer*8 fcomm
      integer*8 fsolver

      call NALU_HYPRE_SStructMaxwellCreate(fcomm, fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellcreate = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellDestroy
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwelldestroy(fsolver)

      integer ierr
      integer*8 fsolver

      call NALU_HYPRE_SStructMaxwellDestroy(fsolver, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwelldestroy = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetup
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetup (fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructMaxwellSetup(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetup = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSolve
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsolve (fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructMaxwellSolve(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsolve = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSolve2
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsolve2(fsolver, fA, fb, fx)

      integer ierr
      integer*8 fsolver
      integer*8 fA
      integer*8 fb
      integer*8 fx

      call NALU_HYPRE_SStructMaxwellSolve2(fsolver, fA, fb, fx, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsolve2 = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_MaxwellGrad
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_maxwellgrad (fgrid, fT)

      integer ierr
      integer*8 fgrid
      integer*8 fT

      call NALU_HYPRE_MaxwellGrad(fgrid, fT, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellgrad = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetGrad
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetgrad (fsolver, fT)

      integer ierr
      integer*8 fsolver
      integer*8 fT

      call NALU_HYPRE_SStructMaxwellSetGrad(fsolver, fT, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetgrad = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetRfactors
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetrfactor (fsolver,frfactors)

      integer ierr
      integer*8 fsolver
      integer*8 frfactors(3)

      call NALU_HYPRE_SStructMaxwellSetRfactors(fsolver, frfactors, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetrfactors = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetTol
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsettol (fsolver, ftol)

      integer ierr
      integer*8 fsolver
      double precision ftol

      call NALU_HYPRE_SStructMaxwellSetTol(fsolver, ftol, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsettol = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetConstantCoef
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetconstan (fsolver,
     1                                            fconstant_coef)

      integer ierr
      integer*8 fsolver
      integer fconstant_coef

      call NALU_HYPRE_SStructMaxwellSetConstant(fsolver, fconstant_coef,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetconstantcoef = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetMaxIter
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetmaxiter (fsolver, fmax_iter)

      integer ierr
      integer*8 fsolver
      integer fmax_iter

      call NALU_HYPRE_SStructMaxwellSetMaxIter(fsolver, fmax_iter, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetmaxiter = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetRelChange
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetrelchan (fsolver, frel_change)

      integer ierr
      integer*8 fsolver
      integer frel_change

      call NALU_HYPRE_SStructMaxwellSetRelChang(fsolver, frel_change, ierr) 

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetrelchange = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetNumPreRelax
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetnumprer (fsolver, 
     1                                            fnum_pre_relax)

      integer ierr
      integer*8 fsolver
      integer fnum_pre_relax

      call NALU_HYPRE_SStructMaxwellSetNumPreRe(fsolver, fnum_pre_relax, 
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetnumprerelax = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetNumPostRelax
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetnumpost (fsolver, 
     1                                            fnum_post_relax)

      integer ierr
      integer*8 fsolver
      integer fnum_post_relax

      call NALU_HYPRE_SStructMaxwellSetNumPostR(fsolver, fnum_post_relax,
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetnumpostrelax = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetLogging
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetlogging (fsolver, flogging)

      integer ierr
      integer*8 fsolver
      integer flogging

      call NALU_HYPRE_SStructMaxwellSetLogging(fsolver, flogging, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetlogging = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellSetPrintLevel
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellsetprintle (fsolver, fprint_level)

      integer ierr
      integer*8 fsolver
      integer flogging

      call NALU_HYPRE_SStructMaxwellSetPrintLev(fsolver, fprint_level, 
     1                                     ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellsetprintlevel = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellPrintLogging
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellprintloggi (fsolver, fmyid)

      integer ierr
      integer*8 fsolver
      integer flogging

      call NALU_HYPRE_SStructMaxwellPrintLoggin(fsolver, fmyid, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellprintlogging = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellGetNumIterations
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellgetnumiter (fsolver, 
     1                                            fnum_iterations)

      integer ierr
      integer*8 fsolver
      integer fnum_iterations

      call NALU_HYPRE_SStructMaxwellGetNumItera(fsolver, 
     1                                     fnum_iterations, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellgetnumiterations = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellgetfinalre (fsolver, fnorm)

      integer ierr
      integer*8 fsolver
      double precision fnorm

      call NALU_HYPRE_SStructMaxwellGetFinalRel(fsolver, fnorm, ierr)

      if(ierr .ne. 0) then
         print *, 
     1      'fnalu_hypre_sstructmaxwellgetfinalrelativeresidualnorm = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellPhysBdy
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellphysbdy (fgrid_l, fnum_levels,
     1                                         frfactors, 
     2                                         fBdryRanks_ptr,
     3                                         fBdryRanksCnt_ptr)

      integer ierr
      integer*8 fgrid_l
      integer*8 frfactors
      integer*8 fBdryRanks_ptr
      integer*8 fBdryRanksCnt_ptr
      integer fnum_levels

      call NALU_HYPRE_SStructMaxwellPhysBdy(fgrid_l, fnum_levels, frfactors,
     1                                 fBdryRanks_ptr, 
     2                                 fBdryRanksCnt_ptr, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellphysbdy = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellEliminateRowsCols
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwelleliminater (fparA, fnrows, frows)

      integer ierr
      integer*8 fparA
      integer*8 frows
      integer*8 fnrows

      call NALU_HYPRE_SStructMaxwellEliminateRo(fparA, fnrows, frows, 
     1                                         ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwelleliminaterows = ', ierr
      endif

      return
      end


!*--------------------------------------------------------------------------
!* NALU_HYPRE_SStructMaxwellZeroVector
!*--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmaxwellzerovector (fb, frows, fnrows)

      integer ierr
      integer*8 fb
      integer*8 frows
      integer*8 fnrows

      call NALU_HYPRE_SStructMaxwellZeroVector(fb, frows, fnrows, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_sstructmaxwellzerovector = ', ierr
      endif

      return
      end
