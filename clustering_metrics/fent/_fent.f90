! F2PY experiment
! http://stackoverflow.com/a/12200671/597371

subroutine minmaxr(a,n,amin,amax)
    implicit none
    !f2py intent(hidden) :: n
    !f2py intent(out) :: amin,amax
    !f2py intent(in) :: a
    integer n, i
    real a(n),amin,amax,acurr
    real :: x = 0

    if (n > 0) then
        amin = a(1)
        amax = a(1)
        do i=2, n
            acurr = a(i)
            if (acurr > amax) then
                amax = acurr
            elseif (acurr < amin) then
                amin = acurr
            endif
        enddo
    else
        ! set return values to (inf, -inf)
        amin = -log(x)
        amax = -amin
    endif
end subroutine minmaxr

! Calculate Expected Mutual Information given margins of RxC table
!
! For the sake of numeric precision, the resulting value is *not* normalized
! by N.
!
! License: BSD 3 clause
!
! .. codeauthor:: Robert Layton <robertlayton@gmail.com>
! .. codeauthor:: Corey Lynch <coreylynch9@gmail.com>
! .. codeauthor:: Eugene Scherba <escherba@gmail.com>
!
! List of changes (Eugene Scherba, 10/2/2015):
!
! 1) Removed/rewritten the lines that were creating RxC intermediate NumPy
! arrays which resulted in the O(n^2) memory requirement. Instead, the
! intermediate values are now calculated inside the loop, which may be
! slightly less efficient for small data sizes, but has huge advantages for
! large or even moderately sized data. This change reduces the memory
! requirements of this code from O(n^2) to O(n).
!
! 2) Removed normalization by N from the calculation. It is actually not
! needed to normalize by N if we also don't normalize the input MI value
! (in the calculation of the adjusted score which is ``MI - E(MI) / MI_max
! - E(MI)``, the N value cancels out). Not normalizing the EMI calculation
! by N avoids having to perform lots of tiny floating point increments to
! the EMI aggregate value and thus improves numeric accuracy, especially
! for small values of EMI.
!
! 3) A Fortran 90 version
!
subroutine emi_from_margins(a,R,b,C,emi)
    implicit none
    !f2py intent(hidden) :: R, C
    !f2py intent(out) :: emi
    !f2py intent(in) :: a, b
    integer(kind=8) R, C, i, j, nij, N, N1, N3, max_ab, ai_1, bj_1, N3_ai_1, N3_ai_bj_1

    real(kind=8) emi, log_ai, log_ab_outer_ij, outer_sum, gln_ai_Nai_Ni
    real(kind=8) log_a(R), log_b(C), gln_ai_Nai_N(R), gln_b_Nb(C)

    integer(kind=8) a(R), b(C), a1(R), b1(C)

    real(kind=8), dimension(:), allocatable :: nijs, gln_nij, log_Nnij
    real(kind=8) :: x = -1.0

    log_a = dlog(dble(a))
    log_b = dlog(dble(b))

    N = sum(a)
    if (N /= sum(b)) then
        ! set return value to nan
        emi = dsqrt(x)
        return
    endif

    ! There are three major terms to the EMI equation, which are multiplied to
    ! and then summed over varying nij values.

    ! term1 is nijs.
    ! While nijs[0] will never be used, having it simplifies the indexing.
    max_ab = max(maxval(a), maxval(b))

    allocate(nijs(max_ab + 1))
    do nij=1, max_ab + 1
        nijs(nij) = nij - 1
    enddo

    nijs(1) = 1.0  ! Stops divide by zero warnings. As its not used, no issue.

    ! term2 is log((N*nij) / (a a b)) == log(N * nij) - log(a * b)
    ! term2 uses log(N * nij)
    allocate(log_Nnij(max_ab + 1))
    log_Nnij = dlog(dble(N)) + dlog(nijs)

    ! term3 is large, and involved many factorials. Calculate these in log
    ! space to stop overflows.
    N1 = N + 1
    N3 = N + 3

    a1 = a + 1
    b1 = b + 1
    gln_ai_Nai_N = dlgama(dble(a1)) + dlgama(dble(N1 - a)) - dlgama(dble(N1))
    gln_b_Nb = dlgama(dble(b1)) + dlgama(dble(N1 - b))

    allocate(gln_nij(max_ab + 1))
    gln_nij = dlgama(nijs + 1.0)

    ! emi itself is a summation over the various values.
    emi = 0.0
    do i=1, R
        ai_1 = a1(i)
        log_ai = log_a(i)
        gln_ai_Nai_Ni = gln_ai_Nai_N(i)
        N3_ai_1 = N3 - ai_1
        do j=1, C
            bj_1 = b1(j)
            log_ab_outer_ij = log_ai + log_b(j)
            outer_sum = gln_ai_Nai_Ni + gln_b_Nb(j)
            N3_ai_bj_1 = N3_ai_1 - bj_1

            do nij=max(1, 1 - N3_ai_bj_1), min(ai_1, bj_1) - 1
                ! Numerators are positive, denominators are negative.
                emi = emi + ( &
                    & dble(nij) * &
                    & (log_Nnij(nij + 1) - log_ab_outer_ij) * &
                    & dexp(outer_sum &
                    &    - gln_nij(nij + 1) &
                    &    - dlgama(dble(ai_1 - nij)) &
                    &    - dlgama(dble(bj_1 - nij)) &
                    &    - dlgama(dble(nij + N3_ai_bj_1)) &
                    & ))
            enddo
        enddo
    enddo

    deallocate(gln_nij)
    deallocate(log_Nnij)
    deallocate(nijs)
end subroutine emi_from_margins
