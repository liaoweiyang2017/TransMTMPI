!===============================================================================
! compTEM1d get TEM response excited by a rectange loop
!
!===============================================================================
subroutine compTEM1d(xLen, yLen, nlayer, rho, thickness, nTimePoint, timePoint, timeResp)

      use tem1dmod
      implicit none

      integer,intent(in):: nTimePoint, nlayer
      real(kind=8),intent(in):: xLen, yLen
      real(kind=8),dimension(nlayer),intent(in) :: rho, thickness
      real(kind=8),dimension(nTimePoint),intent(in):: timePoint
      real(kind=8),dimension(nTimePoint),intent(inout):: timeResp

      ! initialize stuff
      call initTEMModel(xLen, yLen)

      ! calculate frequency response
      call compFreqResponse(nlayer, rho, thickness)

      ! transform response from frequency domain to time domain
      call temsineTransform(ntemFreq,temFreq,temfreqResp,nTimePoint,timePoint,timeResp)

      ! deallocate arrays
      call deallocateTEMFwd

end subroutine


!===============================================================================
! compTEM1d get TEM response excited by a rectange loop
!
!===============================================================================
subroutine compTEM1dResponse(xLen, yLen, nlayer, rho, thickness, nTimePoint, &
                            timePoint, timeResp)

      use tem1dmod
      implicit none

      integer,intent(in):: nTimePoint, nlayer
      real(kind=8),intent(in):: xLen, yLen
      real(kind=8),dimension(nTimePoint),intent(in):: timePoint
      real(kind=8),dimension(nlayer),intent(in):: rho, thickness
      real(kind=8),dimension(nTimePoint),intent(inout):: timeResp

      ! initialize stuff
      call initTEMModel(xLen, yLen)

      ! calculate frequency response
      call compCentralFreqResponse(nlayer, rho, thickness)

      ! transform response from frequency domain to time domain
      call temsineTransform(ntemFreq,temFreq,temfreqResp,nTimePoint,timePoint,timeResp)

      ! deallocate arrays
      call deallocateTEMFwd

end subroutine


subroutine compTEM1dResponseNew(xLen, yLen, nlayer, rho, thickness, nTimePoint, &
                            timePoint, timeResp)

      use tem1dmod
      implicit none

      integer,intent(in):: nTimePoint, nlayer
      real(kind=8),intent(in):: xLen, yLen
      real(kind=8),dimension(nTimePoint),intent(in):: timePoint
      real(kind=8),dimension(nlayer),intent(in):: rho, thickness
      real(kind=8),dimension(nTimePoint),intent(inout):: timeResp

      ! initialize stuff
      call initTEMModel(xLen, yLen)

      ! calculate frequency response
      call compFreqResponseDipole(nLayer, rho, thickness)

      ! transform response from frequency domain to time domain
      call temsineTransform(ntemFreq,temFreq,temfreqResp,nTimePoint,timePoint,timeResp)

      ! deallocate arrays
      call deallocateTEMFwd

end subroutine
