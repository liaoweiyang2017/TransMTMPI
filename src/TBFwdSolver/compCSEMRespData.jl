export comp1DCSEM, compWEM1dResp, compCSEM1dResp
export unwrap, unwrap!
#-------------------------------------------------------------------------------
"""
    `comp1DCSEM(txLoc, rxLoc, freqArray, sigma, depth1D, compFlag)`

Calculate specfic component of EM field due to electric dipole.

# Inputs:
 - `txLoc`     =::Array[nsx5], transmitter location
 - `rxLoc`     =::Array[nrx3], receiver location
 - `freqArray` =::Array, transmission frequencies
 - `sigma`     =::Vector, conductivities of layered model
 - `depth1D`   =::Vector, depths of top of layered model
 - `compFlag`  =::Integer, component identifier

# Outputs:
 - `e1d` =::Array, computed field

"""
function comp1DCSEM(txLoc::Array{T,2}, rxLoc::Array{T,2},
                    freqArray::Array{T}, sigma::Vector{T},
                    depth1D::Vector{T}, compFlag::Integer) where{T<:Real}

    # transmitter and receivers
    nTx = size(txLoc, 1)
    nRx = size(rxLoc, 1)
    nFreq  = length(freqArray)
    nLayer = length(sigma)

    if length(sigma) != length(depth1D)
        error("comp1DCSEM: Conductivity and depth must be the same length!")
    end

    # predicted field component
    e1d = zeros(ComplexF64, nRx, nTx, nFreq)

    #
    ccall((:calldipole1deb_, em1dLibpath), Nothing, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                    Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                    Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}), txLoc, rxLoc, freqArray,
                    sigma, depth1D, Ref(nTx), Ref(nRx), Ref(nFreq), Ref(nLayer), Ref(compFlag), e1d)

    return e1d

end # comp1DCSEM


#-------------------------------------------------------------------------------
"""
    `comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freqArray, sigma, depth1D, compFlag)`

Calculate specific component of EM field due to finite wire source.

# Inputs:
 - `txLoc`   =::Array[nsx5], transmitter location
 - `rxLoc`   =::Array[nrx3], receiver location
 - `dipLen`  =::Float64, length of finite dipole source
 - `nIntPts` =::Integer, Number of points to use for Gauss quadrature
                integration for finite dipole
 - `freqArray` =::Array, transmission frequencies
 - `sigma`     =::Array, conductivities of layered model
 - `depth1D`   =::Array, depth of top of layered model
 - `compFlag`  =::Integer, component identifier

# Outputs:
 - `e1d` =::Array, computed EM field

"""
function comp1DCSEM(txLoc::Array{T}, rxLoc::Array{T}, dipLen::T,
                    nIntPts::Integer, freqArray::Array{T},
                    sigma::Vector{T}, depth1D::Vector{T},
                    compFlag::Integer) where{T<:Real}

    # transmitter and receivers
    nTx = size(txLoc, 1)
    nRx = size(rxLoc, 1)
    nFreq  = length(freqArray)
    nLayer = length(sigma)

    if length(sigma) != length(depth1D)
        error("comp1DCSEM: Conductivity and depth must be the same length!")
    end

    # predicted field component
    e1d = zeros(ComplexF64, nRx, nTx, nFreq)

    #
    ccall((:calldipole1dfinite_, em1dLibpath), Nothing, (Ptr{Float64}, Ptr{Float64},
                        Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64},
                        Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                        Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}), txLoc, rxLoc,
                        Ref(dipLen), Ref(nIntPts), freqArray, sigma, depth1D, Ref(nTx), Ref(nRx),
                        Ref(nFreq), Ref(nLayer), Ref(compFlag), e1d)

    return e1d

end # comp1DCSEM


#-------------------------------------------------------------------------------
"""
    compCSEM1dResp(emData, sigma1D, depth1D)

calculates primary electromagnetic fields at receivers.

"""
function compCSEM1dResp(emData::CSEMData, sigma1D::Array, depth1D::Array)

    #
    txLoc = copy(emData.txLoc)
    rxLoc = emData.rxLoc
    freq = emData.freqArray

    # data index
    dataType = emData.dataType
    dataID   = emData.dataID

    ns  = size(emData.txLoc, 1)
    nr  = size(emData.rxLoc, 1)
    nDT = length(dataType)
    nFreq = length(freq)
    if nFreq == 1; freq = collect(freq); end

    #
    exp = zeros(ComplexF64, 0)
    eyp = zeros(ComplexF64, 0)
    ezp = zeros(ComplexF64, 0)
    bxp = zeros(ComplexF64, 0)
    byp = zeros(ComplexF64, 0)
    bzp = zeros(ComplexF64, 0)
    dipLen = emData.dpLen
    predData = zeros(nDT, nr, ns, nFreq)
    if dipLen < 100.0
        for i = 1:nDT
            dt = dataType[i]
            if occursin("ampEx", dt)
                if isempty(exp)
                    exp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 1)
                end
                predData[i,:,:,:] = abs.(exp)

            elseif occursin("phsEx", dt)
                if isempty(exp)
                    exp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 1)
                end
                predData[i,:,:,:] = atand.(imag(exp), real(exp))

            elseif occursin("ampEy", dt)
                if isempty(eyp)
                    eyp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 2)
                end
                predData[i,:,:,:] = abs.(eyp)

            elseif occursin("phsEy", dt)
                if isempty(eyp)
                    eyp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 2)
                end
                predData[i,:,:,:] = atand.(imag(eyp), real(eyp))

            elseif occursin("ampBx", dt)
                if isempty(bxp)
                    bxp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 4)
                end
                predData[i,:,:,:] = abs.(bxp)

            elseif occursin("phsBx", dt)
                if isempty(bxp)
                    bxp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 4)
                end
                predData[i,:,:,:] = atand.(imag(bxp), real(bxp))

            elseif occursin("ampBy", dt)
                if isempty(byp)
                    byp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 5)
                end
                predData[i,:,:,:] = abs.(byp)

            elseif occursin("phsBy", dt)
                if isempty(byp)
                    byp = comp1DCSEM(txLoc, rxLoc, freq, sigma1D, depth1D, 5)
                end
                predData[i,:,:,:] = atand.(imag(byp), real(byp))

            end
        end
    else
        printstyled("Finite length source is used!\n", color=:cyan)
        nIntPts = 15
        for i = 1:nDT
            dt = dataType[i]
            if occursin("ampEx", dt)
                if isempty(exp)
                    exp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 1)
                end
                predData[i,:,:,:] = abs.(exp)

            elseif occursin("phsEx", dt)
                if isempty(exp)
                    exp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 1)
                end
                predData[i,:,:,:] = atand.(imag(exp), real(exp))

            elseif occursin("ampEy", dt)
                if isempty(eyp)
                    eyp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 2)
                end
                predData[i,:,:,:] = abs.(eyp)

            elseif occursin("phsEy", dt)
                if isempty(eyp)
                    eyp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 2)
                end
                predData[i,:,:,:] = atand.(imag(eyp), real(eyp))

            elseif occursin("ampBx", dt)
                if isempty(bxp)
                    bxp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 4)
                end
                predData[i,:,:,:] = abs.(bxp)

            elseif occursin("phsBx", dt)
                if isempty(bxp)
                    bxp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 4)
                end
                predData[i,:,:,:] = atand.(imag(bxp), real(bxp))

            elseif occursin("ampBy", dt)
                if isempty(byp)
                    byp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 5)
                end
                predData[i,:,:,:] = abs.(byp)

            elseif occursin("phsBy", dt)
                if isempty(byp)
                    byp = comp1DCSEM(txLoc, rxLoc, dipLen, nIntPts, freq, sigma1D, depth1D, 5)
                end
                predData[i,:,:,:] = atand.(imag(byp), real(byp))

            end
        end
    end

    predData = vec(predData)
    predData = predData[dataID]

    return predData

end # getPrimaryRespField


#-------------------------------------------------------------------------------
# Julia implementation of a phase unwrap function
# by Spencer Russell (https://gist.github.com/ssfrr/7995008)
#
#-------------------------------------------------------------------------------
function unwrap(v, inplace=false)
  # currently assuming an array
  unwrapped = inplace ? v : copy(v)
  for i in 2:length(v)
    while unwrapped[i] - unwrapped[i-1] >= pi
      unwrapped[i] -= 2pi
    end
    while unwrapped[i] - unwrapped[i-1] <= -pi
      unwrapped[i] += 2pi
    end
  end
  return unwrapped
end

unwrap!(v) = unwrap(v, true)
