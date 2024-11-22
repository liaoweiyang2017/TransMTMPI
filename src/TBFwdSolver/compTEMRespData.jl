export compTEM1dResponse
#-------------------------------------------------------------------------------
"""
    `compTEM1dResponse(xLen, yLen, timePoint, rho, depth1D)`

"""
function compTEM1dResponse(loopxLen::T, loopyLen::T, timePoint::Vector{T},
                   rho::Vector{T}, depth1D::Vector{T}) where{T<:Float64}

    #
    if length(rho) != length(depth1D) - 1
        error("compTEM1d: resistivity and thickness must be the same length!")
    end

    nt     = length(timePoint)
    nlayer = length(rho)
    thickness = diff(depth1D)
    thickness[nlayer] = 10000.0

    # predicted tem response
    timeResp  = zeros(nt)

    #
    ccall((:comptem1dresponsenew_, em1dLibpath), Nothing, (Ptr{Float64},Ptr{Float64},
        Ptr{Int64},Ptr{Float64},Ptr{Float64},Ptr{Int64},Ptr{Float64},Ptr{Float64}),
        Ref(loopxLen), Ref(loopyLen), Ref(nlayer), rho, thickness, Ref(nt), timePoint, timeResp)

    return timeResp

end
