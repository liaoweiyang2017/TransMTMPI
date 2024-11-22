#-------------------------------------------------------------------------------
#
# module `TBUtility` defines utility routines for transdimensional Bayesian
# inversion.
#
#-------------------------------------------------------------------------------
module TBUtility

using Random

export unirandInteger
export unirandDouble
export findLocation1D, findNearest1D
export getGaussianProbability

#-------------------------------------------------------------------------------
"""
    `unirandInteger(vmin, vmax)`

returns a random integer between `vmin` and `vmax`

"""
function unirandInteger(vmin::Int, vmax::Int)

    rval = rand(vmin:vmax)
    return rval

end


#-------------------------------------------------------------------------------
"""
    `randDouble(vmin, vmax)`

returns a random float number between `vmin` and `vmax`

"""
function unirandDouble(vmin::Float64, vmax::Float64)

    rval = vmin + (vmax - vmin) * rand()
    return rval

end


#-------------------------------------------------------------------------------
"""
    `findLocation1D(point, x)`

"""
function findLocation1D(point::T, x::Vector{T}) where {T}

    (val, idx) = findmin(abs.(x .- point))

    return idx

end


#-------------------------------------------------------------------------------
"""
    `findNearest1D(point, x)`

"""
function findNearest1D(point::T, x::Vector{T}) where {T}

    # first sort x
    idxrank = sortperm(x)
    y   = x[idxrank]
    (val, idx) = findmin(abs.(y .- point))
    val = y[idx]
    if (val<point) & (y[1]<point<y[end])
        idx = idx + 1
    end

    return idxrank[idx]

end


#-------------------------------------------------------------------------------
"""
    `getGaussianProbability(sigma, phi)`

"""
function getGaussianProbability(sigma::Float64, phi::Float64)

    #
    alpha = exp(-phi^2 / (2 * sigma^2) ) / (sqrt(2*pi) * sigma)
    return alpha
end


end # TBUtility
