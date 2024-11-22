#-------------------------------------------------------------------------------
#
# module `TBFileIO` defines routines for prior information and EM data IO.
#
#
#-------------------------------------------------------------------------------
module TBFileIO

using TransdEM.TBStruct
using Printf, SparseArrays

#
export TEMData, CSEMData, MTData
export EMData
abstract type EMData end

#-------------------------------------------------------------------------------
"""
struct `TEMData` encapsalates TEM data structure.

"""
mutable struct TEMData{T<:Real} <: EMData
    #
    loopxLen::T
    loopyLen::T
    timePoint::Vector{T}
    obsData::Vector{T}
    dataErr::Vector{T}

end


#-------------------------------------------------------------------------------
"""
struct `CSEMData` encapsalates CSEM data structure.

"""
mutable struct CSEMData{T<:Real} <: EMData

    #
    seawater::Vector{T}                     # [thickness,sigma] of seawater
    txLoc::Array{T}                         # source location
    rxLoc::Array{T}                         # receiver location
    dpLen::T                                # dipole length
    freqArray::Vector{T}                    # frequency array
    dataType::Vector{String}                # data type
    dataID::Vector{Bool}                    # data index
    #
    obsData::Vector{T}
    dataErr::Vector{T}

end


#-------------------------------------------------------------------------------
"""
struct `MTData` encapsalates MT data structure.

"""
mutable struct MTData{T<:Real} <: EMData

    #
    freqArray::Vector{T}                    # frequency array
    dataType::Vector{String}                # data type
    dataID::Vector{Bool}                    # data index
    #
    obsData::Vector{T}
    dataErr::Vector{T}

end

#-------------------------------------------------------------------------------
include("readEMData.jl")
include("writeEMData.jl")
include("readstartupFile.jl")

end # module
