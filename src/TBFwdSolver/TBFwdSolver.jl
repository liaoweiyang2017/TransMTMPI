#-------------------------------------------------------------------------------
#
# module `TBFwdSolver` defines routines to perform forward modeling.
#
# (c) Ronghua Peng and Bo Han, China University of Geosciences, Wuhan, 2020-2022.
#
#-------------------------------------------------------------------------------
module TBFwdSolver

using TransdEM.TBStruct
using TransdEM.TBFileIO

export getModelParameter
export getPredData

#------------------------------------------------------------------------------
if Sys.isunix()
    libName = "em1dmod.so"
elseif Sys.iswindows()
    libName = "em1dmod.dll"
end

const em1dLibpath = abspath(joinpath(splitdir(Base.source_path())[1], ".",
                    "deps", "lib", libName))

#-------------------------------------------------------------------------------
function getPredData(emData::MTData, mcParam::MCParameter)

    # get model parameter
    dep1D, rho1D = getModelParameter(mcParam)
    sig1D = 1 ./ rho1D

    predData = compMT1dResp(emData, sig1D, dep1D)

    return predData

end

#-------------------------------------------------------------------------------
function getPredData(emData::CSEMData, mcParam::MCParameter)

    # get model parameter
    zNode, rho1D = getModelParameter(mcParam)
    sig1D = 1 ./ rho1D

    sigAir   = 1e-12
    airLayer = -1000000.
    if isempty(emData.seawater)
        sig1D = vcat(sigAir, sig1D)
        dep1D = vcat(airLayer, zNode[1:end-1])
    else
        seaDep = emData.seawater[1]
        sigSea = emData.seawater[2]
        sig1D  = vcat(sigAir, sigSea, sig1D)
        dep1D  = vcat(airLayer, 0.0, zNode[1:end-1] .+ seaDep)
    end
    predData = compCSEM1dResp(emData, sig1D, dep1D)

    return predData

end

#-------------------------------------------------------------------------------
function getPredData(emData::TEMData, mcParam::MCParameter)

    # get model parameter
    dep1D, rho1D = getModelParameter(mcParam)

    xLen = emData.loopxLen
    yLen = emData.loopyLen
    timePoint = emData.timePoint
    predData  = compTEM1dResponse(xLen, yLen, timePoint, rho1D, dep1D)

    return predData

end

#-------------------------------------------------------------------------------
function getModelParameter(mcParam::MCParameter)

    # convert location from logarithmic scale to linear scale
    nLayer = mcParam.nLayer
    zNode  = mcParam.zNode[1:nLayer]

    # location
    idx   = sortperm(zNode)
    zNode = 10 .^ zNode[idx]
    dep1D = vcat(0., zNode)

    # convert resistivity from logarithmic scale to linear scale
    rho = mcParam.rho[1:nLayer]
    rho1D = 10 .^ rho[idx]

    return dep1D, rho1D

end

#-------------------------------------------------------------------------------
include("compTEMRespData.jl")
include("compCSEMRespData.jl")
include("compMTRespData.jl")

end # TBFwdSolver
