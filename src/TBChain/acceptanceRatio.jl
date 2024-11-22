export compDataMisfit
export compAcceptanceRatio
export compBirthAlpha, compDeathAlpha, compPerturbAlpha
export checkMHCriterion!
#-------------------------------------------------------------------------------
"""
    `compDataMisfit(obsData, predData, dataErr)`

compute likelihood and data misfit.

"""
function compDataMisfit(obsData::Array{T}, predData::Array{T},
                        dataErr::Array{T}) where{T<:Float64}

    #
    dataMisfit = (obsData .- predData) ./ dataErr
    likelihood = 0.5 * dot(dataMisfit, dataMisfit)
    dataMisfit = likelihood * 2

    return likelihood, dataMisfit

end


#-------------------------------------------------------------------------------
"""
    `compDataMisfit(emData, predData)`

compute likelihood and data misfit.

"""
function compDataMisfit(emData::EMData, predData::Array{Float64})

    obsData = emData.obsData
    dataErr = emData.dataErr
    (likelihood,dataMisfit) = compDataMisfit(obsData, predData, dataErr)

    return likelihood, dataMisfit

end


#-------------------------------------------------------------------------------
"""
    `compAcceptanceRatio(mcParam, mcParamProposed, mcDatafit, mclimits, mcstep)`

compute the acceptance ratio of current model proposal.

"""
function compAcceptanceRatio(mcParam::MCParameter,
                             mcParamProposed::MCParameter,
                             mcDatafit::MCDataFit,
                             mclimits::MCPrior,
                             mcstep::MChainStep,
                             temperature::Float64=1.0)

    #
    currLikelihood = mcDatafit.currLikelihood
    propLikelihood = mcDatafit.propLikelihood

    if mcstep.isBirth
        alpha = compBirthAlpha(mcParam, mcParamProposed, currLikelihood,
                               propLikelihood, mclimits, temperature)

    elseif mcstep.isDeath
        alpha = compDeathAlpha(mcParam, mcParamProposed, currLikelihood,
                               propLikelihood, mclimits, temperature)

    elseif mcstep.isMove | mcstep.isPerturb
        alpha = compPerturbAlpha(currLikelihood, propLikelihood, temperature)

    end

    return alpha

end


#-------------------------------------------------------------------------------
"""
    `compBirthAlpha(propAlpha, currLikelihood, propLikelihood)`

compute the acceptance probability for a birth step.

"""
function compBirthAlpha(mcParam::MCParameter,
                        mcParamProposed::MCParameter,
                        currLikelihood::T,
                        propLikelihood::T,
                        mclimits::MCPrior,
                        temperature::T=1.0) where{T<:Float64}

    #
    vmin = mclimits.rhomin
    vmax = mclimits.rhomax
    cidx = mcParam.layeridx
    pidx = mcParamProposed.nLayer

    # p(m')/p(m) and q(m|m')/q(m'|m)
    alpha01 = mclimits.rhostd * sqrt(2*pi) / (vmax - vmin)
    alpha01 = log(alpha01)

    alpha02 = (mcParamProposed.rho[pidx] - mcParam.rho[cidx])^2 / (2*mclimits.rhostd^2)

    # p(d|m')/p(d|m)
    alpha03 = (-propLikelihood + currLikelihood) / temperature

    # acceptance ratio for birth move
    # alpha = propAlpha - log(vmax-vmin) - propLikelihood + currLikelihood
    alpha   = alpha01 + alpha02 + alpha03
    alpha = min(0.0, alpha)

    return alpha

end


#-------------------------------------------------------------------------------
"""
    `compDeathAlpha(propAlpha, currLikelihood, propLikelihood)`

compute the acceptance probability for a death step.

"""
function compDeathAlpha(mcParam::MCParameter,
                        mcParamProposed::MCParameter,
                        currLikelihood::T,
                        propLikelihood::T,
                        mclimits::MCPrior,
                        temperature::T=1.0) where{T<:Float64}

    #
    vmin = mclimits.rhomin
    vmax = mclimits.rhomax
    cidx = mcParam.layeridx
    pidx = mcParamProposed.nLayer

    # find the nearest cell in new mesh to the one that was deleted
    idx = findLocation1D(mcParam.zNode[cidx], mcParamProposed.zNode[1:pidx])

    # p(m')/p(m) and q(m|m')/q(m'|m)
    alpha01 = (vmax - vmin) / (mclimits.rhostd * sqrt(2*pi) )
    alpha01 = log(alpha01)

    alpha02 = -(mcParamProposed.rho[idx] - mcParam.rho[cidx])^2 / (2*mclimits.rhostd^2)

    # p(d|m')/p(d|m)
    alpha03 = (-propLikelihood + currLikelihood) / temperature

    # acceptance ratio for death move
    # alpha = propAlpha + log(vmax-vmin) - propLikelihood + currLikelihood
    alpha = alpha01 + alpha02 + alpha03
    alpha = min(0.0, alpha)

    return alpha

end


#-------------------------------------------------------------------------------
"""
    `compPerturbAlpha(currLikelihood, propLikelihood)`

compute the acceptance probability for a move step.

"""
function compPerturbAlpha(currLikelihood::T, propLikelihood::T,
                          temperature::T=1.0) where{T<:Float64}

    #
    alpha = (-propLikelihood + currLikelihood) / temperature
    alpha = min(0.0, alpha)

    return alpha

end


#-------------------------------------------------------------------------------
"""
    `cheackMHCriterion(acceptanceRatio, mcstatus, mcstep)`

"""
function checkMHCriterion!(acceptanceRatio::Float64,
                           mcstatus::MCStatus,
                           mcstep::MChainStep)

    #
    mhNumber = rand()
    mhNumber = log(mhNumber)
    if mhNumber <= acceptanceRatio
        mcstatus.accepted = true
        if mcstep.isBirth
            mcstatus.acceptstats[1] += 1
        elseif mcstep.isDeath
            mcstatus.acceptstats[2] += 1
        elseif mcstep.isMove
            mcstatus.acceptstats[3]  += 1
        elseif mcstep.isPerturb
            mcstatus.acceptstats[4] += 1
        end

    else
        mcstatus.accepted = false
        if mcstep.isBirth
            mcstatus.rejectstats[1] += 1
        elseif mcstep.isDeath
            mcstatus.rejectstats[2] += 1
        elseif mcstep.isMove
            mcstatus.rejectstats[3] += 1
        elseif mcstep.isPerturb
            mcstatus.rejectstats[4] += 1
        end

    end

    return mcstatus

end
