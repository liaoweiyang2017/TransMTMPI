#-------------------------------------------------------------------------------
#
# define routines to perform parallel tempering of MCMC.
#
#-------------------------------------------------------------------------------
export parallelMCMCsampling
export parallelTemperedMCMC
export selectSwapChains
export compSwapRate
export distTemperingParameter, sendData
export runTemperedMCMC
export runMPITemperedMCMC
export parallelMPITemperedMCMC
export initMPITemperingParameter
#-------------------------------------------------------------------------------

#------------------------------------------------------------------------------
"""
    `parallelMCMCsampling(mclimits, emData, pids)`

run multiple Markov chains in parallel without parallel tempering.

"""
function parallelMCMCsampling(mclimits::MCPrior, emData::EMData, pids::Vector)

    #
    np = length(pids)
    result = Array{Any}(undef, np)
    status = Array{Any}(undef, np)

    # run multiple chains in parallel
    i = 1
    nextidx() = (idx = i; i+=1; idx)
    @sync begin

        for p = pids
            @async begin
                while true
                    idx = nextidx()
                    if idx > np
                        break
                    end
                @time (result[idx],status[idx]) = remotecall_fetch(runMCMC, p,
                                            mclimits, emData)
                end
            end # @async
        end # p

    end # @sync

    return result, status

end


#-------------------------------------------------------------------------------
"""
    `runTemperedMCMC(emData, mclimits, tempData, pids)`

"""
function runTemperedMCMC(emData::EMData, mclimits::MCPrior,
                         tempData::TBTemperature, pids::Vector{Int})

    # check
    if tempData.nT != length(pids)
        error("The number of workers should be the same with the number of chains")
    end

    # initialize and distribute rjmcmc paramters for parallel tempering
    # 初始化并行回火贝叶斯的参数
    (mcRef,stRef,dfRef,paramRef) = distTemperingParameter(emData, mclimits, pids)
    # mcArrayDA, mcstatusDA, mcDatafitDA, currParamDA  

    # perform mcmc sampling
    iterNo = 0
    totalsamples = mclimits.totalsamples
    tempLadder   = copy(tempData.tempLadder)
    while (iterNo < totalsamples)

        iterNo = iterNo + 1
        # 返回的似然概率
        lkhdRef = parallelTemperedMCMC(emData, mclimits, tempLadder, mcRef,
                                        paramRef, stRef, dfRef, pids)

        # do parallel tempering
        for k = 1:tempData.nT
            # select a pair of chains
            (tchain01,tchain02) = selectSwapChains(tempLadder)
            # 返回的是两个随机数，范围是1到nT
            tvalue01 = tempLadder[tchain01] # 选的两个温度
            tvalue02 = tempLadder[tchain02]

            # fetch likelihood associated with selected chains
            lkhood01 = lkhdRef[tchain01] # 似然概率
            lkhood02 = lkhdRef[tchain02]

            # calculate swap rate  计算交换律
            doswap = compSwapRate(tchain01,tchain02,lkhood01,lkhood02,tempLadder)
            tempData.swapchain[1,k,iterNo] = tchain01
            tempData.swapchain[2,k,iterNo] = tchain02
            if doswap
                tempLadder[tchain01] = tvalue02
                tempLadder[tchain02] = tvalue01
                tempData.swapchain[3,k,iterNo] = 1
            end
            #
        end
        # update temperature ladder 把新的梯度温度发送到各个工作进程上
        sendData(pids, tempLadder=tempLadder)

    end

    #
    return mcRef,stRef,dfRef,paramRef

end

"""
    `runMPITemperedMCMC(emData, mclimits, tempData, pids)`

"""
function runMPITemperedMCMC(emData::EMData, mclimits::MCPrior,
                            tempData::TBTemperature, nWorkers::Int, rank::Int)

    # initialize rjmcmc paramters for mpi parallel tempering
    # 初始化并行回火贝叶斯的参数
    (mcArrayWorker, mcstatusWorker, mcDatafitWorker, mcParamCurrentWorker) = initMPITemperingParameter(emData, mclimits, nWorkers, rank)

    iterNo = 0
    totalsamples = mclimits.totalsamples
    tempLadder   = copy(tempData.tempLadder)
    
    while (iterNo < totalsamples)
        iterNo = iterNo + 1
        # 返回的似然概率
        lkhdRef = Array{Float64}(undef, nWorkers)
        lkhdRef[rank] = parallelMPITemperedMCMC(emData, mclimits, tempLadder, mcArrayWorker,
                                        mcParamCurrentWorker, mcstatusWorker, mcDatafitWorker, rank)
        # do parallel tempering
        for k = 1:tempData.nT
            # select a pair of chains
            (tchain01,tchain02) = selectSwapChains(tempLadder)
            # 返回的是两个随机数，范围是1到nT
            tvalue01 = tempLadder[tchain01] # 选的两个温度
            tvalue02 = tempLadder[tchain02]

            # fetch likelihood associated with selected chains
            lkhood01 = lkhdRef[tchain01] # 似然概率
            lkhood02 = lkhdRef[tchain02]

            # calculate swap rate  计算交换律
            doswap = compSwapRate(tchain01,tchain02,lkhood01,lkhood02,tempLadder)
            tempData.swapchain[1,k,iterNo] = tchain01
            tempData.swapchain[2,k,iterNo] = tchain02
            if doswap
                tempLadder[tchain01] = tvalue02
                tempLadder[tchain02] = tvalue01
                tempData.swapchain[3,k,iterNo] = 1
            end
            #
        end
        # update temperature ladder 更新梯度温度
        tempData.tempLadder = tempLadder

    end

    #
    return mcArrayWorker, mcstatusWorker, mcDatafitWorker, mcParamCurrentWorker

end


#-------------------------------------------------------------------------------
"""
    `parallelTemperedMCMC(emData, mclimits, tempLadder, mcArrayRef, currParamRef,
    mcstatusRef, mcDatafitRef, pids)`

perform parallel tempered MCMC sampling.

"""
function parallelTemperedMCMC(emData::EMData,
                              mclimits::MCPrior,
                              tempLadder::Vector{Float64},
                              mcArrayRef::DArray,
                              currParamRef::DArray,
                              mcstatusRef::DArray,
                              mcDatafitRef::DArray,
                              pids::Vector{Int})

    #
    np   = length(pids)
    lkhd = zeros(np)

    # run multiple chains in parallel
    @sync begin
        for (idx,ip) in enumerate(pids)
            @async lkhd[idx] = remotecall_fetch(parallelTemperedMCMC,ip,emData,
            mclimits,tempLadder,mcArrayRef,currParamRef,mcstatusRef,mcDatafitRef)
        end

    end # @sync

    return lkhd

end


#-------------------------------------------------------------------------------
"""
    `parallelTemperedMCMC(emData, mclimits, tempLadder, mcArrayRef, currParamRef,
    mcstatusRef, mcDatafitRef)`

perform parallel tempered MCMC sampling.

"""
function parallelTemperedMCMC(emData::EMData,
                              mclimits::MCPrior,
                              tempLadder::Vector{Float64},
                              mcArrayRef::DArray,
                              currParamRef::DArray,
                              mcstatusRef::DArray,
                              mcDatafitRef::DArray)

    # fetch data from worker
    pid = myid()
    idx = pid - 1
    temperature = tempLadder[idx]

    # fetch data from worker
    mcArray   = localpart(mcArrayRef)[1]
    mcstatus  = localpart(mcstatusRef)[1]
    mcDatafit = localpart(mcDatafitRef)[1]
    mcParamCurrent  = localpart(currParamRef)[1]

    # reset proposed and delayed models
    mcParamProposed = duplicateMCParameter(mcParamCurrent)
    mcDatafit.propPredData    = copy(mcDatafit.currPredData)
    mcDatafit.propLikelihood  = mcDatafit.currLikelihood
    mcDatafit.propMisfit      = mcDatafit.currMisfit

    # randomly select a McMC chain step
    rjstep = MChainStep(false, false, false, false)
    rjstep = selectChainStep!(rjstep)

    # which step do we have
    if rjstep.isBirth # birth step
        mcBirth!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isDeath # death step
        mcDeath!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isMove # move step
        mcMoveLocation!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isPerturb # perturb step
        mcPerturbProperty!(mcParamCurrent, mcParamProposed, mclimits)

    end

    # check if proposed model is within the prior bounds
    if mcParamProposed.inBound

        # get the predicted data for the proposed model
        mcDatafit.propPredData = getPredData(emData, mcParamProposed)

        # get model likelihood and data misfit
        (mcDatafit.propLikelihood,mcDatafit.propMisfit) = compDataMisfit(emData,
        mcDatafit.propPredData)

        # calculate acceptance ratio of rjMcMC chain
        propAlpha = compAcceptanceRatio(mcParamCurrent, mcParamProposed,
                               mcDatafit, mclimits, rjstep, temperature)

        # check if Metropolis-Hasting acceptance criterion is meeted
        checkMHCriterion!(propAlpha, mcstatus, rjstep)

    end #

    # update markov chain
    if mcParamProposed.inBound & mcstatus.accepted
        updateChainModel!(mcParamCurrent, mcParamProposed, mcDatafit)
        # data residuals
        mcDatafit.dataResiduals = emData.obsData - mcDatafit.propPredData
    end

    # record in chainarray
    updateChainArray!(mcArray, mcParamCurrent, rjstep, mcstatus, mcDatafit)

    # print iteration info
    cycleNumber  = 1000
    iterNo = mcArray.nsample
    if mod(iterNo, cycleNumber) == 0
         println("iterNo=$(iterNo), dataMisfit=$(mcDatafit.currMisfit)")
    end

    # update associated stuff in the worker
    localpart(mcArrayRef)[1]   = mcArray
    localpart(mcstatusRef)[1]  = mcstatus
    localpart(mcDatafitRef)[1] = mcDatafit
    localpart(currParamRef)[1] = mcParamCurrent

    # return reference to current likelihood
    currLikelihood = mcDatafit.currLikelihood

    return currLikelihood

end


#-------------------------------------------------------------------------------
"""
    `parallelMPITemperedMCMC()`

perform parallel tempered MCMC sampling.
"""
function parallelMPITemperedMCMC(emData::EMData,
                                 mclimits::MCPrior,
                                 tempLadder::Vector{Float64},
                                 mcArrayWorker::Vector{MChainArray},
                                 mcParamCurrentWorker::Vector{MCParameter},
                                 mcstatusWorker::Vector{MCStatus},
                                 mcDatafitWorker::Vector{MCDataFit},
                                 rank::Int)

    # worker temperature                       
    temperature = tempLadder[rank]

    # worker data
    mcArray   = mcArrayWorker[rank]
    mcstatus  = mcstatusWorker[rank]
    mcDatafit = mcDatafitWorker[rank]
    mcParamCurrent  = mcParamCurrentWorker[rank]

    # reset proposed and delayed models
    mcParamProposed = duplicateMCParameter(mcParamCurrent)
    mcDatafit.propPredData    = copy(mcDatafit.currPredData)
    mcDatafit.propLikelihood  = mcDatafit.currLikelihood
    mcDatafit.propMisfit      = mcDatafit.currMisfit

    # randomly select a McMC chain step
    rjstep = MChainStep(false, false, false, false)
    rjstep = selectChainStep!(rjstep)

    # which step do we have
    if rjstep.isBirth # birth step
        mcBirth!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isDeath # death step
        mcDeath!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isMove # move step
        mcMoveLocation!(mcParamCurrent, mcParamProposed, mclimits)

    elseif rjstep.isPerturb # perturb step
        mcPerturbProperty!(mcParamCurrent, mcParamProposed, mclimits)

    end

    # check if proposed model is within the prior bounds
    if mcParamProposed.inBound

        # get the predicted data for the proposed model
        mcDatafit.propPredData = getPredData(emData, mcParamProposed)

        # get model likelihood and data misfit
        (mcDatafit.propLikelihood,mcDatafit.propMisfit) = compDataMisfit(emData,
        mcDatafit.propPredData)

        # calculate acceptance ratio of rjMcMC chain
        propAlpha = compAcceptanceRatio(mcParamCurrent, mcParamProposed,
                               mcDatafit, mclimits, rjstep, temperature)

        # check if Metropolis-Hasting acceptance criterion is meeted
        checkMHCriterion!(propAlpha, mcstatus, rjstep)

    end #

    # update markov chain
    if mcParamProposed.inBound & mcstatus.accepted
        updateChainModel!(mcParamCurrent, mcParamProposed, mcDatafit)
        # data residuals
        mcDatafit.dataResiduals = emData.obsData - mcDatafit.propPredData
    end

    # record in chainarray
    updateChainArray!(mcArray, mcParamCurrent, rjstep, mcstatus, mcDatafit)

    # print iteration info
    cycleNumber  = 1000
    iterNo = mcArray.nsample
    if mod(iterNo, cycleNumber) == 0
         println("Worker #$(rank): iterNo=$(iterNo), dataMisfit=$(mcDatafit.currMisfit)")
    end

    # update associated stuff in the worker
    mcArrayWorker[rank]   = mcArray
    mcstatusWorker[rank]  = mcstatus
    mcDatafitWorker[rank] = mcDatafit
    mcParamCurrentWorker[rank] = mcParamCurrent

    currLikelihood = mcDatafit.currLikelihood

    return currLikelihood

end

#-------------------------------------------------------------------------------
"""
    `selectSwapChains(tempLadder)`

select a pair of chains to swap.

"""
function selectSwapChains(tempLadder::Vector)

    # first select one chain randomly
    nchain = length(tempLadder)
    tchain = randperm(nchain)[1:2]
    tchain01 = tchain[1]
    tchain02 = tchain[2]
    tvalue01 = tempLadder[tchain01]
    tvalue02 = tempLadder[tchain02]
    while tvalue01 == tvalue02
        tchain02 = unirandInteger(1, nchain)
        tvalue02 = tempLadder[tchain02]
    end

    return tchain01, tchain02

end


#-------------------------------------------------------------------------------
"""
    `compSwapRate(tchain01, tchain02, lkhood01, lkhood02, tempLadder)`

calculate the swap rate and determine whether to swap the two chain selected.

"""
function compSwapRate(tchain01::Ti, tchain02::Ti, lkhood01::Tv,
                      lkhood02::Tv, tempLadder::Vector{Tv}) where {Ti<:Int, Tv<:Float64}

    #
    doswap  = false
    tvalue01 = tempLadder[tchain01]
    tvalue02 = tempLadder[tchain02]

    # calculate swap rate
    srate01 = -lkhood01/tvalue02 + lkhood02/tvalue02
    srate02 = -lkhood02/tvalue01 + lkhood01/tvalue01
    srate   = srate01 + srate02
    srate   = min(0.0, srate)
    rvalue  = log(rand())
    if rvalue <= srate
        doswap = true
    end
    return doswap

end


#-------------------------------------------------------------------------------
"""
    `distTemperingParameter(mcParamCurrent, mcstatus, mcArray, mcstatus, mcDatafit
    pids)`

initialize parameters for parallel tempering.

"""
function distTemperingParameter(emData::EMData, mclimits::MCPrior, pids::Vector,
                        bdsteps=zeros(0), rhosteps=zeros(0), zsteps=zeros(0))

    #
    np = length(pids)
    #
    mcArrayRef   = Array{MChainArray}(undef, np)
    mcstatusRef  = Array{MCStatus}(   undef, np)
    mcDatafitRef = Array{MCDataFit}(  undef, np)
    currParamRef = Array{MCParameter}(undef, np)

    for i = 1:np

        if !isempty(bdsteps)
            mclimits.rhostd = bdsteps[i]
        end
        if !isempty(rhosteps)
            mclimits.mrhostd = rhosteps[i]
        end
        if !isempty(zsteps)
            mclimits.zstd = zsteps[i]
        end

        # initialize model parameter
        mcstatus  = MCStatus(false, zeros(Int,4), zeros(Int,4))
        mcDatafit = initMCDataFit()
        mcArray   = initMChainArray(mclimits)
        mcParamCurrent  = initModelParameter(mclimits)

        # compute predicted data, likelihood and data misfit
        currPredData = getPredData(emData, mcParamCurrent)
        (currLikelihood, currMisfit) = compDataMisfit(emData, currPredData)
        mcDatafit.currPredData   = currPredData
        mcDatafit.currLikelihood = currLikelihood
        mcDatafit.currMisfit     = currMisfit
        println("starting data misfit = $(mcDatafit.currMisfit)")

        # record starting model
        mcArray.startModel = duplicateMCParameter(mcParamCurrent)
        mcArrayRef[i]      = mcArray
        mcstatusRef[i]     = mcstatus
        mcDatafitRef[i]    = mcDatafit
        currParamRef[i]    = mcParamCurrent

    end

    # distributed array
    mcArrayDA   = distribute(mcArrayRef,   procs=pids, dist=[np])
    mcstatusDA  = distribute(mcstatusRef,  procs=pids, dist=[np])
    mcDatafitDA = distribute(mcDatafitRef, procs=pids, dist=[np])
    currParamDA = distribute(currParamRef, procs=pids, dist=[np])

    return mcArrayDA, mcstatusDA, mcDatafitDA, currParamDA

end #

"""
    `initMPITemperingParameter(emData, mclimits, nworkers)`
"""
function initMPITemperingParameter(emData::EMData, mclimits::MCPrior, nworkers::Int, rank::Int,
                                   bdsteps=zeros(0), rhosteps=zeros(0), zsteps=zeros(0))

    mcArrayWorker = Array{MChainArray}(undef, nworkers)
    mcParamCurrentWorker = Array{MCParameter}(undef, nworkers)
    mcDatafitWorker = Array{MCDataFit}(undef, nworkers)
    mcstatusWorker = Array{MCStatus}(undef, nworkers)

    if !isempty(bdsteps)
        mclimits.rhostd = bdsteps[rank]
    end
    if !isempty(rhosteps)
        mclimits.mrhostd = rhosteps[rank]
    end
    if !isempty(zsteps)
        mclimits.zstd = zsteps[rank]
    end

    # initialize model parameter
    mcstatus  = MCStatus(false, zeros(Int,4), zeros(Int,4))
    mcDatafit = initMCDataFit()
    mcArray   = initMChainArray(mclimits)
    mcParamCurrent  = initModelParameter(mclimits)

    # compute predicted data, likelihood and data misfit
    currPredData = getPredData(emData, mcParamCurrent)
    (currLikelihood, currMisfit) = compDataMisfit(emData, currPredData)
    mcDatafit.currPredData   = currPredData
    mcDatafit.currLikelihood = currLikelihood
    mcDatafit.currMisfit     = currMisfit
    println("Worker #$(rank): starting data misfit = $(mcDatafit.currMisfit)")

    # record starting model
    mcArray.startModel = duplicateMCParameter(mcParamCurrent)
    mcArrayWorker[rank]      = mcArray
    mcstatusWorker[rank]     = mcstatus
    mcDatafitWorker[rank]    = mcDatafit
    mcParamCurrentWorker[rank]    = mcParamCurrent

    return mcArrayWorker, mcstatusWorker, mcDatafitWorker, mcParamCurrentWorker

end

#------------------------------------------------------------------------------
"""
    `sendData()` sends an arbitrary number of variables to specified processors

"""
function sendData(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, Core.eval(Main, Expr(:(=), nm, val)))
    end
end

#------------------------------------------------------------------------------
"""
    `sendData()` sends an arbitrary number of variables to specified processors

"""
function sendData(p::Vector{Int}; args...)
    for pid in p
        sendData(pid; args...)
    end
end

#-------------------------------------------------------------------------------
