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
	nextidx() = (idx = i; i += 1; idx)
	@sync begin

		for p ∈ pids
			@async begin
				while true
					idx = nextidx()
					if idx > np
						break
					end
					@time (result[idx], status[idx]) = remotecall_fetch(runMCMC, p,
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
	(mcRef, stRef, dfRef, paramRef) = distTemperingParameter(emData, mclimits, pids)
	# mcArrayDA, mcstatusDA, mcDatafitDA, currParamDA  

	# perform mcmc sampling
	iterNo       = 0
	totalsamples = mclimits.totalsamples
	tempLadder   = copy(tempData.tempLadder)
	while (iterNo < totalsamples)

		iterNo = iterNo + 1
		# 返回的似然概率
		lkhdRef = parallelTemperedMCMC(emData, mclimits, tempLadder, mcRef,
			paramRef, stRef, dfRef, pids)

		# do parallel tempering
		for k ∈ 1:tempData.nT
			# select a pair of chains
			(tchain01, tchain02) = selectSwapChains(tempLadder)
			# 返回的是两个随机数，范围是1到nT
			tvalue01 = tempLadder[tchain01] # 选的两个温度
			tvalue02 = tempLadder[tchain02]

			# fetch likelihood associated with selected chains
			lkhood01 = lkhdRef[tchain01] # 似然概率
			lkhood02 = lkhdRef[tchain02]

			# calculate swap rate  计算交换律
			doswap = compSwapRate(tchain01, tchain02, lkhood01, lkhood02, tempLadder)
			tempData.swapchain[1, k, iterNo] = tchain01
			tempData.swapchain[2, k, iterNo] = tchain02
			if doswap
				tempLadder[tchain01] = tvalue02
				tempLadder[tchain02] = tvalue01
				tempData.swapchain[3, k, iterNo] = 1
			end
			#
		end
		# update temperature ladder 把新的梯度温度发送到各个工作进程上
		sendData(pids, tempLadder = tempLadder)

	end

	#
	return mcRef, stRef, dfRef, paramRef

end


"""  
Implementation Method (实现方法):  

This function implements parallel tempering MCMC using MPI for distributed computing.   
The implementation follows these key steps:  
这个函数使用MPI实现并行回火MCMC方法，主要实现步骤如下：  

1. Initialization (初始化):  
   - Each process initializes its own MCMC chain with different temperatures  
   - 每个进程使用不同的温度初始化自己的MCMC链  
   
2. Main Loop (主循环):  
   - Run MCMC sampling for each chain independently  
   - Gather likelihood values from all processes  
   - Attempt temperature exchanges between chains  
   - 独立运行每个链的MCMC采样  
   - 收集所有进程的似然值  
   - 尝试链之间的温度交换  

3. Temperature Exchange (温度交换):  
   - Root process (rank 0) selects chain pairs for exchange  
   - Calculate exchange probability based on Metropolis criterion  
   - Synchronize temperature updates across all processes  
   - 主进程(rank 0)选择要交换的链对  
   - 基于Metropolis准则计算交换概率  
   - 在所有进程间同步温度更新  

4. Synchronization (同步):  
   - Use MPI barriers to ensure proper synchronization  
   - Maintain consistent temperature ladder across all processes  
   - 使用MPI障碍确保proper同步  
   - 在所有进程间保持一致的温度阶梯  

Key Features (主要特点):  
- Distributed parallel computing using MPI  
- Automatic temperature exchange between chains  
- Efficient load balancing across processes  
- 使用MPI进行分布式并行计算  
- 链之间自动温度交换  
- 进程间高效负载均衡  
"""  
function runMPITemperedMCMC(  
    emData::EMData,        # Electromagnetic data structure (电磁数据结构)  
    mclimits::MCPrior,     # MCMC prior limits and parameters (MCMC先验限制和参数)  
    tempData::TBTemperature, # Temperature ladder information (温度阶梯信息)  
    nT::Int,               # Number of temperatures/chains (温度/链的数量)  
    rank::Int,             # Current process rank (当前进程序号)  
    comm::MPI.Comm         # MPI communicator (MPI通信器)  
)  
    # Initialize MCMC parameters for each process  
    # 为每个进程初始化MCMC参数  
    (mcArrayWorker, mcstatusWorker, mcDatafitWorker, mcParamCurrentWorker) =  
        initMPITemperingParameter(emData, mclimits, nT, rank + 1)  

    # Initialize iteration counter and get total samples needed  
    # 初始化迭代计数器并获取所需的总样本数  
    iterNo = 0  
    totalsamples = mclimits.totalsamples  
    # Create local copy of temperature ladder  
    # 创建温度阶梯的本地副本  
    tempLadder = copy(tempData.tempLadder)  

    # Main MCMC loop  
    # 主MCMC循环  
    while iterNo < totalsamples  
        iterNo = iterNo + 1  
        
        # Allocate space for likelihood values from all processes  
        # 为所有进程的似然值分配空间  
        lkhdRef = Array{Float64}(undef, nT)  
        
        # Compute likelihood for current process  
        # 计算当前进程的似然值  
        myLkhd = parallelMPITemperedMCMC(emData, mclimits, tempLadder, mcArrayWorker,  
            mcParamCurrentWorker, mcstatusWorker, mcDatafitWorker, rank)  

        # Gather all likelihoods from all processes  
        # 收集所有进程的似然值  
        lkhdRef = MPI.Allgather(myLkhd, comm)  

        # Temperature exchange phase  
        # 温度交换阶段，单次迭代中的温度交换  
        for k ∈ 1:tempData.nT  
            # Root process selects chains for exchange  
            # 0号进程选择要交换的链  
            if rank == 0  
                (tchain01, tchain02) = selectSwapChains(tempLadder)  
                chains = [tchain01, tchain02]  
            else  
                chains = [0, 0]  
            end  

            # Broadcast selected chains to all processes  
            # 向所有进程广播选择的链  
            chains = MPI.Bcast!(chains, 0, comm)  
            tchain01, tchain02 = chains  

            # Convert to 0-based indices for MPI  
            # 转换为基于0的索引用于MPI  
            chain1 = tchain01 - 1  
            chain2 = tchain02 - 1  

            # Get temperatures and likelihoods for selected chains  
            # 获取选定链的温度和似然值，所有进程都提取选取的链的温度梯度和似然值
            tvalue1 = tempLadder[tchain01]  
            tvalue2 = tempLadder[tchain02]  
            lkhood1 = lkhdRef[tchain01]  
            lkhood2 = lkhdRef[tchain02]  

            # Calculate exchange probability  
            # 计算交换概率  
            doswap = false  
            if rank == min(chain1, chain2)  
                doswap = compSwapRate(tchain01, tchain02, lkhood1, lkhood2, tempLadder)  
            end  

            # Broadcast swap decision to all processes  
            # 向所有进程广播交换决定  
            doswap = MPI.Bcast(doswap, min(chain1, chain2), comm)  

            # Perform temperature swap if accepted  
            # 如果接受则执行温度交换  
            if doswap  
                tempLadder[tchain01] = tvalue2  # 所有进程都执行温度交换
                tempLadder[tchain02] = tvalue1  
                # Record swap in history (only in specific process)  
                # 0号进程记录交换历史  
                if rank == 0  
                    tempData.swapchain[1,k,iterNo] = tchain01  
                    tempData.swapchain[2,k,iterNo] = tchain02  
                    tempData.swapchain[3,k,iterNo] = 1  
                end  
            end  

            # Synchronize all processes  
            # 同步所有进程  
            MPI.Barrier(comm)  
        end  

        # Synchronize temperature ladder across all processes  
        # 在所有进程间同步温度阶梯，一次循环结束后同步温度阶梯
        tempLadder = MPI.Allgather(tempLadder[rank+1], comm)  
    end  

    # Return final MCMC results  
    # 返回最终的MCMC结果  
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
		for (idx, ip) in enumerate(pids)
			@async lkhd[idx] = remotecall_fetch(parallelTemperedMCMC, ip, emData,
				mclimits, tempLadder, mcArrayRef, currParamRef, mcstatusRef, mcDatafitRef)
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
	mcArray = localpart(mcArrayRef)[1]
	mcstatus = localpart(mcstatusRef)[1]
	mcDatafit = localpart(mcDatafitRef)[1]
	mcParamCurrent = localpart(currParamRef)[1]

	# reset proposed and delayed models
	mcParamProposed          = duplicateMCParameter(mcParamCurrent)
	mcDatafit.propPredData   = copy(mcDatafit.currPredData)
	mcDatafit.propLikelihood = mcDatafit.currLikelihood
	mcDatafit.propMisfit     = mcDatafit.currMisfit

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
		(mcDatafit.propLikelihood, mcDatafit.propMisfit) = compDataMisfit(emData,
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
	cycleNumber = 1000
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
	`parallelMPITemperedMCMC(emData, mclimits, tempLadder, mcArrayWorker, mcParamCurrentWorker,
	mcstatusWorker, mcDatafitWorker, rank)`
 
Single MCMC step with parallel tempering  
Returns likelihood for current chain  
"""
function parallelMPITemperedMCMC(emData::EMData,
	mclimits::MCPrior,
	tempLadder::Vector{Float64},
	mcArrayWorker::Vector{MChainArray},
	mcParamCurrentWorker::Vector{MCParameter},
	mcstatusWorker::Vector{MCStatus},
	mcDatafitWorker::Vector{MCDataFit},
	rank::Int)
	# Get current temperature for this process  
	temperature = tempLadder[rank+1]

	# Get worker-specific data  
	mcArray = mcArrayWorker[rank+1]
	mcstatus = mcstatusWorker[rank+1]
	mcDatafit = mcDatafitWorker[rank+1]
	mcParamCurrent = mcParamCurrentWorker[rank+1]

	# Initialize proposed model  
	mcParamProposed = duplicateMCParameter(mcParamCurrent)
	mcDatafit.propPredData = copy(mcDatafit.currPredData)
	mcDatafit.propLikelihood = mcDatafit.currLikelihood
	mcDatafit.propMisfit = mcDatafit.currMisfit

	# Select and perform MCMC step  
	rjstep = selectChainStep!(MChainStep(false, false, false, false))

	# Perform selected step type  
	if rjstep.isBirth
		mcBirth!(mcParamCurrent, mcParamProposed, mclimits)
	elseif rjstep.isDeath
		mcDeath!(mcParamCurrent, mcParamProposed, mclimits)
	elseif rjstep.isMove
		mcMoveLocation!(mcParamCurrent, mcParamProposed, mclimits)
	elseif rjstep.isPerturb
		mcPerturbProperty!(mcParamCurrent, mcParamProposed, mclimits)
	end

	# Process proposed model if within bounds  
	if mcParamProposed.inBound
		# Compute predicted data and likelihood  
		mcDatafit.propPredData = getPredData(emData, mcParamProposed)
		(mcDatafit.propLikelihood, mcDatafit.propMisfit) =
			compDataMisfit(emData, mcDatafit.propPredData)

		# Calculate acceptance ratio with temperature scaling  
		propAlpha = compAcceptanceRatio(mcParamCurrent, mcParamProposed,
			mcDatafit, mclimits, rjstep, temperature)

		# Check Metropolis-Hastings criterion  
		checkMHCriterion!(propAlpha, mcstatus, rjstep)
	end

	# Update chain if proposal accepted  
	if mcParamProposed.inBound && mcstatus.accepted
		updateChainModel!(mcParamCurrent, mcParamProposed, mcDatafit)
		mcDatafit.dataResiduals = emData.obsData - mcDatafit.propPredData
	end

	# Record chain state  
	updateChainArray!(mcArray, mcParamCurrent, rjstep, mcstatus, mcDatafit)

	# Progress output every 1000 iterations  
	cycleNumber = 1000
	iterNo = mcArray.nsample
	if mod(iterNo, cycleNumber) == 0
		print("Worker #$(rank): iterNo=$(iterNo), dataMisfit=$(mcDatafit.currMisfit)\n")
	end

	# Update worker data  
	mcArrayWorker[rank+1] = mcArray
	mcstatusWorker[rank+1] = mcstatus
	mcDatafitWorker[rank+1] = mcDatafit
	mcParamCurrentWorker[rank+1] = mcParamCurrent

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
	lkhood02::Tv, tempLadder::Vector{Tv}) where {Ti <: Int, Tv <: Float64}

	#
	doswap = false
	tvalue01 = tempLadder[tchain01]
	tvalue02 = tempLadder[tchain02]

	# calculate swap rate
	srate01 = -lkhood01 / tvalue02 + lkhood02 / tvalue02
	srate02 = -lkhood02 / tvalue01 + lkhood01 / tvalue01
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
	bdsteps = zeros(0), rhosteps = zeros(0), zsteps = zeros(0))

	#
	np = length(pids)
	#
	mcArrayRef   = Array{MChainArray}(undef, np)
	mcstatusRef  = Array{MCStatus}(undef, np)
	mcDatafitRef = Array{MCDataFit}(undef, np)
	currParamRef = Array{MCParameter}(undef, np)

	for i ∈ 1:np

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
		mcstatus = MCStatus(false, zeros(Int, 4), zeros(Int, 4))
		mcDatafit = initMCDataFit()
		mcArray = initMChainArray(mclimits)
		mcParamCurrent = initModelParameter(mclimits)

		# compute predicted data, likelihood and data misfit
		currPredData = getPredData(emData, mcParamCurrent)
		(currLikelihood, currMisfit) = compDataMisfit(emData, currPredData)
		mcDatafit.currPredData = currPredData
		mcDatafit.currLikelihood = currLikelihood
		mcDatafit.currMisfit = currMisfit
		println("starting data misfit = $(mcDatafit.currMisfit)")

		# record starting model
		mcArray.startModel = duplicateMCParameter(mcParamCurrent)
		mcArrayRef[i]      = mcArray
		mcstatusRef[i]     = mcstatus
		mcDatafitRef[i]    = mcDatafit
		currParamRef[i]    = mcParamCurrent

	end

	# distributed array
	mcArrayDA   = distribute(mcArrayRef, procs = pids, dist = [np])
	mcstatusDA  = distribute(mcstatusRef, procs = pids, dist = [np])
	mcDatafitDA = distribute(mcDatafitRef, procs = pids, dist = [np])
	currParamDA = distribute(currParamRef, procs = pids, dist = [np])

	return mcArrayDA, mcstatusDA, mcDatafitDA, currParamDA

end #

"""
	`initMPITemperingParameter(emData, mclimits, nworkers)`
"""
function initMPITemperingParameter(emData::EMData, mclimits::MCPrior, nworkers::Int, rank::Int, bdsteps = zeros(0), rhosteps = zeros(0), zsteps = zeros(0))

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
	mcstatus = MCStatus(false, zeros(Int, 4), zeros(Int, 4))
	mcDatafit = initMCDataFit()
	mcArray = initMChainArray(mclimits)
	mcParamCurrent = initModelParameter(mclimits)

	# compute predicted data, likelihood and data misfit
	currPredData = getPredData(emData, mcParamCurrent)
	(currLikelihood, currMisfit) = compDataMisfit(emData, currPredData)
	mcDatafit.currPredData = currPredData
	mcDatafit.currLikelihood = currLikelihood
	mcDatafit.currMisfit = currMisfit
	worker = rank - 1
	print("Worker #$(worker): starting data misfit = $(mcDatafit.currMisfit)\n")

	# record starting model
	mcArray.startModel = duplicateMCParameter(mcParamCurrent)
	mcArrayWorker[rank] = mcArray
	mcstatusWorker[rank] = mcstatus
	mcDatafitWorker[rank] = mcDatafit
	mcParamCurrentWorker[rank] = mcParamCurrent

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
