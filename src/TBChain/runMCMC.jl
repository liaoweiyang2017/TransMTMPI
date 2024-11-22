export runMCMC #输出
export MCMCsampling!
export resetChainModel!
export updateChainModel!
export updateChainArray!
export continueMCMCsampling
export readsamplefile

#------------------------------------------------------------------------------
function runMCMC(mclimits::MCPrior, emData::EMData)

    # initialize model parameter 初始化模型参数
    mcParamCurrent = initModelParameter(mclimits) # 得到当前的随机层数，随机深度和随机电阻率大小
    mcDatafit = initMCDataFit() # 该结构体所有数据初始化为0
    # mcDatafit结构体包括当前预测数据，当前似然概率，当前拟合差，和下一步提出的预测数据，
    # 提出的似然概率，提出的拟合差，还有数据残差
    mcstep    = MChainStep(false, false, false, false) # 初始的MC链的转态全设置为否
    # 包括rjmcmc方法的四个步骤，生灭移动和扰动，初始化状态全为否
    mcstatus  = MCStatus(false,zeros(Int,4),zeros(Int,4)) 
    # 接收状态初始为否（布尔型），接收统计和拒绝统计数组都初始化为0

    # compute predicted data 计算预测数据
    currPredData = getPredData(emData, mcParamCurrent) # 计算初始的随机数据每个频率的响应值
    residuals = emData.obsData - currPredData  #计算残差：观测数据与预测数据之差

    # calculate likelihood and data misfit
    (currLikelihood,currMisfit) = compDataMisfit(emData, currPredData)# 计算似然函数和数据拟合差

    # 更新数据参数
    mcDatafit.currPredData   = currPredData #当前预测数据
    mcDatafit.currLikelihood = currLikelihood #当前似然函数
    mcDatafit.currMisfit = currMisfit #当前拟合差

    # initialize proposed model 初始化下一步提出的模型，直接复制当前的MC参数
    mcParamProposed = duplicateMCParameter(mcParamCurrent) #初始化提出mc参数

    # start MCMC sampling 开始蒙特卡洛马可洛夫采样
    mcArray = MCMCsampling!(mcParamCurrent, mcParamProposed, mcstep,
                            mclimits, mcstatus, mcDatafit, emData)

    return mcArray, mcstatus

end


#------------------------------------------------------------------------------
function runMCMC(mclimits::MCPrior, emData::EMData, rank::Int)

    # initialize model parameter 初始化模型参数
    mcParamCurrent = initModelParameter(mclimits) # 得到当前的随机层数，随机深度和随机电阻率大小
    mcDatafit = initMCDataFit() # 该结构体所有数据初始化为0
    # mcDatafit结构体包括当前预测数据，当前似然概率，当前拟合差，和下一步提出的预测数据，
    # 提出的似然概率，提出的拟合差，还有数据残差
    mcstep    = MChainStep(false, false, false, false) # 初始的MC链的转态全设置为否
    # 包括rjmcmc方法的四个步骤，生灭移动和扰动，初始化状态全为否
    mcstatus  = MCStatus(false,zeros(Int,4),zeros(Int,4)) 
    # 接收状态初始为否（布尔型），接收统计和拒绝统计数组都初始化为0

    # compute predicted data 计算预测数据
    currPredData = getPredData(emData, mcParamCurrent) # 计算初始的随机数据每个频率的响应值
    residuals = emData.obsData - currPredData  #计算残差：观测数据与预测数据之差

    # calculate likelihood and data misfit
    (currLikelihood,currMisfit) = compDataMisfit(emData, currPredData)# 计算似然函数和数据拟合差

    # 更新数据参数
    mcDatafit.currPredData   = currPredData #当前预测数据
    mcDatafit.currLikelihood = currLikelihood #当前似然函数
    mcDatafit.currMisfit = currMisfit #当前拟合差

    # initialize proposed model 初始化下一步提出的模型，直接复制当前的MC参数
    mcParamProposed = duplicateMCParameter(mcParamCurrent) #初始化提出mc参数

    # start MCMC sampling 开始蒙特卡洛马可洛夫采样
    mcArray = MCMCsampling!(mcParamCurrent, mcParamProposed, mcstep,
                            mclimits, mcstatus, mcDatafit, emData, rank)

    return mcArray, mcstatus

end


#------------------------------------------------------------------------------
"""
    `MCMCsampling(mcParamCurrent,mcParamProposed,mcstep,mclimits,mcstatus,
    mcDatafit,emData)`

produces an ensemble of samples with rjMCMC algorithm.
    使用rjmcmc算法, 产生一组样本
"""
function MCMCsampling!(mcParamCurrent::T, #MC当前参数
                      mcParamProposed::T, #MC提出参数
                      mcstep::MChainStep, #MC 步骤
                      mclimits::MCPrior,  #MC限制参数：或者说先验
                      mcstatus::MCStatus, #MC状态
                      mcDatafit::MCDataFit, #MC数据拟合
                      emData::EMData) where {T<:MCParameter}

    # mcmc information
    mcArray = initMChainArray(mclimits)#初始化mcArray时，只用了mclimits的总采样次数；

    # record starting model
    mcArray.startModel = duplicateMCParameter(mcParamCurrent)

    iterNo = 0 #计数器
    k=0;
    cycleNumber  = 1000 # 循环数
    totalsamples = mclimits.totalsamples
    while (iterNo < totalsamples)

        iterNo = iterNo + 1
        # reset proposed models 重新设置提出的模型
        resetChainModel!(mcParamCurrent, mcParamProposed, mcDatafit) # 把当前的结构体数据都赋给提出的

        # randomly select a type of model perturbation 随机选择一种模型扰动
        mcstep = selectChainStep!(mcstep, iterNo) # 按照一定的概率在四种模式里面变化

        # which step do we have  
        if mcstep.isBirth # birth step 3/16
            mcBirth!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isDeath # death step 3/16
            mcDeath!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isMove # move step 5/16
            mcMoveLocation!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isPerturb # perturb step 5/16
            mcPerturbProperty!(mcParamCurrent, mcParamProposed, mclimits)

        end

        # check if proposed model is within the prior bounds 核对提出的模型是否在先验边界里
        if mcParamProposed.inBound

            if mod(iterNo, cycleNumber) == 0  # 每1000次打印一次信息
                println("iterNo=$(iterNo), dataMisfit=$(mcDatafit.currMisfit)")
            end

            # get the predicted data for the proposed model 
            # 从四种扰动的提出的模型中，计算出正演响应数据
            mcDatafit.propPredData = getPredData(emData, mcParamProposed)

            # get model likelihood and data misfit 得到模型似然函数和数据拟合差
            # 计算得到的似然函数对数概率是拟合差的1/2
            (mcDatafit.propLikelihood,mcDatafit.propMisfit) = compDataMisfit(emData,
            mcDatafit.propPredData)

            # calculate acceptance ratio of rjMcMC chain 计算接收概率
            propAlpha = compAcceptanceRatio(mcParamCurrent, mcParamProposed,
            mcDatafit, mclimits, mcstep)

            # check if Metropolis-Hasting acceptance criterion is meeted
            checkMHCriterion!(propAlpha, mcstatus, mcstep) #核对MH算法接收概率是否被满足
            # mcstatus 记录的事接收还是拒绝，然后记录了四种步骤接收拒绝的数量

        end #

        # update markov chain 更新马尔科夫链
        if mcParamProposed.inBound & mcstatus.accepted  # 如果接受了且在边界范围内
            updateChainModel!(mcParamCurrent, mcParamProposed, mcDatafit)
            k=k+1;
            # data residuals
            mcDatafit.dataResiduals = emData.obsData - mcDatafit.propPredData #计算残差
        end

        # record in chainarray
        updateChainArray!(mcArray, iterNo, mcParamCurrent, mcstep, mcstatus, mcDatafit) #记录链数组

    end
    println("Accept Ratio:",k/totalsamples)
    return mcArray

end


#------------------------------------------------------------------------------
"""
    `MCMCsampling(mcParamCurrent,mcParamProposed,mcstep,mclimits,mcstatus,
    mcDatafit,emData,rank)`

produces an ensemble of samples with rjMCMC algorithm using MPI.
    使用rjmcmc算法, 产生一组样本
"""
function MCMCsampling!(mcParamCurrent::T, #MC当前参数
                      mcParamProposed::T, #MC提出参数
                      mcstep::MChainStep, #MC 步骤
                      mclimits::MCPrior,  #MC限制参数：或者说先验
                      mcstatus::MCStatus, #MC状态
                      mcDatafit::MCDataFit, #MC数据拟合
                      emData::EMData,
                      rank::Int) where {T<:MCParameter}

    # mcmc information
    mcArray = initMChainArray(mclimits)#初始化mcArray时，只用了mclimits的总采样次数；

    # record starting model
    mcArray.startModel = duplicateMCParameter(mcParamCurrent)

    iterNo = 0 #计数器
    k=0;
    cycleNumber  = 1000 # 循环数
    totalsamples = mclimits.totalsamples
    while (iterNo < totalsamples)

        iterNo = iterNo + 1
        # reset proposed models 重新设置提出的模型
        resetChainModel!(mcParamCurrent, mcParamProposed, mcDatafit) # 把当前的结构体数据都赋给提出的

        # randomly select a type of model perturbation 随机选择一种模型扰动
        mcstep = selectChainStep!(mcstep, iterNo) # 按照一定的概率在四种模式里面变化

        # which step do we have  
        if mcstep.isBirth # birth step 3/16
            mcBirth!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isDeath # death step 3/16
            mcDeath!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isMove # move step 5/16
            mcMoveLocation!(mcParamCurrent, mcParamProposed, mclimits)

        elseif mcstep.isPerturb # perturb step 5/16
            mcPerturbProperty!(mcParamCurrent, mcParamProposed, mclimits)

        end

        # check if proposed model is within the prior bounds 核对提出的模型是否在先验边界里
        if mcParamProposed.inBound

            if mod(iterNo, cycleNumber) == 0  # 每1000次打印一次信息
                println("Worker #$(rank): iterNo=$(iterNo), dataMisfit=$(mcDatafit.currMisfit)")
            end

            # get the predicted data for the proposed model 
            # 从四种扰动的提出的模型中，计算出正演响应数据
            mcDatafit.propPredData = getPredData(emData, mcParamProposed)

            # get model likelihood and data misfit 得到模型似然函数和数据拟合差
            # 计算得到的似然函数对数概率是拟合差的1/2
            (mcDatafit.propLikelihood,mcDatafit.propMisfit) = compDataMisfit(emData,
            mcDatafit.propPredData)

            # calculate acceptance ratio of rjMcMC chain 计算接收概率
            propAlpha = compAcceptanceRatio(mcParamCurrent, mcParamProposed,
            mcDatafit, mclimits, mcstep)

            # check if Metropolis-Hasting acceptance criterion is meeted
            checkMHCriterion!(propAlpha, mcstatus, mcstep) #核对MH算法接收概率是否被满足
            # mcstatus 记录的事接收还是拒绝，然后记录了四种步骤接收拒绝的数量

        end #

        # update markov chain 更新马尔科夫链
        if mcParamProposed.inBound & mcstatus.accepted  # 如果接受了且在边界范围内
            updateChainModel!(mcParamCurrent, mcParamProposed, mcDatafit)
            k=k+1;
            # data residuals
            mcDatafit.dataResiduals = emData.obsData - mcDatafit.propPredData #计算残差
        end

        # record in chainarray
        updateChainArray!(mcArray, iterNo, mcParamCurrent, mcstep, mcstatus, mcDatafit) #记录链数组

    end
    println("Worker #$(rank): Accept Ratio:",k/totalsamples)
    return mcArray

end


#------------------------------------------------------------------------------
"""
    `resetChainModel(mcParam, mcParamProposed, mcDatafit)`

reset proposed chain model.
重设提出链模型
"""
function resetChainModel!(mcParam::T,
                          mcParamProposed::T,
                          mcDatafit::MCDataFit) where{T<:MCParameter}

    # reset proposed model 重设提出链模型
    updateMCParameter!(mcParam, mcParamProposed)

    # update predicted data for proposed model 更新提出模型的预测数据
    mcDatafit.propPredData    = copy(mcDatafit.currPredData) #将当前的预测数据拷贝到提出模型的预测数据里
    mcDatafit.propLikelihood  = mcDatafit.currLikelihood #将当前的似然函数拷贝到预测的似然函数里；
    mcDatafit.propMisfit  = mcDatafit.currMisfit #将当前的拟合差拷贝到预测的拟合差里；

    return mcParamProposed, mcDatafit

end


#-------------------------------------------------------------------------------
"""
    `updateChainModel!(mcParamCurrent, rjdatafit)`
    更新链模型
"""
function updateChainModel!(mcParamCurrent::MCParameter,
                          mcParamProposed::MCParameter,
                          rjdatafit::MCDataFit)

    # model update
    mcParamCurrent = updateMCParameter!(mcParamProposed, mcParamCurrent)  #更新MC参数，将MCParameter中的参数复制给MCparamNew

    #
    rjdatafit.currPredData   = copy(rjdatafit.propPredData) #将预测数据拷贝到当前数据里
    rjdatafit.currLikelihood = copy(rjdatafit.propLikelihood) #将预测的似然函数拷贝到当前的似然函数里
    rjdatafit.currMisfit     = copy(rjdatafit.propMisfit) # 将提出的拟合差拷贝到当前的拟合差里

    return mcParamCurrent, rjdatafit

end


#-------------------------------------------------------------------------------
"""
    `updateChainArray(mcArray, iterNo, mcParamCurrent, mcstep, mcstatus, mcDatafit)`
    更新链数组；
"""
function updateChainArray!(mcArray::MChainArray,
                          iterNo::Int,
                          mcParamCurrent::MCParameter,
                          mcstep::MChainStep,
                          mcstatus::MCStatus,
                          mcDatafit::MCDataFit)

    #
    nLayer   = mcParamCurrent.nLayer #当前的层数为n层；
    layeridx = mcParamCurrent.layeridx #要扰动的层数；
    mcArray.chainnlayer[iterNo]  = nLayer 
    mcArray.chainMisfit[iterNo]  = mcDatafit.currMisfit

    if mcstep.isBirth #判断是否为出生？
        mcArray.chainstep[iterNo, 1] = 1 #mcArray链步骤有两列，第一列记录四种方法的步骤，第二列记录是否接受，1接受，0不接受
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1 #第二列为接收状态，1表示接受，0表示不接受
            mcArray.chainindice[iterNo]  = 0 # 新加一层不记录变化层的索引，因为没法记录
            mcArray.chainvalue[iterNo, 1] = mcParamCurrent.zNode[nLayer] #mcArray链的值也有两列，第一列放最后一层界面深度？
            mcArray.chainvalue[iterNo, 2] = mcParamCurrent.rho[nLayer] #第二列放最后一层电阻率？
        end

    elseif mcstep.isDeath #判断mcstep是否为死亡
        mcArray.chainstep[iterNo, 1] = 2
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo]  = layeridx #链的目录设置为要扰动的层序号
            mcArray.chainvalue[iterNo, 1] = 0.#链值，第一列放0；
            mcArray.chainvalue[iterNo, 2] = 0.#链值，第二列放0；
        end

    elseif mcstep.isMove #判断mcstep是否为界面移动
        mcArray.chainstep[iterNo, 1] = 3 
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo] = layeridx #把链的目录设为要扰动层
            mcArray.chainvalue[iterNo, 1] = mcParamCurrent.zNode[layeridx] #把扰动后的层位置放入链值第一列
            mcArray.chainvalue[iterNo, 2] = 0. #将链值第二列设为0；
        end

    elseif mcstep.isPerturb #判断mcstep是否为电阻率扰动
        mcArray.chainstep[iterNo, 1] = 4
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo] = layeridx
            mcArray.chainvalue[iterNo, 1] = 0. #将链的第一列设为0；
            mcArray.chainvalue[iterNo, 2] = mcParamCurrent.rho[layeridx]# 将扰动的层的电阻率设为链的第二列
        end

    end

    # data residuals
    # if mcstatus.accepted
    #     mcArray.residuals[:, iterNo] = mcDatafit.dataResiduals
    # end

    # reset status
    mcstatus.accepted = false 

end


#------------------------------------------------------------------------------
"""
    `continueMCMCsampling(samplefile, mclimits, emData)`

continue a MCMC sampling from previous Markov chain.

"""
function continueMCMCsampling(samplefile::String, mclimits::MCPrior, emData::EMData)

    # initialize model parameter
    mcDatafit = initMCDataFit()
    mcstep    = MChainStep(false, false, false, false)
    mcstatus  = MCStatus(false,zeros(Int,4),zeros(Int,4))

    # get the last sample of previous Markov chain
    (mcParamCurrent, mcArrayPrev) = readsamplefile(samplefile)

    # compute predicted data
    currPredData = getPredData(emData, mcParamCurrent)
    residuals = emData.obsData - currPredData

    # calculate likelihood and data misfit
    (currLikelihood,currMisfit) = compDataMisfit(emData, currPredData)

    mcDatafit.currPredData   = currPredData
    mcDatafit.currLikelihood = currLikelihood
    mcDatafit.currMisfit = currMisfit

    # initialize proposed model
    mcParamProposed = duplicateMCParameter(mcParamCurrent)

    # start MCMC sampling
    mcArray = MCMCsampling!(mcParamCurrent, mcParamProposed, mcstep,
                            mclimits, mcstatus, mcDatafit, emData)

    return mcArray, mcstatus


end


#-------------------------------------------------------------------------------
"""
    `updateChainArray(mcArray, iterNo, mcParamCurrent, mcstep, mcstatus, mcDatafit)`

"""
function updateChainArray!(mcArray::MChainArray,
                          mcParamCurrent::MCParameter,
                          mcstep::MChainStep,
                          mcstatus::MCStatus,
                          mcDatafit::MCDataFit)

    #
    mcArray.nsample += 1
    iterNo   = mcArray.nsample
    nLayer   = mcParamCurrent.nLayer
    layeridx = mcParamCurrent.layeridx
    mcArray.chainnlayer[iterNo]  = nLayer
    mcArray.chainMisfit[iterNo]  = mcDatafit.currMisfit

    if mcstep.isBirth
        mcArray.chainstep[iterNo, 1] = 1
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo]  = 0
            mcArray.chainvalue[iterNo, 1] = mcParamCurrent.zNode[nLayer]
            mcArray.chainvalue[iterNo, 2] = mcParamCurrent.rho[nLayer]
        end

    elseif mcstep.isDeath
        mcArray.chainstep[iterNo, 1] = 2
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo]  = layeridx
            mcArray.chainvalue[iterNo, 1] = 0.
            mcArray.chainvalue[iterNo, 2] = 0.
        end

    elseif mcstep.isMove
        mcArray.chainstep[iterNo, 1] = 3
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo] = layeridx
            mcArray.chainvalue[iterNo, 1] = mcParamCurrent.zNode[layeridx]
            mcArray.chainvalue[iterNo, 2] = 0.
        end

    elseif mcstep.isPerturb
        mcArray.chainstep[iterNo, 1] = 4
        if mcstatus.accepted
            mcArray.chainstep[iterNo, 2] = 1
            mcArray.chainindice[iterNo] = layeridx
            mcArray.chainvalue[iterNo, 1] = 0.
            mcArray.chainvalue[iterNo, 2] = mcParamCurrent.rho[layeridx]
        end

    end

    # data residuals
    # if mcstatus.accepted
    #     mcArray.residuals[:, iterNo] = mcDatafit.dataResiduals
    # end

    # reset status
    mcstatus.accepted = false

end



#------------------------------------------------------------------------------
"""
    `readsamplefile(samplefile, mclimits)`

get the last sample of the Markov chain, which will be used as the current sample
for a new Markov chain.

"""
function readsamplefile(samplefile::String, mclimits::MCPrior)

    #
    if isfile(samplefile)
        fid = open(samplefile, "r")
    else
        error("$(samplefile) does not exist, please try again.")
    end

    # mcmc information
    mcArray = initMChainArray(mclimits)
    zNode   = []
    rho     = []
    nsample = 0
    while !eof(fid)
        cline = strip(readline(fid))

        # ignore all comments: empty line, or line preceded with !
        while cline[1] == '!' || isempty(cline)
            cline = strip(readline(fid))
        end
        cline = lowercase(cline)

        if occursin("zcoordinate", cline)
            tmp = split(cline)
            nz  = parse(Int, tmp[end])
            zNode = zeros(Float64, nz)
            cline = strip(readline(fid))
            cline = split(cline)
            for k = 1:nz
                zNode[k] = parse(Float64, cline[k])
            end

        elseif occursin("resistivity", cline)
            tmp = split(cline)
            nz  = parse(Int, tmp[end])
            rho = zeros(Float64, nz)
            cline = strip(readline(fid))
            cline = split(cline)
            for k = 1:nz
                rho[k] = parse(Float64, cline[k])
            end

        elseif occursin("samplelist", cline)
            tmp = split(cline)
            nsample = parse(Int, tmp[end])
            for k = 1:nsample
                cline = strip(readline(fid))
                cline = split(cline)
                mcArray.chainstep[k, :] = parse.(Float64, cline[2:3])
                mcArray.chainvalue[k,:] = parse.(Float64, cline[4:5])
                mcArray.chainindice[k]  = parse(Float64, cline[6])
                mcArray.chainncell[k]   = parse(Float64, cline[7])
                mcArray.chainMisfit[k]  = parse(Float64, cline[8])
            end

        end #

    end # while

    close(fid)

    # update model parameters
    for k = 1:nsample
        updatesampleModel!(zNode, rho, k, mcArray)
    end

    mcParam = initModelParameter(mclimits)
    mcParam.zNode = copy(zNode)
    mcParam.rho   = copy(rho)

    return mcParam, mcArray

end
