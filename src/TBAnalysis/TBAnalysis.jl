#-------------------------------------------------------------------------------
#
# module `TBAnalysis` defines routines to perform posterior analysis of model
# parameters.
#
#
#-------------------------------------------------------------------------------
module TBAnalysis

using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBChain

using LinearAlgebra, Printf, Statistics
using Distributed, DistributedArrays

export sampleStatistics
export multipleStatistics
export updatesampleModel!
export outputMCMCsamples
export outputEnsembleModel
export getCredibleIntervalModel
export mksampleArray
export extractModelParam
export Misfit_Statistics
export extractModelParamSingle

export outputPTDataMisfit
export outputPTchainHeader
export outputPTchains
export outputMPIPTchains
export samplePTStatistics
export sampleMPIPTStatistics
export clearPTrefence


#-------------------------------------------------------------------------------
"""
    `outputMCMCsamples(mcArray)`
    输出MCMC采样结果;
"""
function outputMCMCsamples(mcArray::MChainArray, ichain::Int=1)

    nsample = length(mcArray.chainMisfit) #采样数目
    chainnlayer = mcArray.chainnlayer #链的层数

    # chain values
    valuefile = "chainsamples_id$(ichain).dat"
    valID = open(valuefile, "w")
    chainstep = mcArray.chainstep #链的步骤
    chainindice = mcArray.chainindice #链的目录
    chainvalue  = mcArray.chainvalue #链的值

    # start model 开始模型
    @printf(valID,"! starting model \n")
    startModel  = mcArray.startModel
    nlayer   = startModel.nLayer #层数
    zNode   = startModel.zNode #界面深度序列
    rho = startModel.rho #电阻率序列
    @printf(valID, "zcoordinate: %5d \n", nlayer)
    for k = 1:nlayer
        @printf(valID, "%8g ", zNode[k])
    end
    @printf(valID, "\n")
    @printf(valID, "resistivity: %5d \n", nlayer)
    for k = 1:nlayer
        @printf(valID, "%8g ", rho[k])
    end
    @printf(valID, "\n")

    @printf(valID,"! #sample  step  accept layeridx zcoord(lg10)  rho(lg10)  nLayer  dataMisfit\n")
    @printf(valID, "samplelist: %8d\n", nsample)
    for i = 1:nsample
        @printf(valID, "%8d %6d %6d %6d ", i, chainstep[i,1], chainstep[i,2], chainindice[i])
        @printf(valID, "%8g %8g ", chainvalue[i, 1], chainvalue[i, 2]) # 写入层厚和层的电阻率
        @printf(valID, "%8d %12g \n", chainnlayer[i], mcArray.chainMisfit[i]) # 当前的层数和拟合差
    end
    close(valID)

end

function Misfit_Statistics(mcArray::MChainArray, ichain::Int=1)
    nsample = length(mcArray.chainMisfit) #采样数目
    valuefile = "Misfit_Statistics_id$(ichain).dat"
    valID = open(valuefile, "w")
    @printf(valID, " #sample  dataMisfit\n")
    for i = 1:nsample
        @printf(valID, "%8d %12g \n", i, mcArray.chainMisfit[i])
    end
end


#-------------------------------------------------------------------------------
"""
    `sampleStatistics(mcArray, mclimits)`
    样本统计函数
"""
function sampleStatistics(mcArray::MChainArray, mclimits::MCPrior,
                          filestr::String="mc")

    #
    burnin    = mclimits.burninsamples #预热采样数目
    nsamples  = mclimits.totalsamples # 总采样数目 
    nlayermin = mclimits.nlayermin #层数目最小值
    nlayermax = mclimits.nlayermax #层数目最大值
    zmin     = mclimits.zmin #深度最小值
    zmax     = mclimits.zmax #深度最大值
    rhomin   = mclimits.rhomin #电阻率最小值
    rhomax   = mclimits.rhomax #电阻率最大值

    # start model 开始模型
    startModel  = mcArray.startModel 
    zNode   = copy(startModel.zNode) #初始模型的深度序列
    rho     = copy(startModel.rho) #初始模型的电阻率序列

    # intervals
    nzBins = mclimits.nzBins #深度方格
    npBins = mclimits.npBins #电阻率方格
    (zLocBins, zspace) = mksampleArray(zmin, zmax, nzBins) #线性化，深度序列
    (rhoBins,rhospace) = mksampleArray(rhomin, rhomax, npBins) #线性化，电阻率序列

    for k = 1:burnin 
        updatesampleModel!(zNode, rho, k, mcArray) #更新模型函数
    end

    # calculate posteriors 计算后验概率
    rhoModel = zeros(nzBins) #电阻率模型初始化
    nlayerPosterior = zeros(Int, nlayermax) #n层后验概率初始化
    depthPosterior  = zeros(Int, nzBins) #深度后验概率初始化
    valuePosterior  = zeros(Int, nzBins, npBins) #值后验概率

    # statistical moment 统计时刻
    sm01 = zeros(nzBins) 
    sm02 = zeros(nzBins) 

    # collecte mcmc samples after the burnin period 在预热期后搜集mcmc样本
    samplestep = 100 
    # number of samples in ensemble  集合中样本的数量
    nEnsemble  = 0 # 计数器
    for i in burnin+1:nsamples 
        updatesampleModel!(zNode, rho, i, mcArray)

        #
        if mod(i, samplestep) > 0 #如果i是100的倍数，进行统计；
            continue # 不是100的整数倍就跳过该次循环
        end

        nEnsemble += 1
        # posterior layer number
        nlayer  = mcArray.chainnlayer[i]
        nlayerPosterior[nlayer] += 1 # 对不同层数的数量作一个统计

        # get depth posterior 获得深度后验
        for j = 1:nlayer
            zLoc = zNode[j] #
            cidx = findLocation1D(zLoc, zLocBins) #找到层界面在哪一个网格里
            depthPosterior[cidx] += 1  # 深度在某个深度统计箱格的个数统计
        end

        # get values at sampling location 获得采样位置的电阻率值
        for j = 1:nzBins 
            zLoc = zLocBins[j]
            cidx = findNearest1D(zLoc, zNode[1:nlayer])
            rhoModel[j] = rho[cidx]
            #
            idx = findLocation1D(rho[cidx], rhoBins)
            valuePosterior[j, idx] += 1 #统计值某个深度箱格和某个电阻率箱格处数值的个数
        end

        # statistical moments
        sm01 += rhoModel # 电阻率向量和
        sm02 += rhoModel .^ 2 # 电阻率向量平方和

    end

    # evaluate statistical variabless
    sm01 = sm01 / nEnsemble # 随机变量的期望
    sm02 = sm02 / nEnsemble # 随机变量平方的期望

    # arithmetic mean 算术均值
    valueMean = sm01

    # standard deviation
    std = sm02 - sm01 .^ 2 #方差：随机变量平方的期望-期望的平方
    std[std .< 0] .= eps(1.0) # 相当于赋值为0
    valueStd  = sqrt.(std) #标准差

    # mode = maximum a posterior 最大的后验概率
    valueMode = zeros(nzBins)
    for i = 1:nzBins
        (val,idx) = findmax(valuePosterior[i, :]) # 某个深度下，某个箱格内电阻率的最多的统计值
        valueMode[i] = rhoBins[idx] #统计众数
    end

    # meadian 统计中位数
    valueMedian = zeros(nzBins)
    for i = 1:nzBins
        count = 0 
        for j = 1:npBins
            count += valuePosterior[i, j]
            if count > nEnsemble / 2
                valueMedian[i] = rhoBins[j] #统计中位数，大于一半就终止
                break
            end
        end
    end

    # output results 将统计结果写入文件
    outputPosterior(zLocBins, rhoBins, nlayerPosterior, depthPosterior,
    valuePosterior, filestr)

    # credible min and max model 最小可信模型和最大可信模型
    ci = mclimits.credInterval
    # 得到置信区间最大最小的电阻率值
    (valueMin,valueMax) = getCredibleIntervalModel(valuePosterior, rhoBins,
                                                   nEnsemble, ci)
    #
    filename = "posteriorModel-"*filestr*".dat"
    results  = hcat(zLocBins,valueMean,valueMedian,valueMode,valueMin,valueMax,valueStd) #拼装矩阵
    outputEnsembleModel(filename, results) #输出采样模型

end # sampleStatistics

#-------------------------------------------------------------------------------
"""
    `samplePTStatistics(mcArrayRef, tempData, mclimits, filestr)`

calculate statistical models from parallel tempering MCMC results.

"""
function samplePTStatistics(mcArrayRef::DArray, tempData::TBTemperature,
                            mclimits::MCPrior, filestr::String="mc")

    #
    burnin    = mclimits.burninsamples
    nsamples  = mclimits.totalsamples
    nlayermin = mclimits.nlayermin
    nlayermax = mclimits.nlayermax
    zmin     = mclimits.zmin
    zmax     = mclimits.zmax
    rhomin   = mclimits.rhomin
    rhomax   = mclimits.rhomax

    # intervals
    nzBins = mclimits.nzBins
    npBins = mclimits.npBins
    (zLocBins, zspace) = mksampleArray(zmin, zmax, nzBins)
    (rhoBins,rhospace) = mksampleArray(rhomin, rhomax, npBins)

    # multiple chains
    nchain = length(mcArrayRef)

    # calculate posteriors
    rhoModel = zeros(nzBins)
    nlayerPosterior = zeros(Int, nlayermax)
    depthPosterior  = zeros(Int, nzBins)
    valuePosterior  = zeros(Int, nzBins, npBins)

    # statistical moment
    sm01 = zeros(nzBins)
    sm02 = zeros(nzBins)

    # collecte mcmc samples after the burnin period
    samplestep = 100

    # number of samples in ensemble
    nEnsemble  = 0
    tempLadder = copy(tempData.tempLadder)
    chainidx   = collect(1:nchain)
    sortchain  = collect(1:nchain)

    #
    mcArray    = Array{MChainArray}(undef, nchain)
    dataMisfit = zeros(nsamples, nchain)
    swapchain  = tempData.swapchain
    zNodeArray = Array{Any}(undef, nchain)
    rhoArray   = Array{Any}(undef, nchain)
    for ic = 1:nchain
        mcArray[ic] = mcArrayRef[ic]
        dataMisfit[:, ic] = mcArray[ic].chainMisfit
        zNodeArray[ic] = copy(mcArray[ic].startModel.zNode)
        rhoArray[ic]   = copy(mcArray[ic].startModel.rho)
    end

    #
    for k = 1:nsamples

        for ic = 1:nchain
            # update model parameters
            updatesampleModel!(zNodeArray[ic], rhoArray[ic], k, mcArray[ic])
            if swapchain[3,ic,k] > 0
                idx01 = [ swapchain[1,ic,k], swapchain[2,ic,k] ]
                idx02 = [ swapchain[2,ic,k], swapchain[1,ic,k] ]
                chainidx[idx01] = chainidx[idx02]

                # update chain temperature
                tempLadder[idx01] = tempLadder[idx02]
            end
        end
        ind = indexin(sortchain, chainidx)
        dataMisfit[k, :] = dataMisfit[k, ind]

        # collect samples after burnin period
        if k <= burnin
            continue
        else
            if mod(k, samplestep) > 0; continue; end

            for ic = 1:nchain
                if tempLadder[ic] > 1.0; continue; end
                nEnsemble += 1

                # posterior cell number
                nlayer  = mcArray[ic].chainnlayer[k]
                nlayerPosterior[nlayer] += 1

                # get depth posterior
                for j = 1:nlayer
                    zLoc = zNodeArray[ic][j]
                    cidx = findLocation1D(zLoc, zLocBins)
                    depthPosterior[cidx] += 1
                end

                # get values at sampling location
                for j = 1:nzBins
                    zLoc = zLocBins[j]
                    cidx = findNearest1D(zLoc, zNodeArray[ic][1:nlayer])
                    rhoModel[j] = rhoArray[ic][cidx]
                    idx = findLocation1D(rhoArray[ic][cidx], rhoBins)
                    valuePosterior[j, idx] += 1

                end

                # statistical moments
                sm01 += rhoModel
                sm02 += rhoModel .^ 2

            end # ic

        end # if

    end # k

    # evaluate statistical variables
    sm01 = sm01 / nEnsemble
    sm02 = sm02 / nEnsemble

    # arithmetic mean
    valueMean = sm01

    # standard deviation
    std = sm02 - sm01 .^ 2
    std[std .< 0] .= eps(1.0)
    valueStd  = sqrt.(std)

    # mode = maximum a posterior
    valueMode = zeros(nzBins)
    for i = 1:nzBins
        (val,idx) = findmax(valuePosterior[i,:])
        valueMode[i] = rhoBins[idx]
    end

    # meadian
    valueMedian = zeros(nzBins)
    for i = 1:nzBins
        count = 0
        for j = 1:npBins
            count += valuePosterior[i, j]
            if count > nEnsemble / 2
                valueMedian[i] = rhoBins[j]
                break
            end
        end
    end

    # output results
    outputPosterior(zLocBins, rhoBins, nlayerPosterior, depthPosterior,
    valuePosterior, filestr)

    # credible min and max model
    ci = mclimits.credInterval
    (valueMin,valueMax) = getCredibleIntervalModel(valuePosterior, rhoBins,
                                                   nEnsemble, ci)

    # output results
    filename = "posteriorModel-"*filestr*".dat"
    results  = hcat(zLocBins,valueMean,valueMedian,valueMode,valueMin,valueMax,valueStd)
    comments = "#z(lg10) mean(lg10) median(lg10) mode(lg10) credmin(lg10) credmax(lg10) std"
    outputEnsembleModel(filename, results, comments)

    # data misfit
    filename = "chainMisfit-"*filestr*".dat"
    outputEnsembleModel(filename, dataMisfit)

    return nEnsemble

end


#-------------------------------------------------------------------------------
"""
    `sampleMPIPTStatistics(mcArrayArray, tempData, mclimits, filestr)`

calculate statistical models from parallel MPI tempering MCMC results.

"""
function sampleMPIPTStatistics(mcArrayArray::Array, tempData::TBTemperature,
                            mclimits::MCPrior, filestr::String="mc")

    #
    burnin    = mclimits.burninsamples
    nsamples  = mclimits.totalsamples
    nlayermin = mclimits.nlayermin
    nlayermax = mclimits.nlayermax
    zmin     = mclimits.zmin
    zmax     = mclimits.zmax
    rhomin   = mclimits.rhomin
    rhomax   = mclimits.rhomax

    # intervals
    nzBins = mclimits.nzBins
    npBins = mclimits.npBins
    (zLocBins, zspace) = mksampleArray(zmin, zmax, nzBins)
    (rhoBins,rhospace) = mksampleArray(rhomin, rhomax, npBins)

    # multiple chains
    nchain = length(mcArrayArray)

    # calculate posteriors
    rhoModel = zeros(nzBins)
    nlayerPosterior = zeros(Int, nlayermax)
    depthPosterior  = zeros(Int, nzBins)
    valuePosterior  = zeros(Int, nzBins, npBins)

    # statistical moment
    sm01 = zeros(nzBins)
    sm02 = zeros(nzBins)

    # collecte mcmc samples after the burnin period
    samplestep = 100

    # number of samples in ensemble
    nEnsemble  = 0
    tempLadder = copy(tempData.tempLadder)
    chainidx   = collect(1:nchain)
    sortchain  = collect(1:nchain)

    #
    mcArray    = Array{MChainArray}(undef, nchain)
    dataMisfit = zeros(nsamples, nchain)
    swapchain  = tempData.swapchain
    zNodeArray = Array{Any}(undef, nchain)
    rhoArray   = Array{Any}(undef, nchain)
    for ic = 1:nchain
        mcArray[ic] = mcArrayArray[ic]
        dataMisfit[:, ic] = mcArray[ic].chainMisfit
        zNodeArray[ic] = copy(mcArray[ic].startModel.zNode)
        rhoArray[ic]   = copy(mcArray[ic].startModel.rho)
    end

    #
    for k = 1:nsamples

        for ic = 1:nchain
            # update model parameters
            updatesampleModel!(zNodeArray[ic], rhoArray[ic], k, mcArray[ic])
            if swapchain[3,ic,k] > 0
                idx01 = [ swapchain[1,ic,k], swapchain[2,ic,k] ]
                idx02 = [ swapchain[2,ic,k], swapchain[1,ic,k] ]
                chainidx[idx01] = chainidx[idx02]

                # update chain temperature
                tempLadder[idx01] = tempLadder[idx02]
            end
        end
        ind = indexin(sortchain, chainidx)
        dataMisfit[k, :] = dataMisfit[k, ind]

        # collect samples after burnin period
        if k <= burnin
            continue
        else
            if mod(k, samplestep) > 0; continue; end

            for ic = 1:nchain
                if tempLadder[ic] > 1.0; continue; end
                nEnsemble += 1

                # posterior cell number
                nlayer  = mcArray[ic].chainnlayer[k]
                nlayerPosterior[nlayer] += 1

                # get depth posterior
                for j = 1:nlayer
                    zLoc = zNodeArray[ic][j]
                    cidx = findLocation1D(zLoc, zLocBins)
                    depthPosterior[cidx] += 1
                end

                # get values at sampling location
                for j = 1:nzBins
                    zLoc = zLocBins[j]
                    cidx = findNearest1D(zLoc, zNodeArray[ic][1:nlayer])
                    rhoModel[j] = rhoArray[ic][cidx]
                    idx = findLocation1D(rhoArray[ic][cidx], rhoBins)
                    valuePosterior[j, idx] += 1

                end

                # statistical moments
                sm01 += rhoModel
                sm02 += rhoModel .^ 2

            end # ic

        end # if

    end # k

    # evaluate statistical variables
    sm01 = sm01 / nEnsemble
    sm02 = sm02 / nEnsemble

    # arithmetic mean
    valueMean = sm01

    # standard deviation
    std = sm02 - sm01 .^ 2
    std[std .< 0] .= eps(1.0)
    valueStd  = sqrt.(std)

    # mode = maximum a posterior
    valueMode = zeros(nzBins)
    for i = 1:nzBins
        (val,idx) = findmax(valuePosterior[i,:])
        valueMode[i] = rhoBins[idx]
    end

    # meadian
    valueMedian = zeros(nzBins)
    for i = 1:nzBins
        count = 0
        for j = 1:npBins
            count += valuePosterior[i, j]
            if count > nEnsemble / 2
                valueMedian[i] = rhoBins[j]
                break
            end
        end
    end

    # output results
    outputPosterior(zLocBins, rhoBins, nlayerPosterior, depthPosterior,
    valuePosterior, filestr)

    # credible min and max model
    ci = mclimits.credInterval
    (valueMin,valueMax) = getCredibleIntervalModel(valuePosterior, rhoBins,
                                                   nEnsemble, ci)

    # output results
    filename = "posteriorModel-"*filestr*".dat"
    results  = hcat(zLocBins,valueMean,valueMedian,valueMode,valueMin,valueMax,valueStd)
    comments = "#z(lg10) mean(lg10) median(lg10) mode(lg10) credmin(lg10) credmax(lg10) std"
    outputEnsembleModel(filename, results, comments)

    # # data misfit
    filename = "chainMisfit-"*filestr*".dat"
    outputDataMisfit(filename, dataMisfit)

    return nEnsemble

end


#-------------------------------------------------------------------------------
"""
    `updatesampleModel!(zNode, rho, k, mcArray)`
    更新样本模型
"""
function updatesampleModel!(zNode::Vector{T}, rho::Vector{T}, k::Int,
                           mcArray::MChainArray) where{T<:Float64}

    mcstep = mcArray.chainstep[k, 1] # mc步骤：扰动情况
    accept = mcArray.chainstep[k, 2] # 是否接受
    nlayer = mcArray.chainnlayer[k]  # 模型层数
    cidx   = mcArray.chainindice[k]  # 扰动的层数
    nlayermax = length(zNode)   # 最大层数

    # birth step
    if mcstep==1 && accept>0
        zNode[nlayer] = mcArray.chainvalue[k, 1] #        
        rho[nlayer]   = mcArray.chainvalue[k, 2] # 
        zNode[nlayer+1:nlayermax] .= 0. # n+1层的界面设为零
        rho[nlayer+1:nlayermax]   .= 0. # n+1层的电阻率设为零

    # death step
    elseif mcstep==2 && accept>0 
        zNode[cidx] = zNode[nlayer+1] #将n+1层的界面位置替换要删除的界面位置
        rho[cidx]   = rho[nlayer+1]   #将n+1层的电阻率值替换要扰动的层的电阻率值
        zNode[nlayer+1:nlayermax] .= 0. #将n+1层到最大层的界面位置都设为0；
        rho[nlayer+1:nlayermax]   .= 0. #将n+1层到最大层的电阻率值都设为0；

    # move step
    elseif mcstep==3 && accept>0 
        zNode[cidx] = mcArray.chainvalue[k, 1] #将扰动的层界面深度赋值

    # perturb step
    elseif mcstep==4 && accept>0
        rho[cidx] = mcArray.chainvalue[k, 2] #将扰动的层电阻率值赋值

    end

end # updatesampleModel


#-------------------------------------------------------------------------------
"""
    `outputEnsembleModel(filename, coords, results)`
    输出采样模型
"""
function outputEnsembleModel(filename::String, coords::Vector, results::Vector)

    fileID = open(filename, "w")

    for k = 1:length(coords)
        @printf(fileID, "%12g %12g \n", coords[k], results[k])
    end

    close(fileID)

end


function outputEnsembleModel(filename::String, results::Array{Float64})

    fileID = open(filename, "w")
    @printf(fileID, "#depth(lg10)  mean(lg10) meadian     mode      credmin    credmax     std \n")
    (nBins, nm) = size(results)
    if nm != 7
        error("The number of model should be 7.")
    end
    for i = 1:nBins
        for j = 1:nm
            @printf(fileID, "%10g ", results[i,j])
        end
        @printf(fileID, "\n")
    end

    close(fileID)

end

function outputDataMisfit(filename::String, results::Array{Float64})

    fileID = open(filename, "w")
    (nBins, nm) = size(results)
    for i = 1:nBins
        for j = 1:nm
            @printf(fileID, "%10g ", results[i,j])
        end
        @printf(fileID, "\n")
    end

    close(fileID)

end

# #-------------------------------------------------------------------------------
# """
#     `outputEnsembleModel(filename, results, comments)`

# """
function outputEnsembleModel(filename::String, results::Array{Float64}, comments::String)

    fileID = open(filename, "w")
    if !isempty(comments)
        @printf(fileID, "%s\n", comments)
    end

    (nBins, nm) = size(results)
    for i = 1:nBins
        for j = 1:nm
            @printf(fileID, "%10g ", results[i,j])
        end
        @printf(fileID, "\n")
    end

    close(fileID)

end

#-------------------------------------------------------------------------------
"""
    `outputPosterior(coordBins, rhoBins, nlayerPosterior, depthPosterior,
    valuePosterior)`
    输出后验概率结果
"""
function outputPosterior(coordBins, rhoBins, nlayerPosterior,
    depthPosterior, valuePosterior, filestr)

    filename ="depthBins-"*filestr*".dat" #写深度网格文件
    fid = open(filename, "w")
    for k = 1:length(coordBins)
        @printf(fid, "%12g\n", coordBins[k])
    end
    close(fid)

    filename ="rhoBins-"*filestr*".dat" #写电阻率网格文件
    fid = open(filename, "w")
    for k = 1:length(rhoBins)
        @printf(fid, "%12g\n", rhoBins[k])
    end
    close(fid)

    filename ="nlayerHistogram-"*filestr*".dat" #写入层的后验概率统计
    fid = open(filename, "w")
    for k = 1:length(nlayerPosterior)
        @printf(fid, "%4d %8d\n", k, nlayerPosterior[k])
    end
    close(fid)

    filename = "depthHistogram-"*filestr*".dat" #写入深度网格的后验概率
    fid = open(filename, "w")
    for k = 1:length(depthPosterior)
        @printf(fid, "%8d \n", depthPosterior[k])
    end
    close(fid)

    filename = "depthrhoHistogram-"*filestr*".dat" #写入深度-电阻率直方图
    fid = open(filename, "w") #行是电阻率，列是深度
    for i = 1:length(rhoBins)
        for j = 1:length(coordBins)
            @printf(fid, "%8d ", valuePosterior[j, i])
        end
        @printf(fid, "\n")
    end
    close(fid)

end


#-------------------------------------------------------------------------------
"""
    `getCredibleIntervalModel(valuePosterior, rhoBins, nEnsemble, credInterval)`

get the minimum and maximum credible model.
 得到最小和最大可信模型
"""
function getCredibleIntervalModel(valuePosterior::Array{Ti},
                                  rhoBins::Vector{Tv},
                                  nEnsemble::Ti,
                                  credInterval::Tv) where {Ti<:Int,Tv<:Float64}

    #
    (nzBins,npBins) = size(valuePosterior)
    valueMin = zeros(nzBins)
    valueMax = zeros(nzBins)
    credsample = nEnsemble * (1 - credInterval) / 2 #置信样本基数

    # minimum credible model
    for i = 1:nzBins
        count = 0
        for j = 1:npBins # 一个从最小的电阻率开始
            count += valuePosterior[i, j]
            if count > credsample
                valueMin[i] = rhoBins[j]
                break
            end
        end
    end

    # maximum credible model
    for i = 1:nzBins
        count = 0
        for j = npBins:-1:1 # 一个从最大的电阻率开始
            count += valuePosterior[i, j]
            if count > credsample
                valueMax[i] = rhoBins[j]
                break
            end
        end
    end

    return valueMin, valueMax

end


#-------------------------------------------------------------------------------
function mksampleArray(vmin::T, vmax::T, nsample::Int) where{T<:Number}
    #制作采样序列 
    dx = (vmax-vmin) / nsample
    x1 = vmin + 0.5 * dx
    x2 = vmax - 0.5 * dx #这两步操作啥意思？
    val = LinRange(x1, x2, nsample) #线性划分

    return collect(val), dx #这里的collect 是啥意思？

end


#-------------------------------------------------------------------------------
"""
    `multipleStatistics(mcArray, mclimits, nData, filestr)`

"""
function multipleStatistics(results::Array, mclimits::MCPrior, nData::Int,
                            filestr::String="mc", rscale::Float64=1.2)

    #
    burnin    = mclimits.burninsamples
    nsamples  = mclimits.totalsamples
    nlayermin = mclimits.nlayermin
    nlayermax = mclimits.nlayermax
    zmin     = mclimits.zmin
    zmax     = mclimits.zmax
    rhomin   = mclimits.rhomin
    rhomax   = mclimits.rhomax

    # multiple chains
    nchain = length(results)

    # calculate posteriors
    nzBins = mclimits.nzBins
    npBins = mclimits.npBins
    (zLocBins, zspace) = mksampleArray(zmin, zmax, nzBins)
    (rhoBins,rhospace) = mksampleArray(rhomin, rhomax, npBins)

    # calculate posteriors
    rhoModel = zeros(nzBins)
    nlayerPosterior = zeros(Int, nlayermax)
    depthPosterior  = zeros(Int, nzBins)
    valuePosterior  = zeros(Int, nzBins, npBins)

    # statistical moment
    sm01 = zeros(nzBins)
    sm02 = zeros(nzBins)

    # collecte mcmc samples after the burnin period
    samplestep = 100

    # number of samples in ensemble
    nEnsemble  = 0
    for ic = 1:nchain

        mcArray = results[ic]
        # exclude chains that are not convergent
        if mcArray.chainMisfit[end] > rscale * nData
            println("chain No.$(ic) has been excluded!")
            continue
        end
        #
        println("Output sampling results for chain No.$(ic) ...")
        outputMCMCsamples(mcArray, ic)

        # start model
        startModel  = mcArray.startModel
        zNode   = copy(startModel.zNode)
        rho     = copy(startModel.rho)

        # update model parameters
        for k = 1:burnin
            updatesampleModel!(zNode, rho, k, mcArray)
        end

        for i in burnin+1:nsamples
            updatesampleModel!(zNode, rho, i, mcArray)

            # exclude sample that is not convergent
            if mcArray.chainMisfit[i] > rscale * nData
                continue
            end

            #
            if mod(i, samplestep) > 0; continue; end
            nEnsemble += 1

            # posterior cell number
            nlayer  = mcArray.chainnlayer[i]
            nlayerPosterior[nlayer] += 1

            # get depth posterior
            for j = 1:nlayer
                zLoc = zNode[j]
                cidx = findLocation1D(zLoc, zLocBins)
                depthPosterior[cidx] += 1
            end

            # get values at sampling location
            for j = 1:nzBins
                zLoc = zLocBins[j]
                cidx = findNearest1D(zLoc, zNode[1:nlayer])
                rhoModel[j] = rho[cidx]
                idx = findLocation1D(rho[cidx], rhoBins)
                valuePosterior[j, idx] += 1

            end

            # statistical moments
            sm01 += rhoModel
            sm02 += rhoModel .^ 2

        end

    end # nchain

    # evaluate statistical variables
    sm01 = sm01 / nEnsemble
    sm02 = sm02 / nEnsemble

    # arithmetic mean
    valueMean = sm01

    # standard deviation
    std = sm02 - sm01 .^ 2
    std[std .< 0] .= eps(1.0)
    valueStd  = sqrt.(std)

    # mode = maximum a posterior
    valueMode = zeros(nzBins)
    for i = 1:nzBins
        (val,idx) = findmax(valuePosterior[i,:])
        valueMode[i] = rhoBins[idx]
    end

    # meadian
    valueMedian = zeros(nzBins)
    for i = 1:nzBins
        count = 0
        for j = 1:npBins
            count += valuePosterior[i, j]
            if count > nEnsemble / 2
                valueMedian[i] = rhoBins[j]
                break
            end
        end
    end

    # output results
    outputPosterior(zLocBins, rhoBins, nlayerPosterior, depthPosterior,
    valuePosterior, filestr)

    # credible min and max model
    ci = mclimits.credInterval
    (valueMin,valueMax) = getCredibleIntervalModel(valuePosterior, rhoBins,
                                                   nEnsemble, ci)

    #
    filename = "posteriorModel-"*filestr*".dat"
    results  = hcat(zLocBins,valueMean,valueMedian,valueMode,valueMin,valueMax,valueStd)
    outputEnsembleModel(filename, results)

end # multipleStatistics


#-------------------------------------------------------------------------------
"""
    `extractModelParam(mcArray, mclimits, nLayer)`
    提取模型参数
"""
function extractModelParam(mcArray::MChainArray, mclimits::MCPrior, nLayer::Int)

    #
    burnin    = mclimits.burninsamples #预热采样次数
    nsamples  = mclimits.totalsamples  #总采样次数

    # start model
    startModel  = mcArray.startModel
    zNode   = copy(startModel.zNode)
    rho     = copy(startModel.rho)

    for k = 1:burnin
        updatesampleModel!(zNode, rho, k, mcArray)
    end

    # collecte mcmc samples after the burnin period
    samplestep = 100

    #
    nidx = findall(mcArray.chainnlayer[burnin:samplestep:nsamples] .== nLayer) #层数等于nlayer的模型的索引
    num  = length(nidx) #模型数量
    nrow = 2*nLayer - 1 # 电阻率信息个数和深度信息个数
    km   = 1  #计数器
    modparam = Array{Float64, 2}(undef, nrow, num)
    for i in burnin+1:nsamples

        updatesampleModel!(zNode, rho, i, mcArray)
        if mod(i, samplestep) > 0; continue; end # 每隔samplestep次才采样一次，否则跳过

        # posterior layer number
        pnlayer = mcArray.chainnlayer[i]
        if pnlayer == nLayer
            idx = sortperm(zNode[1:nLayer]) #将层的深度序列排序
            dep1D = zNode[idx]
            modparam[1:nLayer, km]     = copy(rho[idx]) #先放电阻率，再放界面深度位置
            modparam[nLayer+1:nrow,km] = copy(dep1D[1:nLayer-1])
            km = km + 1
        end

    end

    return modparam

end


#-------------------------------------------------------------------------------
function extractModelParam(results::Array, mclimits::MCPrior, nLayer::Int)

    # number of chains
    nchain = length(results)

    filename = "modelParam_$(nLayer)layer.dat"
    fileID   = open(filename, "w")
    for ic = 1:nchain
        modparam = extractModelParam(results[ic], mclimits, nLayer)
        (nr,nc)  = size(modparam)
        for j = 1:nc
            for k = 1:nr
                @printf(fileID, "%8g ", modparam[k,j])
            end
            @printf(fileID, "\n")
        end
    end
    close(fileID)

end


#-------------------------------------------------------------------------------
"""
    `outputPTDataMisfit(mcArrayRef, tempData, mclimits, filestr)`

output data misfit of the samples in each Markov chain for convergence analysis.

"""
function outputPTDataMisfit(mcArrayRef::DArray, tempData::TBTemperature,
                            mclimits::MCPrior, filestr::String="mc")

    #
    nsamples  = mclimits.totalsamples
    nchain = length(mcArrayRef)

    #
    dataMisfit = zeros(nsamples, nchain)
    swapchain  = tempData.swapchain

    for ic = 1:nchain
        dataMisfit[:, ic] = mcArrayRef[ic].chainMisfit
    end

    #
    for k = 1:nsamples
        for ic = 1:nchain
            if swapchain[3,ic,k] > 0
                idx01 = [swapchain[1,ic,k],swapchain[2,ic,k]]
                idx02 = [swapchain[2,ic,k],swapchain[1,ic,k]]
                chainidx[idx01] = chainidx[idx02]
            end
        end
        ind = indexin(sortchain, chainidx)
        dataMisfit[k, :] = dataMisfit[k, ind]
    end

    # output
    filename = "chainMisfit-"*filestr*".dat"
    fileID   = open(filename, "w")
    for k = 1:nsamples
        for ic = 1:nchain
            @printf(fileID, "%8g ", dataMisfit[k,ic])
        end
        @printf(fileID, "\n")
    end

    close(fileID)

end


#-------------------------------------------------------------------------------
"""
    `outputPTchainHeader(mcArray, chainidx)`

"""
function outputPTchainHeader(mcArray::Array{MChainArray}, chainidx::Vector, filestr::String="mc")

    #
    nchain = length(chainidx)
    valID  = Array{IOStream}(undef, nchain)
    for ic = 1:nchain

        id = chainidx[ic]
        valuefile = filestr*"_chainsamples_id$(ic).dat"
        valID[ic] = open(valuefile, "w")
        chainstep = mcArray[id].chainstep
        chainindice = mcArray[id].chainindice
        chainvalue  = mcArray[id].chainvalue
        nsamples    = length(mcArray[id].chainMisfit)
        # sart model
        @printf(valID[ic],"! starting model \n")
        startModel  = mcArray[id].startModel
        nlayer   = startModel.nLayer
        zNode   = startModel.zNode
        rho = startModel.rho
        @printf(valID[ic], "zcoordinate: %5d \n", nlayer)
        for k = 1:nlayer
            @printf(valID[ic], "%8g ", zNode[k])
        end
        @printf(valID[ic], "\n")
        @printf(valID[ic], "resistivity: %5d \n", nlayer)
        for k = 1:nlayer
            @printf(valID[ic], "%8g ", rho[k])
        end
        @printf(valID[ic], "\n")

        #
        @printf(valID[ic],"! #sample  step  accept layeridx zcoord(lg10)  rho(lg10)  nLayer  dataMisfit\n")
        @printf(valID[ic], "samplelist: %8d\n", nsamples)
        k = 1
        @printf(valID[ic], "%8d %6d %6d %6d ", k, mcArray[id].chainstep[k, 1],
        mcArray[ic].chainstep[k, 2], mcArray[id].chainindice[k])
        @printf(valID[ic], "%8g %8g ", mcArray[id].chainvalue[k,1],
        mcArray[id].chainvalue[k, 2])
        @printf(valID[ic], "%8d %12g \n", mcArray[id].chainnlayer[k],
        mcArray[id].chainMisfit[k])

    end

    return valID

end


#-------------------------------------------------------------------------------
"""
    `outputPTchains(mcArrayRef, tempData)`

write out Markov chains in parallel tempering into file.

"""
function outputPTchains(mcArrayRef::DArray, tempData::TBTemperature, filestr::String="mc")

    #
    nchain     = length(mcArrayRef)
    swapchain  = tempData.swapchain
    chainidx   = collect(1:nchain)
    sortchain  = collect(1:nchain)
    mcArray    = Array{MChainArray}(undef, nchain)
    valID      = Array{IOStream}(undef, nchain)

    #
    for ic = 1:nchain
        mcArray[ic] = mcArrayRef[ic]
    end
    nsamples = length(mcArray[1].chainMisfit)
    for k = 1:nsamples

        for ic = 1:nchain
            if swapchain[3,ic,k] > 0
                idx01 = [ swapchain[1,ic,k], swapchain[2,ic,k] ]
                idx02 = [ swapchain[2,ic,k], swapchain[1,ic,k] ]
                chainidx[idx01] = chainidx[idx02]
            end
        end
        ind = indexin(sortchain, chainidx)

        # output samples
        if k < 2
            valID = outputPTchainHeader(mcArray, ind, filestr)
        else
            for ic = 1:nchain
                id = ind[ic]
                @printf(valID[ic], "%8d %6d %6d %6d ", k, mcArray[id].chainstep[k, 1],
                mcArray[ic].chainstep[k, 2], mcArray[id].chainindice[k])
                @printf(valID[ic], "%8g %8g ", mcArray[id].chainvalue[k,1],
                mcArray[id].chainvalue[k, 2])
                @printf(valID[ic], "%8d %12g \n", mcArray[id].chainnlayer[k],
                mcArray[id].chainMisfit[k])
            end
        end
    end

    #
    for ic = 1:nchain
        close(valID[ic])
    end

end

#-------------------------------------------------------------------------------
"""
    `outputMPIPTchains(mcArrayRef, tempData)`

write out Markov chains in parallel MPI tempering into file.

"""
function outputMPIPTchains(mcArrayArray::Array, tempData::TBTemperature, filestr::String="mc")

    #
    nchain     = length(mcArrayArray)
    swapchain  = tempData.swapchain
    chainidx   = collect(1:nchain)
    sortchain  = collect(1:nchain)
    mcArray    = Array{MChainArray}(undef, nchain)
    valID      = Array{IOStream}(undef, nchain)

    #
    for ic = 1:nchain
        mcArray[ic] = mcArrayArray[ic]
    end
    nsamples = length(mcArray[1].chainMisfit)
    for k = 1:nsamples

        for ic = 1:nchain
            if swapchain[3,ic,k] > 0
                idx01 = [ swapchain[1,ic,k], swapchain[2,ic,k] ]
                idx02 = [ swapchain[2,ic,k], swapchain[1,ic,k] ]
                chainidx[idx01] = chainidx[idx02]
            end
        end
        ind = indexin(sortchain, chainidx)

        # output samples
        if k < 2
            valID = outputPTchainHeader(mcArray, ind, filestr)
        else
            for ic = 1:nchain
                id = ind[ic]
                @printf(valID[ic], "%8d %6d %6d %6d ", k, mcArray[id].chainstep[k, 1],
                mcArray[ic].chainstep[k, 2], mcArray[id].chainindice[k])
                @printf(valID[ic], "%8g %8g ", mcArray[id].chainvalue[k,1],
                mcArray[id].chainvalue[k, 2])
                @printf(valID[ic], "%8d %12g \n", mcArray[id].chainnlayer[k],
                mcArray[id].chainMisfit[k])
            end
        end
    end

    #
    for ic = 1:nchain
        close(valID[ic])
    end

end


function extractModelParamSingle(results::MChainArray, mclimits::MCPrior, nLayer::Int)

    # number of chains
    filename = "modelParam_$(nLayer)layersingle.dat"
    fileID   = open(filename, "w")
    
    modparam = extractModelParam(results, mclimits, nLayer)
    (nr,nc)  = size(modparam)
    for j = 1:nc
        for k = 1:nr
            @printf(fileID, "%8g ", modparam[k,j])
        end
        @printf(fileID, "\n")
    end
    
    close(fileID)

end

#------------------------------------------------------------------------------
"""
    `clearPTrefence()`

release all darrays created during the inversion explicitly.

"""
function clearPTrefence(mcRef::DArray, stRef::DArray, dfRef::DArray, paramRef::DArray)

    #
    close(mcRef)
    close(stRef)
    close(dfRef)
    close(paramRef)
    mcRef = []
    stRef = []
    dfRef = []
    paramRef = []
    d_closeall()

end

#-------------------------------------------------------------------------------

end # TBAnalysis
