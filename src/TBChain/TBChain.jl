#-------------------------------------------------------------------------------
#
# module `TBChain` defines routines to perform transdimensional MCMC sampling.
#
#
#-------------------------------------------------------------------------------
module TBChain

using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBFileIO
using TransdEM.TBFwdSolver
using MPI, Distributed, DistributedArrays

#
using LinearAlgebra, Printf, Random

export mcBirth!, mcDeath!, mcMoveLocation!, mcPerturbProperty!
export checkPropertyBounds
export duplicateMCParameter, updateMCParameter!
export selectChainStep!
export TBTemperature

#-------------------------------------------------------------------------------
"""
    `duplicateMCParameter(mcParam)`

"""
function duplicateMCParameter(mcParam::MCParameter)
    # 复制MC参数
    #
    nLayer   = mcParam.nLayer #层数
    layeridx = mcParam.layeridx #层索引
    zNode    = copy(mcParam.zNode) #深度节点
    rho      = copy(mcParam.rho)  #电阻率节点
    inBound  = mcParam.inBound  #inBound 
    mcParamNew = MCParameter(nLayer, layeridx, zNode, rho, inBound) # 生成一个新参数

    return mcParamNew

end


#-------------------------------------------------------------------------------
"""
    `updateMCParameter(mcParam, mcParamNew)`
    更新MC参数,将MCParameter中的参数复制给MCparamNew
"""
function updateMCParameter!(mcParam::MCParameter, mcParamNew::MCParameter)

    #
    mcParamNew.nLayer   = mcParam.nLayer
    mcParamNew.layeridx = mcParam.layeridx
    mcParamNew.zNode    = copy(mcParam.zNode)
    mcParamNew.rho      = copy(mcParam.rho)
    mcParamNew.inBound  = mcParam.inBound

    return mcParamNew

end


#-------------------------------------------------------------------------------
"""
    `mcBirth!(mcParam, mcParamProposed, mclimits)`

add a new layer to current model.
    添加一个新层到当前模型
"""
function mcBirth!(mcParam::MCParameter,
                  mcParamProposed::MCParameter,
                  mclimits::MCPrior)

    # check if current layer number is within the prior bound
    currLayerNumber = mcParam.nLayer 
    if currLayerNumber == mclimits.nlayermax #判断此时的层数是否到最大层数了，如果到了就判断到边界了
        mcParamProposed.inBound = false
        return
    end

    #
    propLayerNumber  = currLayerNumber + 1 #当前层的数目加一
    mcParamProposed.nLayer = propLayerNumber

    # get random location for the new layer interface
    zLoc = 0.0
    iter = 0
    while iter < 1000 # 循环1000次是为了保证，添加的层厚大于hmin,不满足就继续添加
        iter += 1
        zLoc = unirandDouble(mclimits.zmin, mclimits.zmax) # 最大最小值此时都是对数值
        mcParamProposed.zNode[propLayerNumber] = zLoc # 新增层的深度位置放在位置数组的最后一层

        # check if meet the mimimum layer thickness constraint
        # 判断是否满足最小层厚限制
        if mclimits.hmin > 0.
            hmin = mclimits.hmin
            zNode = copy(mcParamProposed.zNode[1:propLayerNumber])
            sort!(zNode)
            zNode = 10 .^ zNode
            if minimum(diff(zNode)) > hmin  
                break
            end
        else
            break
        end
    end

    # get resistivity for the new layer by first finding the location nearest to
    # the new layer in the previous model, then perturb its resistivity
    layeridx = findLocation1D(zLoc, mcParam.zNode[1:currLayerNumber]) #找到新生成的层序号
    mcParam.layeridx = layeridx # 需要扰动层的索引
    mcParamProposed.layeridx = layeridx #下一次要扰动的层序号就是新增的这层

    # assign resistivity of the new layer
    drho = mclimits.rhostd * randn() #电阻率标准差乘以一个随机数；
    # 扰动时都以对数电阻率进行扰动
    mcParamProposed.rho[propLayerNumber] =  mcParam.rho[layeridx] + drho #给该扰动层一个电阻率值（加上扰动值后的值）

    # check if the new resistivity value in prior bounds
    mcParamProposed.inBound = checkPropertyBounds(mcParamProposed, mclimits) #

    if iter == 1000 #这里的1000表示，最多循环1000次；
        mcParamProposed.inBound = false
    end

    return mcParamProposed

end


#-------------------------------------------------------------------------------
"""
    `mcDeath!(mcParam, mcParamProposed, mclimits)`

delete an old layer from the current model.
从当前模型中删掉一个层；
"""
function mcDeath!(mcParam::MCParameter,
                  mcParamProposed::MCParameter,
                  mclimits::MCPrior)

    #
    currLayerNumber = mcParam.nLayer
    if currLayerNumber == mclimits.nlayermin  #判断当前层数，是否为最小层；
        mcParamProposed.inBound = false # 是的话就不在最小边界了
        return
    end

    #
    propLayerNumber = currLayerNumber - 1 #当前层数减一
    mcParamProposed.nLayer = propLayerNumber

    # randomly choose an old layer to delete
    layeridx = unirandInteger(1, currLayerNumber)
    mcParam.layeridx = layeridx
    mcParamProposed.layeridx = layeridx

    # replace the deleted layer with the last layer 用最后一层替换删掉层，
    # 把最后一层的电阻率数据赋值给删除层，再把最后一层的值赋值为0
    zLoc = mcParam.zNode[currLayerNumber] 
    mcParamProposed.zNode[layeridx] = zLoc
    mcParamProposed.rho[layeridx]   = mcParam.rho[currLayerNumber]

    # delete the chosen layer 
    mcParamProposed.zNode[currLayerNumber:mclimits.nlayermax] .= 0.  # 删除先前模型的最后一层到最大层深的数据
    mcParamProposed.rho[currLayerNumber:mclimits.nlayermax]   .= 0.  # 删除先前模型最后一层到最大层的电阻率

    return mcParamProposed
end


#-------------------------------------------------------------------------------
"""
    `mcMoveLocation!(mcParam, mcParamProposed)`

move the location of one layer interface in the current model.
 在当前模型中，移动层界面的位置；
"""
function mcMoveLocation!(mcParam::MCParameter,
                         mcParamProposed::MCParameter,
                         mclimits::MCPrior)

    # randomly choose a layer to perturb
    layeridx = unirandInteger(1, mcParam.nLayer) #随机选择一个层开始扰动；
    mcParam.layeridx = layeridx
    mcParamProposed.layeridx = layeridx

    # perturb the location of the chosen layer interface 对选定层的的层深加以扰动
    mcParamProposed.zNode[layeridx] = mcParam.zNode[layeridx] + mclimits.zstd * randn()

    # check if the new location in prior bounds 核对扰动的位置是否在先验边界里；
    mcParamProposed.inBound = checkPropertyBounds(mcParamProposed, mclimits)

    return mcParamProposed
end


#-------------------------------------------------------------------------------
"""
    `mcPerturbProperty!(mcParam, mcParamProposed)`

perturb the resistivity of one layer in the current model
扰动当前模型一层的电阻率
"""
function mcPerturbProperty!(mcParam::MCParameter,
                           mcParamProposed::MCParameter,
                           mclimits::MCPrior)

    # randomly choose a layer to perturb
    layeridx = unirandInteger(1, mcParam.nLayer)
    mcParam.layeridx = layeridx
    mcParamProposed.layeridx = layeridx

    # perturb the resistivity of the chosen layer 对选定层电阻率加以扰动
    mcParamProposed.rho[layeridx] = mcParam.rho[layeridx] + mclimits.rhostd * randn()

    # check if the new location in prior bounds
    mcParamProposed.inBound = checkPropertyBounds(mcParamProposed, mclimits)

    return mcParamProposed

end


#-------------------------------------------------------------------------------
"""
    `checkPropertyBounds(mcParam, mclimits)`

check if the value of corresponding variables within specified bounds.
    核对相应变量的值是否在具体的范围之内
"""
function checkPropertyBounds(mcParam::MCParameter, mclimits::MCPrior)

    #
    inBound = true
    nLayer  = mcParam.nLayer

    # check location of layer interface
    if !all(x->(mclimits.zmin<=x<=mclimits.zmax), mcParam.zNode[1:nLayer])
        inBound = false
    end

    # check layer resistivity
    if !all(x->(mclimits.rhomin<=x<=mclimits.rhomax), mcParam.rho[1:nLayer])
        inBound = false
    end

    return inBound

end


#------------------------------------------------------------------------------
"""
    `selectChainStep(iterNo)`

randomly select a type of model perturbation for model proposal.
    随机选择一种扰动类型
"""
function selectChainStep(iterNo::Int)

    #
    num = unirandInteger(1, 16)
    if num < 4
        step = 1
    elseif 3 < num < 7
        step = 2
    elseif 6 < num < 12
        step = 3
    else
        step = 4
    end

    return step

end #

#随机生成一种扰动状态
function selectChainStep!(chainstep::MChainStep, iterNo::Int=1)

    #
    chainstep.isBirth = false
    chainstep.isDeath = false
    chainstep.isMove  = false
    chainstep.isPerturb = false

    #
    step = selectChainStep(iterNo)

    if step == 1
        chainstep.isBirth = true
    elseif step == 2
        chainstep.isDeath = true
    elseif step == 3
        chainstep.isMove = true
    elseif step == 4
        chainstep.isPerturb = true
    else
        error("step $(step) is not supported")
    end

    return chainstep

end

"""
    struct `TBTemperature` encapsalates data related to parallel tempering

"""
mutable struct TBTemperature{Ti<:Int, Tv<:Float64}

    nT::Ti                      # total number of temperature values
    tempLadder::Vector{Tv}      # current temperature ladder

    # record swap history
    swapchain::Array{Ti}       # dimension=[3xnTxnsample]

end


#-------------------------------------------------------------------------------
include("acceptanceRatio.jl")
include("runMCMC.jl")
include("MPIUtilities.jl")
include("parallelTempering.jl")

end # module RJChains
