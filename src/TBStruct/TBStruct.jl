#-------------------------------------------------------------------------------
#
# module `TBStruct` defines data structure and routines for transdimensional
# MCMC sampling.
#
#
#-------------------------------------------------------------------------------
module TBStruct

using TransdEM.TBUtility
export MCParameter, MCPrior
export MCStatus, MChainStep
export MCDataFit
export MChainArray
export initMCPrior, initMCParameter
export initMCDataFit, initMChainArray
export initModelParameter

#-------------------------------------------------------------------------------
"""
    Struct `MCParameter` defines parameter structure for MCMC chains.

"""
mutable struct MCParameter{Ti<:Int,Tv<:Float64}

    #
    nLayer::Ti               # current layer number
    layeridx::Ti             # selected layer index to perturb
    zNode::Vector{Tv}        # interface depth
    rho::Vector{Tv}          # layer resistivity
    inBound::Bool            # indicator for inside or outside prior bounds

end # MCParameter


#-------------------------------------------------------------------------------
"""
    Struct `MCPrior` defines prior bounds of variables.

"""
mutable struct MCPrior{Ti<:Int,Tv<:Float64}

    #
    burninsamples::Ti
    totalsamples::Ti

    # min/max number of layers
    nlayermin::Ti
    nlayermax::Ti

    # lower/upper bounds for interface depth
    zmin::Tv
    zmax::Tv

    # minimum layer thickness
    hmin::Tv

    # lower/upper bounds for layer resistivity
    rhomin::Tv
    rhomax::Tv

    # standard derivation associated with variables
    zstd::Tv
    rhostd::Tv

    # parameters for post analysis
    nzBins::Ti
    npBins::Ti
    credInterval::Tv

end # RJPrior


#-------------------------------------------------------------------------------
"""
    Struct `MCStatus` stores proposal statistics of MCMC chain.

"""
mutable struct MCStatus{T<:Int}

    #
    accepted::Bool #是否被接收
    acceptstats::Vector{T}   # acceptance statistics
    rejectstats::Vector{T}   # rejection statistics

end # RJStatus


#-------------------------------------------------------------------------------
"""
    Struct `MChainStep` records the movement of MCMC chain at every step.

"""
mutable struct MChainStep{T<:Bool} #MC链状态

    isBirth::T #是否出生
    isDeath::T #是否死亡
    isMove::T  #是否移动
    isPerturb::T #是否扰动

end


#-------------------------------------------------------------------------------
"""
    Struct `MCDataFit` stores data from current and proposed models of MCMC
chain at every step.

"""
mutable struct MCDataFit{T<:Real}

    # current model
    currPredData::Vector{T} # 当前预测数据
    currLikelihood::T     #当前似然函数
    currMisfit::T         #当前拟合差

    # proposed model
    propPredData::Vector{T}  #提出预测数据
    propLikelihood::T        #提出似然度
    propMisfit::T            #提出拟合差

    # data fitting
    dataResiduals::Vector{T}  #数据残差

end


#-------------------------------------------------------------------------------
"""
    Struct `MChainArray` stores status of MCMC chain at every step.

"""
mutable struct MChainArray{Ti<:Int,Tv<:Real}

    # current sample number
    nsample::Ti #总采样次数

    # starting model
    startModel::MCParameter # 初始模型
    chainstep::Array{Ti,2} # 链的步数
    chainvalue::Array{Tv,2} # 链的值
    chainindice::Vector{Ti} # 链的目录
    chainnlayer::Vector{Ti} # 链的层数
    chainMisfit::Array{Tv} #链的拟合差
    residuals::Array{Tv}   #残差

end


#-------------------------------------------------------------------------------
"""
    `initMCParameter()` initializes parameter struct for MCMC chain.

"""
function initMCParameter()

    #
    itmp = zero(0)
    pvec = zeros(0)
    mcParam = MCParameter(itmp, itmp, pvec, pvec, false)
    return mcParam

end # initMCParameter


#-------------------------------------------------------------------------------
"""
    `initMCDataFit()` initializes datafit struct for MCMC chain.

"""
function initMCDataFit()

    #
    it = 0.0
    iv = zeros(0)
    mcDatafit = MCDataFit(iv, it, it, iv, it, it, iv) #MC数据拟合结构体

    return mcDatafit

end # initMCDataFit

#------------------------------------------------------------------------------
"""
    `initMCPrior()`

"""
function initMCPrior() #初始化MC先验信息

    #
    dtmp = 0.05 
    mclimits = MCPrior(100, 1000, 2, 10, 0.0, 5.0, 0.0, 1.0, 5.0, dtmp, dtmp,
        400, 400, 0.9)  

    return mclimits

end # initMCPrior


#-------------------------------------------------------------------------------
"""
    `initMChainArray(tblimits::MCPrior)` initializes struct `MChainArray`.

"""
function initMChainArray(mclimits::MCPrior)

    #
    currsample = 0 #当前采样
    startModel = initMCParameter() #初始模型
    #
    nsample = mclimits.totalsamples #总采样次数
    chainstep = zeros(Int, nsample, 2) #链的步骤？
    chainvalue = zeros(Float64, nsample, 2) #链的值
    chainindice = zeros(Int, nsample) #链目录
    chainncell = zeros(Int, nsample) #链单元
    chainMisfit = zeros(Float64, nsample) #链的拟合差
    residuals = zeros(Float64, 0)    #残差

    mcArray = MChainArray(currsample, startModel, chainstep, chainvalue,
        chainindice, chainncell, chainMisfit, residuals) # MC链结构体

    return mcArray

end # initMChainArray


#------------------------------------------------------------------------------
"""
    `initModelParameter(mclimits)`

initialize model parameter.

"""
function initModelParameter(mclimits::MCPrior)

    #
    mcParam = initMCParameter()

    # set number of layer
    # 返回最大值和最小整数值之间的一个随机值
    currLayerNumber = unirandInteger(mclimits.nlayermin, mclimits.nlayermax)
    mcParam.nLayer = currLayerNumber

    # set the locations of interface depth randomly 设置随机界面深度，和随机电阻率大小
    mcParam.zNode = zeros(Float64, mclimits.nlayermax)
    mcParam.rho = zeros(Float64, mclimits.nlayermax)
    for i = 1:currLayerNumber
        mcParam.zNode[i] = unirandDouble(mclimits.zmin, mclimits.zmax)

        # set the layer resistivity randomly
        mcParam.rho[i] = unirandDouble(mclimits.rhomin, mclimits.rhomax)

    end

    return mcParam

end


end # TBStruct
