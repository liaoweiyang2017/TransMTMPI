#------------------------------------------------------------------------------
# synthetic data for MT case
#
#------------------------------------------------------------------------------
push!(LOAD_PATH, "/home/acer/Downloads/TransdMT")
push!(LOAD_PATH, "/home/acer/Downloads/TransdMT/src")
using TransdEM.TBUtility
using TransdEM.TBStruct
using TransdEM.TBFileIO
using TransdEM.TBFwdSolver

#------------------------------------------------------------------------------
# 1D synthetic model revised from Guo et al., 2011.
#
sigma = [0.004, 0.04, 0.1, 0.001]
hLayer = [600, 1000, 2000, 4000]
depth1D = vcat(0.0, cumsum(hLayer))

# simulated data
period = 10 .^ LinRange(log10(1 / 320), log10(1000), 40)
freqs = 1 ./ period
zimp = compMT1DImpedance(freqs, sigma, depth1D)

# add 5% noise
nfreq = length(freqs)
predData = zeros(2 * nfreq)
predData[1:2:end] = real.(zimp)
predData[2:2:end] = imag.(zimp)
errlvl = 0.05
dataErr = abs.(predData) * errlvl
obsData = predData + dataErr .* randn(2 * nfreq)

# output
dataType = ["realZxy", "imagZxy"]
dataID = ones(Bool, 2, nfreq)
dataID = vec(dataID)
mtData = MTData(freqs, dataType, dataID, obsData, dataErr)
datafile = "mt4_3data.dat"
writeMTData(datafile, mtData, obsData, dataErr)
