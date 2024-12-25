# TransMTMPI: Supercharge Your EM Data Inversion with MPI-Powered MCMC! 🚀

Welcome to TransMTMPI - your powerful ally in the world of electromagnetic (EM) data inversion! Built with the lightning-fast Julia programming language, this package brings the power of parallel computing to your geophysical research through state-of-the-art MPI implementation. Whether you're wrestling with magnetotelluric (MT) data or diving deep into transient electromagnetic (TEM) analysis, TransMTMPI has got your back! 💪

## ✨ What Makes TransMTMPI Special?

- 🏃‍♂️ Blazing-fast MPI-based parallel processing
- 🎯 Smart RJMCMC and parallel tempering MCMC implementations
- 🎨 Seamless handling of 1D MT and TEM data inversion
- 📊 Rock-solid uncertainty quantification
- 🚄 High-performance computing that actually performs!

## 📜 License

TransMTMPI is free as a bird! It's distributed under the GNU General Public License. Check out the full details at the [license documentation](http://www.gnu.org/licenses/).

## 🗂️ What's in the Box?

- 📚 **./doc:** Your go-to guide for all things TransMTMPI
- 🎮 **./examples:** Real-world and synthetic examples to get you started
- 💻 **./src:** The beating heart of TransMTMPI - optimized MPI-powered source code

## 🛠️ Getting Started

### Step 1: Get Julia Up and Running

First things first - you'll need Julia v1.0 or newer. Here's how to get it:

To install julia in workstation or pc, you can refer to [`juliaup`](https://github.com/JuliaLang/juliaup)

### Step 2: MPI - The Secret Sauce 🌶️

You'll need MPI to unlock TransMTMPI's full potential:

#### 🪟 Windows Warriors
1. Download [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
2. Install and you're ready to roll!

#### 🐧 Linux Lovers
```bash
# Ubuntu/Debian fans
sudo apt update && sudo apt install openmpi-bin libopenmpi-dev

# CentOS/RHEL enthusiasts
sudo yum install openmpi openmpi-devel
```

## 🎮 Let's Get This Party Started!

### Setting Up Your Playground

1. Grab TransMTMPI:
```bash
git clone https://github.com/liaoweiyang2017/TransMTMPI.git
cd TransMTMPI
```

2. Load up the good stuff:
```julia
] add MPI BenchmarkTools Distributions Statistics LinearAlgebra Printf Random SparseArrays Serialization Test
```

### 🚀 Time to Launch!

Tell Julia where to find TransMTMPI:

```julia
# Linux crew
push!(LOAD_PATH, "/path/to/TransMTMPI")

# Windows gang
push!(LOAD_PATH, "D:\\path\\to\\TransMTMPI")
```

### 🏃‍♂️ Running at Full Speed

TransMTMPI offers two turbocharged modes:

1. **Standard RJMCMC with MPI Boost**:
```bash
# Linux style
mpirun -np 7 julia runMPIMCMCScript.jl > runMPIInfo.txt

# Windows flavor
mpiexec -np 7 julia runMPIMCMCScript.jl > runMPIInfo.txt
```

2. **Parallel Tempering MCMC with MPI Magic**:
```bash
# Linux style
mpirun -np 6 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt

# Windows flavor
mpiexec -np 6 julia runMPIPTMCMCScript.jl > runMPIPTInfo.txt
```

The `-np` flag is your power dial - set it based on your hardware's muscles! 💪

### 📝 Example Scripts to Get You Started

Check out the `./examples` folder for some ready-to-rock scripts:

- `runMPIMCMCScript.jl`: Your MPI-powered RJMCMC adventure
- `runMPIPTMCMCScript.jl`: Parallel tempering with an MPI twist

These scripts are loaded with comments to help you customize them for your specific needs!

## 🏎️ Performance Tips

- MPI implementation leaves Julia's Distributed package in the dust, especially for parallel tempering
- Match your MPI processes to your CPU cores for maximum zoom
- Keep an eye on your system's memory - don't bite off more than you can chew!

## 📚 Show Some Love

If TransMTMPI helps your research, we'd be thrilled if you cited us:

```bibtex
@article{peng2022julia,
  title={A Julia software package for transdimensional Bayesian inversion of electromagnetic data over horizontally stratified media},
  author={Peng, Ronghua and Han, Bo and Liu, Yajun and Hu, Xiangyun},
  journal={Geophysics},
  volume={87},
  number={5},
  pages={F55--F66},
  year={2022},
  publisher={Society of Exploration Geophysicists}
}
@article{liao20223,
  title={3-D joint inversion of MT and CSEM data for imaging a high-temperature geothermal system in Yanggao Region, Shanxi Province, China},
  author={Liao, Weiyang and Peng, Ronghua and Hu, Xiangyun and Zhou, Wenlong and Huang, Guoshu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--13},
  year={2022},
  publisher={IEEE}
}
@article{liao2024,
  title = {Fast forward modeling of magnetotelluric data in complex continuous media using an extended Fourier DeepONet architecture},
  author = { Weiyang Liao  and  Ronghua Peng  and  Xiangyun Hu  and  Yue Zhang  and  Wenlong Zhou  and  Xiaonian Fu  and  Haikun Lin },
  journal = {GEOPHYSICS},
  volume = {0},
  number = {ja},
  pages = {1-62},
  year = {2024},
  publisher={Society of Exploration Geophysicists}
  doi = {10.1190/geo2023-0613.1}   
}
```

## 🤝 Need Help?

Got questions? Hit a snag? No worries! Open an issue on our GitHub repository, and we'll help you get back on track. Remember, we're all in this together! 

Happy inverting! 🎉