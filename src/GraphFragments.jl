"""
Calculate fragmentation (Borgatti) on a graph
"""
module GraphFragments


using Distributed
using DistributedArrays
using Graphs
using LinearAlgebra
using Random
using SharedArrays
using SparseArrays
using StatsBase


# custom modules TEMPORARY
#using Pkg
#Pkg.develop(path = "/Users/jsyme/Documents/Projects/git_jbus/GraphDistanceAlgorithms.jl")
using GraphDistanceAlgorithms



##  EXPORTS

export calculate_fragmentation,
       calculate_fragmentation_parallel,
       Examples,
       fragmentation,
       get_default_kpp_nodes,
       select_algorithm_from_benchmark_for_iteration,
       try_parallel



include("Fragmentation.jl")
include("Utilities.jl")
include("Examples.jl")

end # module GraphFragments
