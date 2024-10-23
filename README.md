# GraphFragments.jl

## Introduction

`GraphFragments.jl` implements the fragmentation metric introduced by [Borgatti](https://download.ssrn.com/08/06/22/ssrn_id1149843_code1053981.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEMaCXVzLWVhc3QtMSJHMEUCICXndhOA5JS0Bx%2FLPRMtVsQz3tTFAi3AgP1qOmpkHQgMAiEAkLxQJRX3mEk3PeiKpwWpiMlfLKTneMIB8V7PO9%2FsRFMqxgUIrP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwzMDg0NzUzMDEyNTciDLqPEy7ZxsrqhdKcISqaBQV93KQ4Wt5pY5lMvUpmsdF3lU1Yelqce2uyax7nMBdKWrUd%2B0mg%2F9M6d%2FhZKIpWwjEQIgBP2hO4pBN1aJq2GFCPFJBdjptDEoja7R78tbp2%2BbvPaEc2Dhb51J81kdWwZPEbIpkVdrGaenaAWvMYISnS6qOeX0poNcH6ax72cLvPNxpv%2FwRknJBoMne51scDSauEzTdEk0enXkBKiV4DKhyRyx0fqItc8gAolFFMNhVbOtIZog3POOWiEJaDwS0lEpDvKcpP3ZwJ54Nb4%2BU0xwUsxUmypM5%2Fq1nD%2BM5Vn4LmMT12%2FsRCGDn866UDuXnzHLbvtk0VQky2OVC%2FzBPlZGzLVpOP%2BOiuckDJV66EPF0BNE3JdExHypNklDioWv3AU1sebtTe%2BfelUyO777aDWkaGFbb2O4aPBiv5tQZasCOjo5iAfPNUhnQ%2B6EsNzAr4loi73ik0MQ3ztcO6phaLBWzmvnYh9zaHotHkHwKNKa3pp59NTUfvgJZ6y7rqiI0v1np1oZWzl3cBh0U13oWN6FXlljdh42N76jpNx0j29cYYcy5Krk%2Bq1AzGHH748mA%2BTJnKlYmaSpdR3D3N%2Btg3fE4ZCtkr8tm7YWr%2FiHMyu1wI3rf%2FuUZbSyY%2BzPw%2B1YeB6BQE6qX6l1xJclrSFuRtV55o%2B8YKDAZD9nAIHbsQHUoEdPgo9Q4%2B3Ne3X8J0TirZbk9%2BzSiPnVaHLWGJQ8%2BGw3biVwRx3hSushGgOwtRXXHcoKUPt2ZLrcfiA1C1bSuPPBaZp9T6Mqz%2FdK25vViDlXYK9G%2FvyDsCpmqJejhNbGr8M40Ycn6zucNQ58VQbCSCN6F7GuKoNcCsu4U71%2BM2EYQQe60KzL2dkwYnSghTp8yo%2BZ0nri8NFtjZmDDJ6N%2B4BjqxAS3pgufZt4WFit1EvOerhCjUXKHLp4B24aKaPasoY4dkl6S%2FXV%2BP3cfoKLcUq%2FPrbtGJf68arsHYzX6M19D5a%2FU03OOnXebBdGyqp6tugmOD41VnE1VXh%2BdcX0WU%2BYnB%2F5%2FWa0KYArC3CKPMVDqWMNKPrm0FbxY1lvbtH60UPuGKI5mhUIWjNIzj8gwF4q5W9Q4G%2By1BD77G4eIwnT6wAzgakzkkEhZtQlbq1vMs9bFCeg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241022T194922Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWESEE55P7E%2F20241022%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=210b3dab679c9ca54a84e708b26f6f43dd31086eeacf9376079e536cde2a2d8e&abstractId=1149843). The package includes standard implementation, distance-based fragmentation, and serial and distributed approaches.



## Use

The fragmentation of the graph can be calculated using the `fragmentation` function. This function will automatically try to run in parallel if the keyword argument `parallel_approach = true` (default) is set; this can be fixed to run serially if desired or in parallel (if a script is setup to automatically run in parallel). 

Fragmentation relies on the calculation of shortest paths across the graph from each starting node; if parallelized, the calculation will use synchronous data parallelization to distribute each source node on an available compute node. A dictionary of `DistributedArrays.DArray` objects for use by the distance algorithm can be passed using the `dict_arrays` argument, though the user should ensure these arrays match those necessary for the algorithm specificed by `distance_algorithm`. These arrays can include intermediate algorithmic vectors as well as any applicable heap cache arrays.

The user can access information about these arguments using the docstrings for `fragmentation` in Julia. 

```
fragmentation(
    graph::Union{AbstractGraph, Nothing}, 
    dict_arrays::Union{Dict{Symbol, Vector}, Dict{Symbol, DArray}, Nothing} = nothing;
    D_invs::Union{Matrix{Int64}, Matrix{Float64}, Nothing} = nothing,
    distance_algorithm::Symbol = :auto,
    parallel_approach::Symbol = :auto,
    use_distance_weighting::Bool = true,
    kwargs...
)
```

```
fragmentation(
    adj::Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}};
    kwargs...
)
```



## Data

The code only needs a Graph to work off of. This can be loaded in a Julia session using `Graphs.jl`. If using `DiscreteGraphAlgorithms.jl`, then the `graph` property of a `GraphWrapper` can be used.



## Project information

The authors are grateful to RAND Center for Global Risk and Security Advisory Board members Michael Munemann and Paul Cronson for funding this project. All code was developed between April 2023 and October 2024.



## References/Bibliography

Borgatti, Steve, The Key Player Problem (September 21, 2002). Available at SSRN: https://ssrn.com/abstract=1149843 or http://dx.doi.org/10.2139/ssrn.1149843

 

## Copyright and License

Copyright (C) <2024> RAND Corporation. This code is made available under the MIT license.

 

## Authors and Reference

James Syme, 2024.

@misc{GDA2024,
  author       = {Syme, James},
  title        = {GraphFragments.jl: Distributed implementation of Key-Player Problem negative (KPP-Negative) metric.},
  year         = 2024,
  url = {URLHERE}
}
