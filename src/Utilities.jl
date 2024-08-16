"""
Select a GraphDistanceAlgorithm to use for iteration based on how it performs on a quick benchmark of fragmentation.

    NOTE: If not using for iteration, dijkstra_quickheaps can outperform 
    dijkstra_kary.


# Constructs

```
select_algorithm_from_benchmark_for_iteration(
    graph::AbstractGraph;
    algorithms_to_try::Union{Vector{Symbol}, Nothing} = nothing,
    n_runs::Int64 = 10,
    selection_metric::Symbol = :mean,
)
```


##  Function Arguments

- `graph`: graph to select on
- `algorithms_to_try`: one of the following
    * `:bellman_ford`
    * `:dijkstra_kary`
    * nothing (evaluate all available)
- `n_runs`: number of tries to use for benchmarking


##  Keyword Arguments

- `selection_metric`: One of the following, used to select the algorithm
    * `:mean`: use the mean of runtimes
    * `:median`: use the median of runtimes
    * `:min_max`: use the runtime with the lowest maximum
"""
function select_algorithm_from_benchmark_for_iteration(
    graph::AbstractGraph;
    algorithms_to_try::Union{Vector{Symbol}, Nothing} = nothing,
    n_runs::Int64 = 10,
    selection_metric::Symbol = :mean,
)
    ##  INITIALIZATION AND CHECKS

    # check algorithms
    algorithms_all = [
        :bellman_ford,
        :dijkstra_kary
    ]

    algorithms_to_try = (
        isa(algorithms_to_try, Nothing) 
        ? algorithms_all 
        : [x for x in algorithms_to_try if x in algorithms_all]
    )

    if length(algorithms_to_try) == 0
        
        valid = print_valid_values(algorithms_all)
        msg = "No valid algorithms defined in select_algorithm_from_benchmark_for_iteration. Valid algorithms are $(valid)."
        @error(msg)

        return nothing
    end
    
    # check selection metric
    selection_metric = (
        !(selection_metric in [:mean, :median, :min_max])
        ? :mean
        : selection_metric
    )
    
    
    # some init, including number of runs, parallelization q, and what types of arrays to use
    n_runs = max(n_runs, 1)
    par_q = try_parallel(graph)
    arr_types = par_q ? :DistributedArray : :Vector

    # intialize output metrics
    dict_metrics = Dict()

        
    ##  ITERATE OVER ALGORITHMS TO TRY
    
    # init
    all_time = ones(Float64, n_runs)
        
    for alg in algorithms_to_try

        dict_arrays = spawn_arrays(
            graph,
            alg;
            type = arr_types
        )
        
        for i in 1:n_runs
            time = @elapsed fragmentation(
                graph,
                dict_arrays;
                distance_algorithm = alg,
                parallel_approach = (par_q ? :parallel : :serial),
            )
                
            all_time[i] = time
        end
        
        # 
        if selection_metric == :mean
            metric = sum(all_time)/n_runs
        elseif selection_metric == :median
            metric = median(all_time)
        elseif selection_metric == :min_max
            metric = maximum(all_time)
        end
            
        # add to dictionary
        dict_metrics[alg] = metric
       
        # close darrays if necessary
        if arr_types == :DistributedArray
            for (k, v) in dict_arrays
                close(v)
            end
        end
    end
    
    # clean up
    GC.gc()
    
    # iterate to choose
    best = Inf
    best_key = :none
    for (k, v) in dict_metrics
        (v < best) && (best = v; best_key = k;)
    end

    return best_key
    
end
