
##################################################
#    FRAGMENTATION CALCULATION WRAP FUNCTIONS    #
##################################################

"""
Inner fragmentation calculation on a graph (KPP-Negative)


# Constructs

```
calculate_fragmentation(
    graph::Union{SimpleGraph, SimpleDiGraph, Nothing};
    alg_func::Function = dijkstra_shortest_paths,
)
```

```
calculate_fragmentation_parallel(
    graph::Union{SimpleGraph, SimpleDiGraph, Nothing};
    alg_func::Function = dijkstra_shortest_paths,
)
```


## Function Arguments

- `graph`: SimpleDiGraph or SimpleGraph on which to calculate fragmentation


## Keyword Arguments

- `alg_func`: shortest-paths distance function to use
- `kwargs...`: passed to alg_func
"""
function calculate_fragmentation(
    graph::Union{SimpleGraph, SimpleDiGraph, Nothing},
    dict_arrays::Union{Dict{Symbol, Vector}, Nothing} = nothing;
    alg_func::Symbol = :dijkstra_kary,
    kwargs...
)

    n = length(vertices(graph))
    denom = n*(n - 1)
    v_dist = ones(Int64, n)

    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            alg_func;
            type = :Vector,
        )
    end
    
    
    ##  RUN USING

    if alg_func == :dijkstra_kary
        fract = cfs_dijkstra(
            graph,
            dict_arrays,
        )
    
    elseif alg_func == :bellman_ford
        fract = cfs_bellman_ford(
            graph,
            dict_arrays,
        )
    end

    return fract
end



"""
Inner fragmentation calculation on a graph (KPP-Negative)


# Constructs

```
calculate_fragmentation(
    graph::Union{SimpleGraph, SimpleDiGraph, Nothing};
    alg_func::Function = dijkstra_shortest_paths,
)
```

```
calculate_fragmentation_parallel(
    graph::Union{SimpleGraph, SimpleDiGraph, Nothing};
    alg_func::Function = dijkstra_shortest_paths,
    dict_arrays::Union{Dict{Symbol, DArray}, Nothing} = nothing,
)
```


## Function Arguments

- `graph`: SimpleDiGraph or SimpleGraph on which to calculate fragmentation
- `dict_arrays`: dictionary of arrays to use for data parallelization; maps 
    required elements for iterative algorithms to be passed to DistributedArrays

## Keyword Arguments

- `alg_func`: shortest-paths distance algorithm to use:
    * :bellman_fird
    * :dijkstra_kary
- `kwargs...`: passed to alg_func
"""
function calculate_fragmentation_parallel(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_arrays::Union{Dict{Symbol, DArray}, Nothing} = nothing;
    alg_func::Symbol = :dijkstra_kary,
)
 
    # open arrays?
    close_darrays = false
    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            alg_func;
            type = :DistributedArray,
        )

        close_darrays = true
    end
    
    
    ##  RUN USING

    if alg_func == :dijkstra_kary
        fract = cfp_dijkstra(
            graph,
            dict_arrays,
        )
    
    elseif alg_func == :bellman_ford
        fract = cfp_bellman_ford(
            graph,
            dict_arrays,
        )
    end

    # close the arrays?
    if close_darrays
        for (k, v) in dict_arrays
            close(v)
        end
    end

    return fract
end


#
# REVISE THIS
#
function calculate_fragmentation_parallel(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_sarrays::Dict{Symbol, SharedArray};
    alg_func::Function = GraphDistanceAlgorithms.dijkstra_kary,
)

    n = nv(graph)
    denom = Float64(n*(n - 1))
    # 
    ad = get(dict_sarrays, :dists, nothing)
    ap = get(dict_sarrays, :parents, nothing)
    ahd = get(dict_sarrays, :heap_data, nothing)
    ahi = get(dict_sarrays, :heap_index, nothing)
    ahl = get(dict_sarrays, :heap_index_lookup, nothing)
    
    # use parallel reduction (see https://docs.julialang.org/en/v1/manual/distributed-computing/#Remote-References-and-AbstractChannels)
    begin
        val = @distributed (+) for i in 1:n

            #vd = 
            v_dist = alg_func(
                graph, 
                i; 
                dists = ad[:, ad.pidx],
                parents = ap[:, ap.pidx],
                heap_data = ahd[:, ahd.pidx],
                heap_index = ahi[:, ahi.pidx],
                heap_index_lookup = ahl[:, ahl.pidx],
            )
            #v_dist = ad[SharedArrays.localindices(ad)]
            sum(1.0 ./ v_dist[v_dist .> 0])
        end
    end
    
    fract = 1.0 - val/denom

    return fract
end



"""
Calculate fragmentation in parallel using Bellman-Ford
"""
function cfp_bellman_ford(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_arrays::Union{Dict{Symbol, DArray}, Nothing} = nothing;
)
    n = length(vertices(graph))
    denom = Float64(n*(n - 1))

    close_darrays = false
    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            :bellman_ford;
            type = :DistributedArray,
        )

        close_darrays = true
    end
    
    # retrieve some arrays
    darray_active = get(dict_arrays, :active, nothing)
    darray_dists = get(dict_arrays, :dists, nothing)
    darray_new_active = get(dict_arrays, :new_active, nothing)
    darray_parents = get(dict_arrays, :parents, nothing)

    # use parallel reduction (see https://docs.julialang.org/en/v1/manual/distributed-computing/#Remote-References-and-AbstractChannels)
    val = @distributed (+) for i in 1:n
        bellman_ford!(
            darray_dists[:L],
            graph, 
            i; 
            active = darray_active[:L],
            parents = darray_parents[:L],
            new_active = darray_new_active[:L],
        )

        sum(1.0 ./ darray_dists[:L][darray_dists[:L] .> 0])
    end
    
    # close the arrays?
    if close_darrays
        for (k, v) in dict_arrays
            close(v)
        end
    end

    fract = 1.0 - val/denom

    return fract
end



"""
Calculate fragmentation in parallel using Dijkstra
"""
function cfp_dijkstra(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_arrays::Union{Dict{Symbol, DArray}, Nothing} = nothing;
)
    n = length(vertices(graph))
    denom = Float64(n*(n - 1))

    close_darrays = false
    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            :dijkstra_kary;
            type = :DistributedArray,
        )

        close_darrays = true
    end
    
    # retrieve some arrays
    darray_dists = get(dict_arrays, :dists, nothing)
    darray_parents = get(dict_arrays, :parents, nothing)
    darray_heap_data = get(dict_arrays, :heap_data, nothing)
    darray_heap_index = get(dict_arrays, :heap_index, nothing)
    darray_heap_index_lookup = get(dict_arrays, :heap_index_lookup, nothing)

    # use parallel reduction (see https://docs.julialang.org/en/v1/manual/distributed-computing/#Remote-References-and-AbstractChannels)
    val = @distributed (+) for i in 1:n
        dijkstra_kary!(
            darray_dists[:L],
            graph, 
            i; 
            parents = darray_parents[:L],
            heap_data = darray_heap_data[:L],
            heap_index = darray_heap_index[:L],
            heap_index_lookup = darray_heap_index_lookup[:L],

        )#.dists

        sum(1.0 ./ darray_dists[:L][darray_dists[:L] .> 0])
    end
    
    # close the arrays?
    if close_darrays
        for (k, v) in dict_arrays
            close(v)
        end
    end

    fract = 1.0 - val/denom

    return fract
end



"""
Calculate fragmentation in serial using Bellman-Ford
"""
function cfs_bellman_ford(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_arrays::Union{Dict{Symbol, Vector}, Nothing} = nothing;
)
    n = length(vertices(graph))
    denom = Float64(n*(n - 1))

    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            :bellman_ford;
            type = :Vector,
        )
    end
    
    # retrieve some arrays
    vec_active = get(dict_arrays, :active, nothing)
    vec_dists = get(dict_arrays, :dists, nothing)
    vec_new_active = get(dict_arrays, :new_active, nothing)
    vec_parents = get(dict_arrays, :parents, nothing)

    # use parallel reduction (see https://docs.julialang.org/en/v1/manual/distributed-computing/#Remote-References-and-AbstractChannels)
    val = 0.0
    for i in 1:n
        bellman_ford!(
            vec_dists,
            graph, 
            i; 
            active = vec_active,
            new_active = vec_new_active,
            parents = vec_parents,
        )

        val += sum(1.0 ./ vec_dists[vec_dists .> 0])
    end
    
    fract = 1.0 - val/denom

    return fract
end



"""
Calculate fragmentation in serial using Dijkstra
"""
function cfs_dijkstra(
    graph::Union{SimpleGraph, SimpleDiGraph},
    dict_arrays::Union{Dict{Symbol, Vector}, Nothing} = nothing;
)
    n = length(vertices(graph))
    denom = Float64(n*(n - 1))

    if isa(dict_arrays, Nothing)
        dict_arrays = spawn_arrays(
            graph,
            :dijkstra_kary;
            type = :Vector,
        )
    end
    
    # retrieve some arrays
    vec_dists = get(dict_arrays, :dists, nothing)
    vec_parents = get(dict_arrays, :parents, nothing)
    vec_heap_data = get(dict_arrays, :heap_data, nothing)
    vec_heap_index = get(dict_arrays, :heap_index, nothing)
    vec_heap_index_lookup = get(dict_arrays, :heap_index_lookup, nothing)

    # use parallel reduction (see https://docs.julialang.org/en/v1/manual/distributed-computing/#Remote-References-and-AbstractChannels)
    val = 0.0

    for i in 1:n
        dijkstra_kary!(
            vec_dists,
            graph, 
            i; 
            parents = vec_parents,
            heap_data = vec_heap_data,
            heap_index = vec_heap_index,
            heap_index_lookup = vec_heap_index_lookup,
        )#.dists

        val += sum(1.0 ./ vec_dists[vec_dists .> 0])
    end

    fract = 1.0 - val/denom

    return fract
end



"""
Calculate the fragmentation of a graph (KPP-Negative)


# Constructs

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
    adj::Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}};;
    kwargs...
)
```


## Function Arguments

- `graph`: graph on which to calculate fragmentation
- `dict_arrays`: optional dictionary mapping relevant algorithm keys to arrays--
    DArrays, SharedArrays (not recommended unless size is very large), or 
    Vectors--storing intermediate calculations.
    
    * Only effective if fixing `distance_algorithm` to align with the arrays
        that are passed
    * See ?GraphDistanceAlgorithms.spawn_arrays for more information on the 
        inputs required.


## Keyword Arguments

- `D_invs`: optional matrix (with 0 diagonal) of inverse distances. Passing 
    this optional argument can speed up calculations considerably.
    
    **CAUTION** This function assumes that `D_invs` is complete with 0s along 
        the diagonal.
- `distance_algorithm`: distance489503algorithm to use in computing distances. Called
    if `D_invs` is not specified
- `parallel_approach`: `fragmentation` will automatically try to implement 
    parallel calculation if `try_parallel(graph) == true`. `parallel_approach`
    can take one of three values:
    * `:auto`: choose based on `try_parallel(graph)`
    * `:parallel`: Force a parallel implementation (slower on small graphs)
    * `:serial`: Force a serial implementation (slower on large graphs)
- `use_distance_weighting`: use distance-weigthed fragmentation? If False, 
    defaults to adjacency only. 
- `kwargs...`: passed to distance algorithm. Include options for heap vectors 
    etc.
"""
function fragmentation(
    graph::Union{AbstractGraph, Nothing}, 
    dict_arrays::Union{Dict{Symbol, Vector}, Dict{Symbol, DArray}, Nothing} = nothing;
    D_invs::Union{Matrix{Int64}, Matrix{Float64}, Nothing} = nothing,
    distance_algorithm::Symbol = :auto,
    parallel_approach::Symbol = :auto,
    use_distance_weighting::Bool = true,
    kwargs...
)::Union{Float64, Nothing}
    
    # components = connected_components(graph)
    
    # some initialization
    fract = nothing
    n, m = size(graph)
    denom = n*(n - 1)

    if use_distance_weighting & isa(D_invs, Nothing)

        alg_func = select_algorithm(
            distance_algorithm; 
            n_vertices = n
        )
        
        func = (
            try_parallel(graph; parallel_approach = parallel_approach, )
            ? calculate_fragmentation_parallel
            : calculate_fragmentation
        )

        fract = func(
            graph, 
            dict_arrays; 
            alg_func = alg_func, 
            kwargs...
        )
        
    else
        # case where matrix indices can be used
        mat = use_distance_weighting ? D_invs : adjacency_matrix(graph) 
        fract = 1 - sum(mat)/denom
    end

    return fract 
end

function fragmentation(
    adj::Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}};
    kwargs...
)::Union{Float64, Nothing}
    graph = LinearAlgebra.issymmetric(adj) ? SimpleGraph(adj) : SimpleDiGraph(adj)
    fract = fragmentation(graph; kwargs...)
    return fract 
end



"""
Get a default set of nodes to start from based on a centrality measure.

##  Constructs

```
get_default_kpp_nodes(
    graph::Union{SimpleDiGraph, SimpleGraph},
    n_nodes::Int64;
    centrality::Symbol = :betweenness,
)
```

##  Function Arguments

- `graph`: graph object to tuse
- `n_nodes`: number of nodes to sample


##  Keyword Arguments

- `centrality`: measure of centrality to use to start with. Acceptable values are
    * :betweenness
    * :eigenvector
"""
function get_default_kpp_nodes(
    graph::Union{SimpleDiGraph, SimpleGraph},
    n_nodes::Int64;
    centrality::Symbol = :betweenness,
)
    # get the centrality measure
    centrality = !(centrality in [:betweenness, :eigenvector]) ? :betweenness : centrality
    
    cent_func = (
        centrality == :betweenness
        ? betweenness_centrality
        : eigenvector_centrality # ADD MORE HERE
    )
    
    n_v, n_e = size(graph)
    
    vec_measure = cent_func(graph);
    vec_measure = sort(
        collect(zip(vec_measure, 1:n_v)); 
        by = x -> x[1], 
        rev = true
    )
    
    vec_measure = [x[2] for x in vec_measure[1:n_nodes]]
    
    return vec_measure
end



"""
Try parallel implementation of fragmentation?

##  Constructs

```
try_parallel(
    graph::Union{SimpleDiGraph, SimpleGraph};
    override::Bool = false,
    vertex_threshold::Int64 = 100,
)
```

##  Function Arguments

- `graph`: graph, used to determine node size


##  Keyword Arguments

- `parallel_approach`: `fragmentation` will automatically try to implement 
    parallel calculation if `try_parallel(graph) == true`. `parallel_approach`
    can take one of three values:
    * `:auto`: choose based on `try_parallel(graph)`
    * `:parallel`: Force a parallel implementation (slower on small graphs)
    * `:serial`: Force a serial implementation (slower on large graphs)
- `vertex_threshold`: number of vertices to use as threshold between serial and
    parallel. There is a time cost to passing information between worker nodes
    and the manager, meaning that, on small graphs, parallelization can be 
    slower.
"""
function try_parallel(
    graph::Union{SimpleDiGraph, SimpleGraph};
    parallel_approach::Symbol = :auto,
    vertex_threshold::Int64 = 100,
)::Bool
    # no need to do anything else if we don't have access to processes 
    (nprocs() == 1) && (return false);

    valid_approaches = [:auto, :parallel, :serial]
    parallel_approach = !(parallel_approach in valid_approaches) ? :auto : parallel_approach

    # must be positive
    out = length(vertices(graph)) > maximum([vertex_threshold, 1]) 
    out |= (parallel_approach == :parallel)
    out &= (parallel_approach != :serial)

    return out
end




#
#   MULTITHREADING--INCOMPLETE   #
#

"""
Get the partition used split up the calculation across threads
"""
function get_thread_partition(
    graph::AbstractGraph,
)
    n = nv(graph)
    chunks = Base.Iterators.partition(
        collect(1:n), 
        Int64(ceil(n/Threads.nthreads()))
    )
    
    return chunks
end
    



function calculate_fragmentation_multithread(
    graph::Union{SimpleGraph, SimpleDiGraph};
    alg_func::Function = dijkstra_kary!,
    chunks::Union{Base.Iterators.PartitionIterator, Nothing} = nothing,
    dists::Vector{Vector{Int64}} = nothing,
    parents::Vector{Vector{Int64}} = nothing,
    heap_data::Vector{Vector{Int64}} = nothing,
    heap_index::Vector{Vector{Int64}} = nothing,
    heap_index_lookup::Vector{Vector{Int64}} = nothing,
)::Float64
    
    n = nv(graph)
    denom = Float64(n*(n - 1))
    
    # verify the partition
    isa(chunks, Nothing) && (chunks = get_thread_partition(graph, ));
    
    # map over each chunk
    tasks = map(enumerate(chunks)) do chunk_tup
        
        j, chunk = chunk_tup
        
        Threads.@spawn begin
        
            v_cur = 0.0
            # :static IS TEMPORARY non-ideal solution
            # BETTER: write using partition, chunk, etc
            # https://julialang.org/blog/2023/07/PSA-dont-use-threadid/#quickfix_replace_threads_with_threads_static
            #Threads.@threads :static for i in 1:n

            for i in chunk
                alg_func(
                    dists[j],
                    graph, 
                    i; 
                    parents = parents[j],
                    heap_data = heap_data[j],
                    heap_index = heap_index[j],
                    heap_index_lookup = heap_index_lookup[j],
                )

                v_cur += sum(1.0 ./ dists[j][dists[j] .> 0])

            end
            
            return v_cur
            
        end
    end
    
    states = fetch.(tasks)
    
    val = sum(states)
    fract = 1.0 - val/denom

    return fract
end



