"""
Build some validation examples for communicating fragmentation and the
    Key-Player Problem. See 
    [Borgatti 2006](https://doi.org/10.1007/s10588-006-7084-x)
    
    Borgatti, S.P. Identifying sets of key players in a social network. 
    Comput Math Organiz Theor 12, 21-34 (2006). 
    https://doi.org/10.1007/s10588-006-7084-x

Use `Examples.get(:EXAMPLE_NAME)` to get examples. The following examples
    are defined:

    * :borgatti_figure_5a
    * :borgatti_figure_5b
 
"""
struct Examples
    get_example::Function 
    
    function Examples()
        
        dict_matrices = Dict{Symbol, Matrix{Float64}}()
    
        # 5a - paired complete graphs
        dict_matrices[:borgatti_figure_5a] = [
            [0 1 1 1 1 0 0 0 0 0];
            [1 0 1 1 1 0 0 0 0 0];
            [1 1 0 1 1 0 0 0 0 0];
            [1 1 1 0 1 0 0 0 0 0];
            [1 1 1 1 0 0 0 0 0 0];
            [0 0 0 0 0 0 1 1 1 1];
            [0 0 0 0 0 1 0 1 1 1];
            [0 0 0 0 0 1 1 0 1 1];
            [0 0 0 0 0 1 1 1 0 1];
            [0 0 0 0 0 1 1 1 1 0];
        ]
        
        # 5b - paired graphs with less cohesion (linear graphs?)
        dict_matrices[:borgatti_figure_5b] = [
            [0 1 0 0 0 0 0 0 0 0];
            [1 0 1 0 0 0 0 0 0 0];
            [0 1 0 1 0 0 0 0 0 0];
            [0 0 1 0 1 0 0 0 0 0];
            [0 0 0 1 0 0 0 0 0 0];
            [0 0 0 0 0 0 1 0 0 0];
            [0 0 0 0 0 1 0 1 0 0];
            [0 0 0 0 0 0 1 0 1 0];
            [0 0 0 0 0 0 0 1 0 1];
            [0 0 0 0 0 0 0 0 1 0];
        ]

        function get_example(
            matrix::Symbol,
        )::Union{Matrix{Float64}, Nothing}

            matrix_out = get(dict_matrices, matrix, nothing)
            matrix_out = !isa(matrix_out, Nothing) ? Float64.(matrix_out) : matrix_out

            return matrix_out
        end

        return new(
            get_example
        )
    end
    
end