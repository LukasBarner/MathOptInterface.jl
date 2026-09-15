# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct _SubexpressionStorage{T}
    nodes::Vector{Nonlinear.Node}
    adj::SparseArrays.SparseMatrixCSC{Bool,Int}
    const_values::Vector{T}
    forward_storage::Vector{T}
    partials_storage::Vector{T}
    reverse_storage::Vector{T}
    partials_storage_ϵ::Vector{T}
    linearity::Linearity

    function _SubexpressionStorage(
        expr::Nonlinear.Expression{T},
        subexpression_linearity,
        moi_index_to_consecutive_index,
        want_hess::Bool,
    ) where {T}
        nodes =
            _replace_moi_variables(expr.nodes, moi_index_to_consecutive_index)
        adj = Nonlinear.adjacency_matrix(nodes)
        N = length(nodes)
        linearity = if want_hess
            _classify_linearity(nodes, adj, subexpression_linearity)[1]
        else
            NONLINEAR
        end
        return new{T}(
            nodes,
            adj,
            expr.values,
            zeros(T, N),  # forward_storage,
            zeros(T, N),  # partials_storage,
            zeros(T, N),  # reverse_storage,
            T[],
            linearity,
        )
    end
end

struct _FunctionStorage{T}
    nodes::Vector{Nonlinear.Node}
    adj::SparseArrays.SparseMatrixCSC{Bool,Int}
    const_values::Vector{T}
    forward_storage::Vector{T}
    partials_storage::Vector{T}
    reverse_storage::Vector{T}
    grad_sparsity::Vector{Int}
    # Nonzero pattern of Hessian matrix
    hess_I::Vector{Int}
    hess_J::Vector{Int}
    rinfo::Coloring.RecoveryInfo # coloring info for hessians
    seed_matrix::Matrix{T}
    linearity::Linearity
    # subexpressions which this function depends on, ordered for forward pass.
    dependent_subexpressions::Vector{Int}

    function _FunctionStorage(
        nodes::Vector{Nonlinear.Node},
        const_values::Vector{T},
        num_variables,
        coloring_storage::Coloring.IndexedSet,
        want_hess::Bool,
        subexpressions::Vector{_SubexpressionStorage{T}},
        dependent_subexpressions,
        subexpression_linearity,
        subexpression_edgelist,
        subexpression_variables,
        moi_index_to_consecutive_index,
    ) where {T}
        nodes = _replace_moi_variables(nodes, moi_index_to_consecutive_index)
        adj = Nonlinear.adjacency_matrix(nodes)
        N = length(nodes)
        empty!(coloring_storage)
        _compute_gradient_sparsity!(coloring_storage, nodes)
        for k in dependent_subexpressions
            _compute_gradient_sparsity!(
                coloring_storage,
                subexpressions[k].nodes,
            )
        end
        grad_sparsity = sort!(collect(coloring_storage))
        empty!(coloring_storage)
        if want_hess
            linearity = _classify_linearity(nodes, adj, subexpression_linearity)
            edgelist = _compute_hessian_sparsity(
                nodes,
                adj,
                linearity,
                subexpression_edgelist,
                subexpression_variables,
            )
            hess_I, hess_J, rinfo = Coloring.hessian_color_preprocess(
                edgelist,
                num_variables,
                coloring_storage,
            )
            seed_matrix = T.(Coloring.seed_matrix(rinfo))
            return new{T}(
                nodes,
                adj,
                const_values,
                zeros(T, N),  # forward_storage,
                zeros(T, N),  # partials_storage,
                zeros(T, N),  # reverse_storage,
                grad_sparsity,
                hess_I,
                hess_J,
                rinfo,
                seed_matrix,
                linearity[1],
                dependent_subexpressions,
            )
        else
            return new{T}(
                nodes,
                adj,
                const_values,
                zeros(T, N),  # forward_storage,
                zeros(T, N),  # partials_storage,
                zeros(T, N),  # reverse_storage,
                grad_sparsity,
                Int[],
                Int[],
                Coloring.RecoveryInfo(),
                Matrix{T}(undef, 0, 0),
                NONLINEAR,
                dependent_subexpressions,
            )
        end
    end
end

"""
    NLPEvaluator(
        model::Nonlinear.Model,
        ordered_variables::Vector{MOI.VariableIndex},
    )

Return an `NLPEvaluator` object that implements the `MOI.AbstractNLPEvaluator`
interface.

!!! warning
    Before using, you must initialize the evaluator using `MOI.initialize`.
"""
mutable struct NLPEvaluator{T} <: MOI.AbstractNLPEvaluator
    data::Nonlinear.Model
    ordered_variables::Vector{MOI.VariableIndex}

    objective::Union{Nothing,_FunctionStorage{T}}
    constraints::Vector{_FunctionStorage{T}}
    subexpressions::Vector{_SubexpressionStorage{T}}
    subexpression_order::Vector{Int}
    # Storage for the subexpressions in reverse-mode automatic differentiation.
    subexpression_forward_values::Vector{T}
    subexpression_reverse_values::Vector{T}
    subexpression_linearity::Vector{Linearity}

    # A cache of the last x. This is used to guide whether we need to re-run
    # reverse-mode automatic differentiation.
    last_x::Vector{T}

    # Temporary storage for computing Jacobians. This is also used as temporary
    # storage for the input of multivariate functions.
    jac_storage::Vector{T}
    # Temporary storage for the gradient of multivariate functions
    user_output_buffer::Vector{T}

    # storage for computing hessians
    # these T vectors are reinterpreted to hold multiple epsilon components
    # so the length should be multiplied by the maximum number of epsilon components
    disable_2ndorder::Bool # don't offer Hess or HessVec
    want_hess::Bool
    partials_storage_ϵ::Vector{T} # (longest expression excluding subexpressions)
    storage_ϵ::Vector{T} # (longest expression including subexpressions)
    input_ϵ::Vector{T} # (number of variables)
    output_ϵ::Vector{T} # (number of variables)
    subexpression_forward_values_ϵ::Vector{T} # (number of subexpressions)
    subexpression_reverse_values_ϵ::Vector{T} # (number of subexpressions)
    hessian_sparsity::Vector{Tuple{Int64,Int64}}
    max_chunk::Int # chunk size for which we've allocated storage

    function NLPEvaluator(
        data::Nonlinear.Model{T},
        ordered_variables::Vector{MOI.VariableIndex},
    ) where {T}
        return new{T}(data, ordered_variables)
    end
end
