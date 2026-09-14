# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    NodeType

An enum describing the possible node types. Each [`Node`](@ref) has a `.index`
field, which should be interpreted as follows:

 * `NODE_CALL_MULTIVARIATE`: the index into `operators.multivariate_operators`
 * `NODE_CALL_UNIVARIATE`: the index into `operators.univariate_operators`
 * `NODE_LOGIC`: the index into `operators.logic_operators`
 * `NODE_COMPARISON`: the index into `operators.comparison_operators`
 * `NODE_MOI_VARIABLE`: the value of `MOI.VariableIndex(index)` in the user's
   space of the model.
 * `NODE_VARIABLE`: the 1-based index of the internal vector
 * `NODE_VALUE`: the index into the `.values` field of `Expression`
 * `NODE_PARAMETER`: the index into `data.parameters`
 * `NODE_SUBEXPRESSION`:  the index into `data.expressions`
"""
@enum(
    NodeType,
    # Index into multivariate operators
    NODE_CALL_MULTIVARIATE,
    # Index into univariate operators
    NODE_CALL_UNIVARIATE,
    # Index into logic operators
    NODE_LOGIC,
    # Index into comparison operators
    NODE_COMPARISON,
    # Index is the value of `MOI.VariableIndex`. This is from the original
    # model, and is not consecutive.
    NODE_MOI_VARIABLE,
    # Index of the internal, consecutive, and ordered `MOI.VariableIndex`.
    NODE_VARIABLE,
    # Index is into the list of constants
    NODE_VALUE,
    # Index is into the list of parameters
    NODE_PARAMETER,
    # Index is into the list of subexpressions
    NODE_SUBEXPRESSION,
)

"""
    struct Node
        type::NodeType
        index::Int
        parent::Int
    end

A single node in a nonlinear expression tree. Used by
[`Expression`](@ref).

See the MathOptInterface documentation for information on how the nodes and
values form an expression tree.
"""
struct Node
    type::NodeType
    index::Int
    parent::Int
end

"""
    struct Expression{T}
        nodes::Vector{Node}
        values::Vector{T}
    end

The core type that represents a nonlinear expression. See the MathOptInterface
documentation for information on how the nodes and values form an expression
tree.
"""
struct Expression{T}
    nodes::Vector{Node}
    values::Vector{T}

    Expression{T}() where {T} = new{T}(Node[], T[])
end

Expression() = Expression{Float64}()

function Base.:(==)(x::Expression, y::Expression)
    return x.nodes == y.nodes && x.values == y.values
end

"""
    struct Constraint{T}
        expression::Expression{T}
        set::Union{
            MOI.LessThan{T},
            MOI.GreaterThan{T},
            MOI.EqualTo{T},
            MOI.Interval{T},
        }
    end

A type to hold information relating to the nonlinear constraint `f(x) in S`,
where `f(x)` is defined by `.expression`, and `S` is `.set`.
"""
struct Constraint{T}
    expression::Expression{T}
    set::Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    }
end

"""
    ParameterIndex

An index to a nonlinear parameter that is returned by [`add_parameter`](@ref).
Given `data::Model` and `p::ParameterIndex`, use `data[p]` to retrieve
the current value of the parameter and `data[p] = value` to set a new value.
"""
struct ParameterIndex
    value::Int
end

"""
    ExpressionIndex

An index to a nonlinear expression that is returned by [`add_expression`](@ref).

Given `data::Model` and `ex::ExpressionIndex`, use `data[ex]` to
retrieve the corresponding [`Expression`](@ref).
"""
struct ExpressionIndex
    value::Int
end

"""
    ConstraintIndex

An index to a nonlinear constraint that is returned by [`add_constraint`](@ref).

Given `data::Model` and `c::ConstraintIndex`, use `data[c]` to
retrieve the corresponding [`Constraint`](@ref).
"""
struct ConstraintIndex
    value::Int
end

"""
    Model()

The core datastructure for representing a nonlinear optimization problem.

It has the following fields:
 * `objective::Union{Nothing,Expression}` : holds the nonlinear objective
   function, if one exists, otherwise `nothing`.
 * `expressions::Vector{Expression}` : a vector of expressions in the model.
 * `constraints::OrderedDict{ConstraintIndex,Constraint}` : a map from
   [`ConstraintIndex`](@ref) to the corresponding [`Constraint`](@ref). An
   `OrderedDict` is used instead of a `Vector` to support constraint deletion.
 * `parameters::Vector{T}` : holds the current values of the parameters.
 * `operators::OperatorRegistry` : stores the operators used in the model.
"""
mutable struct Model{T}
    objective::Union{Nothing,Expression{T}}
    expressions::Vector{Expression{T}}
    constraints::OrderedDict{ConstraintIndex,Constraint{T}}
    parameters::Vector{T}
    operators::OperatorRegistry
    last_constraint_index::Int64

    function Model{T}() where {T}
        return new{T}(
            nothing,
            Expression{T}[],
            OrderedDict{ConstraintIndex,Constraint{T}}(),
            T[],
            OperatorRegistry(),
            0,
        )
    end
end

Model() = Model{Float64}()

"""
    AbstractAutomaticDifferentiation

An abstract type for extending [`Evaluator`](@ref).
"""
abstract type AbstractAutomaticDifferentiation end

function MOI.Utilities.map_indices(
    ::F,
    backend::AbstractAutomaticDifferentiation,
) where {F<:Function}
    return backend
end

"""
    Evaluator(
        model::Model,
        backend::AbstractAutomaticDifferentiation,
        ordered_variables::Vector{MOI.VariableIndex},
    )

Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct Evaluator{T,B} <: MOI.AbstractNLPEvaluator
    # The internal datastructure.
    model::Model{T}
    # The abstract-differentiation backend
    backend::B
    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{ConstraintIndex}
    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{T}
    # Timers
    initialize_timer::Float64
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_objective_timer::Float64
    eval_hessian_constraint_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function Evaluator(
        model::Model{T},
        backend::B = nothing,
    ) where {T,B<:Union{Nothing,MOI.AbstractNLPEvaluator}}
        return new{T,B}(
            model,
            backend,
            ConstraintIndex[],
            T[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

_bound(s::MOI.LessThan) = MOI.NLPBoundsPair(-Inf, s.upper)
_bound(s::MOI.GreaterThan) = MOI.NLPBoundsPair(s.lower, Inf)
_bound(s::MOI.EqualTo) = MOI.NLPBoundsPair(s.value, s.value)
_bound(s::MOI.Interval) = MOI.NLPBoundsPair(s.lower, s.upper)

"""
    MOI.NLPBlockData(evaluator::Evaluator)

Create an [`MOI.NLPBlockData`](@ref) object from an [`Evaluator`](@ref)
object.
"""
function MOI.NLPBlockData(evaluator::Evaluator)
    return MOI.NLPBlockData(
        [_bound(c.set) for (_, c) in evaluator.model.constraints],
        evaluator,
        evaluator.model.objective !== nothing,
    )
end

"""
    ExprGraphOnly() <: AbstractAutomaticDifferentiation

The default implementation of `AbstractAutomaticDifferentiation`. The only
supported feature is `:ExprGraph`.
"""
struct ExprGraphOnly <: AbstractAutomaticDifferentiation end

function Evaluator(model::Model, ::ExprGraphOnly, ::Vector{MOI.VariableIndex})
    return Evaluator(model)
end

"""
    SparseReverseMode() <: AbstractAutomaticDifferentiation

An implementation of `AbstractAutomaticDifferentiation` that uses sparse
reverse-mode automatic differentiation to compute derivatives. Supports all
features in the MOI nonlinear interface.
"""
struct SparseReverseMode <: AbstractAutomaticDifferentiation end

function Evaluator(
    model::Model,
    ::SparseReverseMode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, ReverseAD.NLPEvaluator(model, ordered_variables))
end

"""
    SymbolicMode() <: AbstractAutomaticDifferentiation

A type for setting as the value of the `MOI.AutomaticDifferentiationBackend()`
attribute to enable symbolic automatic differentiation.
"""
struct SymbolicMode <: AbstractAutomaticDifferentiation end

function Evaluator(
    model::Model,
    ::SymbolicMode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, SymbolicAD.Evaluator(model, ordered_variables))
end
