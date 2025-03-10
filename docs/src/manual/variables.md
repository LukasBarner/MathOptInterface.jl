```@meta
CurrentModule = MathOptInterface
DocTestSetup = quote
    using MathOptInterface
    const MOI = MathOptInterface
end
DocTestFilters = [r"MathOptInterface|MOI"]
```

# Variables

## Add a variable

Use [`add_variable`](@ref) to add a single variable.

```jldoctest variables; setup=:(model = MOI.Utilities.Model{Float64}(); )
julia> x = MOI.add_variable(model)
MathOptInterface.VariableIndex(1)
```
[`add_variable`](@ref) returns a [`VariableIndex`](@ref) type, which is used to
refer to the added variable in other calls.

Check if a [`VariableIndex`](@ref) is valid using [`is_valid`](@ref).
```jldoctest variables
julia> MOI.is_valid(model, x)
true
```

Use [`add_variables`](@ref) to add a number of variables.
```jldoctest variables
julia> y = MOI.add_variables(model, 2)
2-element Vector{MathOptInterface.VariableIndex}:
 MathOptInterface.VariableIndex(2)
 MathOptInterface.VariableIndex(3)
```

!!! warning
    The integer does not necessarily correspond to the column inside an
    optimizer!

## Delete a variable

Delete a variable using [`delete`](@ref).

```jldoctest variables
julia> MOI.delete(model, x)

julia> MOI.is_valid(model, x)
false
```

!!! warning
    Not all `ModelLike` models support deleting variables. A
    [`DeleteNotAllowed`](@ref) error is thrown if this is not supported.

## Variable attributes

The following attributes are available for variables:

* [`VariableName`](@ref)
* [`VariablePrimalStart`](@ref)
* [`VariablePrimal`](@ref)

Get and set these attributes using [`get`](@ref) and [`set`](@ref).

```jldoctest variables
julia> MOI.set(model, MOI.VariableName(), x, "var_x")

julia> MOI.get(model, MOI.VariableName(), x)
"var_x"
```
