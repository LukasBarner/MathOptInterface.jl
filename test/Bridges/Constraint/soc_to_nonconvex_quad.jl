# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestConstraintSOCtoNonConvexQuad

using Test

using MathOptInterface
const MOI = MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

include("../utilities.jl")

function test_RSOCtoNonConvexQuad()
    mock = MOI.Utilities.MockOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    )
    config = MOI.Test.Config(
        exclude = Any[MOI.ConstraintDual, MOI.DualObjectiveValue],
    )
    @test MOI.Bridges.Constraint.RSOCtoNonConvexQuadBridge{Float64} ==
          MOI.Bridges.Constraint.concrete_bridge_type(
        MOI.Bridges.Constraint.RSOCtoNonConvexQuadBridge{Float64},
        MOI.VectorOfVariables,
        MOI.RotatedSecondOrderCone,
    )
    @test MOI.supports_constraint(
        MOI.Bridges.Constraint.RSOCtoNonConvexQuadBridge{Float64},
        MOI.VectorOfVariables,
        MOI.RotatedSecondOrderCone,
    )
    @test !MOI.supports_constraint(
        MOI.Bridges.Constraint.RSOCtoNonConvexQuadBridge{Float64},
        MOI.ScalarAffineFunction{Float64},
        MOI.RotatedSecondOrderCone,
    )
    bridged_mock = MOI.Bridges.Constraint.RSOCtoNonConvexQuad{Float64}(mock)
    MOI.Test.runtests(
        bridged_mock,
        config,
        include = ["test_basic_VectorOfVariables_RotatedSecondOrderCone"],
    )
    MOI.empty!(bridged_mock)
    mock.optimize! =
        (mock::MOI.Utilities.MockOptimizer) ->
            MOI.Utilities.mock_optimize!(mock, [0.5, 1.0, 1 / √2, 1 / √2])
    MOI.Test.test_conic_RotatedSecondOrderCone_VectorOfVariables(
        bridged_mock,
        config,
    )
    ci = first(
        MOI.get(
            bridged_mock,
            MOI.ListOfConstraintIndices{
                MOI.VectorOfVariables,
                MOI.RotatedSecondOrderCone,
            }(),
        ),
    )
    _test_delete_bridge(
        bridged_mock,
        ci,
        4,
        (
            (MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}, 0),
            (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}, 0),
        ),
    )
    return
end

function test_SOCtoNonConvexQuad()
    mock = MOI.Utilities.MockOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    )
    config = MOI.Test.Config(
        exclude = Any[MOI.ConstraintDual, MOI.DualObjectiveValue],
    )
    @test MOI.Bridges.Constraint.SOCtoNonConvexQuadBridge{Float64} ==
          MOI.Bridges.Constraint.concrete_bridge_type(
        MOI.Bridges.Constraint.SOCtoNonConvexQuadBridge{Float64},
        MOI.VectorOfVariables,
        MOI.SecondOrderCone,
    )
    @test MOI.supports_constraint(
        MOI.Bridges.Constraint.SOCtoNonConvexQuadBridge{Float64},
        MOI.VectorOfVariables,
        MOI.SecondOrderCone,
    )
    @test !MOI.supports_constraint(
        MOI.Bridges.Constraint.SOCtoNonConvexQuadBridge{Float64},
        MOI.ScalarAffineFunction{Float64},
        MOI.SecondOrderCone,
    )
    bridged_mock = MOI.Bridges.Constraint.SOCtoNonConvexQuad{Float64}(mock)
    MOI.Test.runtests(
        bridged_mock,
        config,
        include = ["test_basic_VectorOfVariables_SecondOrderCone"],
    )
    MOI.empty!(bridged_mock)
    mock.optimize! =
        (mock::MOI.Utilities.MockOptimizer) ->
            MOI.Utilities.mock_optimize!(mock, [1.0, 1 / √2, 1 / √2])
    MOI.Test.test_conic_SecondOrderCone_VectorOfVariables(bridged_mock, config)
    ci = first(
        MOI.get(
            bridged_mock,
            MOI.ListOfConstraintIndices{
                MOI.VectorOfVariables,
                MOI.SecondOrderCone,
            }(),
        ),
    )
    _test_delete_bridge(
        bridged_mock,
        ci,
        3,
        (
            (MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}, 0),
            (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}, 0),
        ),
    )
    return
end

function test_runtests()
    MOI.Bridges.runtests(
        MOI.Bridges.Constraint.SOCtoNonConvexQuadBridge,
        """
        variables: t, x, y
        [t, x, y] in SecondOrderCone(3)
        """,
        """
        variables: t, x, y
        1.0 * x * x + 1.0 * y * y + -1.0 * t * t <= 0.0
        1.0 * t >= 0.0
        """,
    )
    MOI.Bridges.runtests(
        MOI.Bridges.Constraint.RSOCtoNonConvexQuadBridge,
        """
        variables: t, u, x
        [t, u, x] in RotatedSecondOrderCone(3)
        """,
        """
        variables: t, u, x
        1.0 * x * x + -2.0 * t * u <= 0.0
        1.0 * t >= 0.0
        1.0 * u >= 0.0
        """,
    )
    return
end

end  # module

TestConstraintSOCtoNonConvexQuad.runtests()
