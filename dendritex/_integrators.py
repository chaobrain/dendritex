# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Union, Sequence

import brainstate as bst
import brainunit as u
import diffrax as dfx
import jax

from ._base import State4Integral, DendriteDynamics
from ._misc import set_module_as

__all__ = [
    'diffrax_solve_adjoint',
    'diffrax_solve',
    'euler_step',
    'midpoint_step',
    'rk2_step',
    'heun2_step',
    'ralston2_step',
    'rk3_step',
    'heun3_step',
    'ssprk3_step',
    'ralston3_step',
    'rk4_step',
    'ralston4_step',
]

diffrax_solvers = {
    # explicit RK
    'euler': dfx.Euler,
    'revheun': dfx.ReversibleHeun,
    'heun': dfx.Heun,
    'midpoint': dfx.Midpoint,
    'ralston': dfx.Ralston,
    'bosh3': dfx.Bosh3,
    'tsit5': dfx.Tsit5,
    'dopri5': dfx.Dopri5,
    'dopri8': dfx.Dopri8,

    # implicit RK
    'ieuler': dfx.ImplicitEuler,
    'kvaerno3': dfx.Kvaerno3,
    'kvaerno4': dfx.Kvaerno4,
    'kvaerno5': dfx.Kvaerno5,
}


def _is_quantity(x):
    return isinstance(x, u.Quantity)


def _diffrax_solve(
    model: Callable,
    solver: str,
    t0: u.Quantity,
    t1: u.Quantity,
    dt0: u.Quantity,
    adjoint: str,
    saveat: Optional[u.Quantity] = None,
    savefn: Optional[Callable] = None,
    args: Tuple[bst.typing.PyTree] = (),
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_steps: int = None,
):
    if isinstance(adjoint, str):
        if adjoint == 'adjoint':
            adjoint = dfx.BacksolveAdjoint()
        elif adjoint == 'checkpoint':
            adjoint = dfx.RecursiveCheckpointAdjoint()
        elif adjoint == 'direct':
            adjoint = dfx.DirectAdjoint()
        else:
            raise ValueError(f"Unknown adjoint method: {adjoint}. Only support 'checkpoint', 'direct', and 'adjoint'.")
    elif isinstance(adjoint, dfx.AbstractAdjoint):
        adjoint = adjoint
    else:
        raise ValueError(f"Unknown adjoint method: {adjoint}. Only support 'checkpoint', 'direct', and 'adjoint'.")

    # processing times
    dt0 = dt0.in_unit(u.ms)
    t0 = t0.in_unit(u.ms)
    t1 = t1.in_unit(u.ms)
    if saveat is not None:
        saveat = saveat.in_unit(u.ms)

    # stepsize controller
    if rtol is None and atol is None:
        stepsize_controller = dfx.ConstantStepSize()
    else:
        if rtol is None:
            rtol = atol
        if atol is None:
            atol = rtol
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    # numerical solver
    if solver not in diffrax_solvers:
        raise ValueError(f"Unknown solver: {solver}")
    solver = diffrax_solvers[solver]()

    def model_to_derivative(t, *args):
        with bst.environ.context(t=t * u.ms):
            with bst.StateTraceStack() as trace:
                model(t * u.ms, *args)
                derivatives = []
                for st in trace.states:
                    if isinstance(st, State4Integral):
                        a = u.get_unit(st.derivative) * u.ms
                        b = u.get_unit(st.value)
                        assert a.has_same_dim(b), f'Unit mismatch. Got {a} != {b}'
                        if isinstance(st.derivative, u.Quantity):
                            st.derivative = st.derivative.in_unit(u.get_unit(st.value) / u.ms)
                        derivatives.append(st.derivative)
                    else:
                        raise ValueError(f"State {st} is not for integral.")
                return derivatives

    # stateful function and make jaxpr
    stateful_fn = bst.compile.StatefulFunction(model_to_derivative).make_jaxpr(0., *args)

    # states
    states = stateful_fn.get_states()

    def vector_filed(t, state_vals, args):
        new_state_vals, derivatives = stateful_fn.jaxpr_call(state_vals, t, *args)
        derivatives = tuple(d.mantissa if isinstance(d, u.Quantity) else d
                            for d in derivatives)
        return derivatives

    def save_y(t, state_vals, args):
        for st, st_val in zip(states, state_vals):
            st.value = u.Quantity(st_val, unit=st.value.unit) if isinstance(st.value, u.Quantity) else st_val
        assert callable(savefn), 'savefn must be callable.'
        rets = savefn(t * u.ms, *args)
        nonlocal return_units
        if return_units is None:
            return_units = jax.tree.map(lambda x: x.unit if isinstance(x, u.Quantity) else None, rets,
                                        is_leaf=_is_quantity)
        return jax.tree.map(lambda x: x.mantissa if isinstance(x, u.Quantity) else x, rets, is_leaf=_is_quantity)

    return_units = None
    if savefn is None:
        return_units = tuple(st.value.unit if isinstance(st.value, u.Quantity) else None for st in states)
        if saveat is None:
            if isinstance(adjoint, dfx.BacksolveAdjoint):
                raise ValueError('saveat must be specified when using backsolve adjoint.')
            saveat = dfx.SaveAt(steps=True)
        else:
            saveat = dfx.SaveAt(ts=saveat.mantissa, t1=True)
    else:
        subsaveat_a = dfx.SubSaveAt(t1=True)
        if saveat is None:
            subsaveat_b = dfx.SubSaveAt(steps=True, fn=save_y)
        else:
            subsaveat_b = dfx.SubSaveAt(ts=saveat.mantissa, fn=save_y)
        saveat = dfx.SaveAt(subs=[subsaveat_a, subsaveat_b])

    # solving differential equations
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vector_filed),
        solver,
        t0=t0.mantissa,
        t1=t1.mantissa,
        dt0=dt0.mantissa,
        y0=tuple((v.value.mantissa if isinstance(v.value, u.Quantity) else v.value) for v in states),
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    if savefn is None:
        # assign values back to states
        for st, st_value in zip(states, sol.ys):
            st.value = u.Quantity(st_value[-1], unit=st.unit) if isinstance(st, u.Quantity) else st_value[-1]
        # record solver state
        return (
            sol.ts * u.ms,
            jax.tree.map(
                lambda y, unit: (u.Quantity(y, unit=unit) if unit is not None else y),
                sol.ys,
                return_units
            ),
            sol.stats
        )
    else:
        # assign values back to states
        for st, st_value in zip(states, sol.ys[0]):
            st.value = u.Quantity(st_value[0], unit=st.unit) if isinstance(st, u.Quantity) else st_value[0]
        # record solver state
        return (
            sol.ts[1] * u.ms,
            jax.tree.map(
                lambda y, unit: (u.Quantity(y, unit=unit) if unit is not None else y),
                sol.ys[1],
                return_units
            ),
            sol.stats
        )


def diffrax_solve_adjoint(
    model: Callable,
    solver: str,
    t0: u.Quantity,
    t1: u.Quantity,
    dt0: u.Quantity,
    saveat: Optional[u.Quantity],
    savefn: Optional[Callable] = None,
    args: Tuple[bst.typing.PyTree] = (),
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_steps: Optional[int] = None,
):
    """
    Solve the differential equations using `diffrax <https://docs.kidger.site/diffrax>`_ which
    is compatible with the adjoint backpropagation.

    Args:
      model: The model function to solve.
      solver: The solver to use. Available solvers are:
        - 'euler'
        - 'revheun'
        - 'heun'
        - 'midpoint'
        - 'ralston'
        - 'bosh3'
        - 'tsit5'
        - 'dopri5'
        - 'dopri8'
        - 'ieuler'
        - 'kvaerno3'
        - 'kvaerno4'
        - 'kvaerno5'
      t0: The initial time.
      t1: The final time.
      dt0: The initial step size.
      saveat: The time points to save the solution. If None, save the solution at every step.
      savefn: The function to save the solution. If None, save the solution at every step.
      args: The arguments to pass to the model function.
      rtol: The relative tolerance.
      atol: The absolute tolerance.
      max_steps: The maximum number of steps.

    Returns:
      The solution of the differential equations, including the following items:
        - The time points.
        - The solution.
        - The running step statistics.
    """
    return _diffrax_solve(
        model=model,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        saveat=saveat,
        savefn=savefn,
        adjoint='adjoint',
        max_steps=max_steps,
        args=args,
        rtol=rtol,
        atol=atol,
    )


def diffrax_solve(
    model: Callable,
    solver: str,
    t0: u.Quantity,
    t1: u.Quantity,
    dt0: u.Quantity,
    saveat: Optional[u.Quantity] = None,
    savefn: Optional[Callable] = None,
    args: Tuple[bst.typing.PyTree] = (),
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    max_steps: Optional[int] = None,
    adjoint: Union[str, dfx.AbstractAdjoint] = 'checkpoint',
) -> Tuple[u.Quantity, bst.typing.PyTree[u.Quantity], Dict]:
    """
    Solve the differential equations using `diffrax <https://docs.kidger.site/diffrax>`_.

    Args:
      model: The model function to solve.
      solver: The solver to use. Available solvers are:
        - 'euler'
        - 'revheun'
        - 'heun'
        - 'midpoint'
        - 'ralston'
        - 'bosh3'
        - 'tsit5'
        - 'dopri5'
        - 'dopri8'
        - 'ieuler'
        - 'kvaerno3'
        - 'kvaerno4'
        - 'kvaerno5'
      t0: The initial time.
      t1: The final time.
      dt0: The initial step size.
      saveat: The time points to save the solution. If None, save the solution at every step.
      savefn: The function to save the solution. If None, save the solution at every step.
      args: The arguments to pass to the model function.
      rtol: The relative tolerance.
      atol: The absolute tolerance.
      max_steps: The maximum number of steps.
      adjoint: The adjoint method. Available methods are:
        - 'adjoint'
        - 'checkpoint'
        - 'direct'

    Returns:
      The solution of the differential equations, including the following items:
        - The time points.
        - The solution.
        - The running step statistics.
    """
    return _diffrax_solve(
        model=model,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        saveat=saveat,
        savefn=savefn,
        adjoint=adjoint,
        args=args,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    A: Sequence[Sequence]  # The A matrix in the Butcher tableau.
    B: Sequence  # The B vector in the Butcher tableau.
    C: Sequence  # The C vector in the Butcher tableau.


def _update(dt, coeff, st, y0, *ks):
    assert len(coeff) == len(ks), 'The number of coefficients must be equal to the number of ks.'

    def _step(y0_, *k_):
        kds = [c_ * k_ for c_, k_ in zip(coeff, k_)]
        update = kds[0]
        for kd in kds[1:]:
            update += kd
        return y0_ + update * dt

    st.value = jax.tree.map(_step, y0, *ks, is_leaf=u.math.is_quantity)


@set_module_as('dendritex')
def _general_rk_step(
    tableau: ButcherTableau,
    target: DendriteDynamics,
    t: jax.typing.ArrayLike,
    *args
):
    dt = bst.environ.get_dt()
    time_dtype = u.math.get_dtype(dt)
    ks = []

    # before one-step integration
    target.before_integral(*args)

    # k1
    with bst.environ.context(t=t + u.math.asarray(tableau.C[0], dtype=time_dtype) * dt):
        with bst.StateTraceStack() as trace:
            # compute derivative
            target.compute_derivative(*args)

            # state collection
            states = tuple([st for st in trace.states if isinstance(st, State4Integral)])

            # initial values
            y0 = list([
                val
                for st, val in zip(trace.states, trace.original_state_values)
                if isinstance(st, State4Integral)
            ])

            # derivatives
            k1hs = [st.derivative for st in states]
            ks.append(k1hs)

    for i in range(1, len(tableau.C)):
        with bst.environ.context(t=t + u.math.asarray(tableau.C[i], dtype=time_dtype) * dt):
            with bst.check_state_value_tree():
                for st, y0_, *ks_ in zip(states, y0, *ks):
                    _update(dt, tableau.A[i], st, y0_, *ks_)
                # after one-step derivative
                target.post_derivative(*args)
            target.compute_derivative(*args)
            ks.append([st.derivative for st in states])

    # final step
    with bst.check_state_value_tree():
        # update states with derivatives
        for st, y0_, *ks_ in zip(states, y0, *ks):
            _update(dt, tableau.B, st, y0_, *ks_)
        # update other states
        target.post_derivative(*args)


euler_tableau = ButcherTableau(
    A=((),),
    B=(1.0,),
    C=(0.0,),
)
midpoint_tableau = ButcherTableau(
    A=[(), (0.5,)],
    B=(0.0, 1.0),
    C=(0.0, 0.5),
)
rk2_tableau = ButcherTableau(
    A=[(), (2 / 3,)],
    B=(1 / 4, 3 / 4),
    C=(0.0, 2 / 3),
)
heun2_tableau = ButcherTableau(
    A=[(), (1.,)],
    B=[0.5, 0.5],
    C=[0, 1],
)
ralston2_tableau = ButcherTableau(
    A=[(), (2 / 3,)],
    B=[0.25, 0.75],
    C=[0, 2 / 3],
)
rk3_tableau = ButcherTableau(
    A=[(), (0.5,), (-1, 2)],
    B=[1 / 6, 2 / 3, 1 / 6],
    C=[0, 0.5, 1],
)
heun3_tableau = ButcherTableau(
    A=[(), (1 / 3,), (0, 2 / 3)],
    B=[0.25, 0, 0.75],
    C=[0, 1 / 3, 2 / 3],
)
ralston3_tableau = ButcherTableau(
    A=[(), (0.5,), (0, 0.75)],
    B=[2 / 9, 1 / 3, 4 / 9],
    C=[0, 0.5, 0.75],
)
ssprk3_tableau = ButcherTableau(
    A=[(), (1,), (0.25, 0.25)],
    B=[1 / 6, 1 / 6, 2 / 3],
    C=[0, 1, 0.5],
)
rk4_tableau = ButcherTableau(
    A=[(), (0.5,), (0., 0.5), (0., 0., 1)],
    B=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    C=[0, 0.5, 0.5, 1],
)
ralston4_tableau = ButcherTableau(
    A=[(), (.4,), (.29697761, .15875964), (.21810040, -3.05096516, 3.83286476)],
    B=[.17476028, -.55148066, 1.20553560, .17118478],
    C=[0, .4, .45573725, 1],
)


@set_module_as('dendritex')
def euler_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(euler_tableau, target, t, *args)


@set_module_as('dendritex')
def midpoint_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(midpoint_tableau, target, t, *args)


@set_module_as('dendritex')
def rk2_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk2_tableau, target, t, *args)


@set_module_as('dendritex')
def heun2_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(heun2_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston2_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston2_tableau, target, t, *args)


@set_module_as('dendritex')
def rk3_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk3_tableau, target, t, *args)


@set_module_as('dendritex')
def heun3_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(heun3_tableau, target, t, *args)


@set_module_as('dendritex')
def ssprk3_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ssprk3_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston3_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston3_tableau, target, t, *args)


@set_module_as('dendritex')
def rk4_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk4_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston4_step(target: DendriteDynamics, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston4_tableau, target, t, *args)
