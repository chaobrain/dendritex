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
from typing import Sequence

import brainstate as bst
import brainunit as u
import jax
from brainstate._state import record_state_value_write

from ._misc import set_module_as

__all__ = [
    'DiffEqState',
    'DiffEqModule',
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


class DiffEqState(bst.ShortTermState):
    """
    A state that integrates the state of the system to the integral of the state.

    Attributes
    ----------
    derivative: The derivative of the differential equation state.
    diffusion: The diffusion of the differential equation state.

    """

    __module__ = 'dendritex'

    # derivative of this state
    derivative: bst.typing.PyTree
    diffusion: bst.typing.PyTree

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derivative = None
        self._diffusion = None

    @property
    def derivative(self):
        """
        The derivative of the state.

        It is used to compute the derivative of the ODE system,
        or the drift of the SDE system.
        """
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        record_state_value_write(self)
        self._derivative = value

    @property
    def diffusion(self):
        """
        The diffusion of the state.

        It is used to compute the diffusion of the SDE system.
        If it is None, the system is considered as an ODE system.
        """
        return self._diffusion

    @diffusion.setter
    def diffusion(self, value):
        record_state_value_write(self)
        self._diffusion = value


class DiffEqModule(bst.mixin.Mixin):
    __module__ = 'dendritex'

    def pre_integral(self, *args, **kwargs):
        pass

    def compute_derivative(self, *args, **kwargs):
        raise NotImplementedError

    def post_integral(self, *args, **kwargs):
        pass


@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    A: Sequence[Sequence]  # The A matrix in the Butcher tableau.
    B: Sequence  # The B vector in the Butcher tableau.
    C: Sequence  # The C vector in the Butcher tableau.


def _rk_update(
    coeff: Sequence,
    st: bst.State,
    y0: bst.typing.PyTree,
    *ks
):
    assert len(coeff) == len(ks), 'The number of coefficients must be equal to the number of ks.'

    def _step(y0_, *k_):
        kds = [c_ * k_ for c_, k_ in zip(coeff, k_)]
        update = kds[0]
        for kd in kds[1:]:
            update += kd
        return y0_ + update * bst.environ.get_dt()

    st.value = jax.tree.map(_step, y0, *ks, is_leaf=u.math.is_quantity)


@set_module_as('dendritex')
def _general_rk_step(
    tableau: ButcherTableau,
    target: DiffEqModule,
    t: jax.typing.ArrayLike,
    *args
):
    dt = bst.environ.get_dt()

    # before one-step integration
    target.pre_integral(*args)

    # Runge-Kutta stages
    ks = []

    # k1: first derivative step
    assert len(tableau.A[0]) == 0, f'The first row of A must be empty. Got {tableau.A[0]}'
    with bst.environ.context(t=t + tableau.C[0] * dt), bst.StateTraceStack() as trace:
        # compute derivative
        target.compute_derivative(*args)

        # collection of states, initial values, and derivatives
        states = []  # states
        k1hs = []  # k1hs: k1 holder
        y0 = []  # initial values
        for st, val, writen in zip(trace.states, trace.original_state_values, trace.been_writen):
            if isinstance(st, DiffEqState):
                assert writen, f'State {st} must be written.'
                y0.append(val)
                states.append(st)
                k1hs.append(st.derivative)
            else:
                if writen:
                    raise ValueError(f'State {st} is not for integral.')
        ks.append(k1hs)

    # intermediate steps
    for i in range(1, len(tableau.C)):
        with bst.environ.context(t=t + tableau.C[i] * dt), bst.check_state_value_tree():
            for st, y0_, *ks_ in zip(states, y0, *ks):
                _rk_update(tableau.A[i], st, y0_, *ks_)
            target.compute_derivative(*args)
            ks.append([st.derivative for st in states])

    # final step
    with bst.check_state_value_tree():
        # update states with derivatives
        for st, y0_, *ks_ in zip(states, y0, *ks):
            _rk_update(tableau.B, st, y0_, *ks_)


euler_tableau = ButcherTableau(
    A=((),),
    B=(1.0,),
    C=(0.0,),
)
midpoint_tableau = ButcherTableau(
    A=[(),
       (0.5,)],
    B=(0.0, 1.0),
    C=(0.0, 0.5),
)
rk2_tableau = ButcherTableau(
    A=[(),
       (2 / 3,)],
    B=(1 / 4, 3 / 4),
    C=(0.0, 2 / 3),
)
heun2_tableau = ButcherTableau(
    A=[(),
       (1.,)],
    B=[0.5, 0.5],
    C=[0, 1],
)
ralston2_tableau = ButcherTableau(
    A=[(),
       (2 / 3,)],
    B=[0.25, 0.75],
    C=[0, 2 / 3],
)
rk3_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (-1, 2)],
    B=[1 / 6, 2 / 3, 1 / 6],
    C=[0, 0.5, 1],
)
heun3_tableau = ButcherTableau(
    A=[(),
       (1 / 3,),
       (0, 2 / 3)],
    B=[0.25, 0, 0.75],
    C=[0, 1 / 3, 2 / 3],
)
ralston3_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (0, 0.75)],
    B=[2 / 9, 1 / 3, 4 / 9],
    C=[0, 0.5, 0.75],
)
ssprk3_tableau = ButcherTableau(
    A=[(),
       (1,),
       (0.25, 0.25)],
    B=[1 / 6, 1 / 6, 2 / 3],
    C=[0, 1, 0.5],
)
rk4_tableau = ButcherTableau(
    A=[(),
       (0.5,),
       (0., 0.5),
       (0., 0., 1)],
    B=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    C=[0, 0.5, 0.5, 1],
)
ralston4_tableau = ButcherTableau(
    A=[(),
       (.4,),
       (.29697761, .15875964),
       (.21810040, -3.05096516, 3.83286476)],
    B=[.17476028, -.55148066, 1.20553560, .17118478],
    C=[0, .4, .45573725, 1],
)


@set_module_as('dendritex')
def euler_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(euler_tableau, target, t, *args)


@set_module_as('dendritex')
def midpoint_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(midpoint_tableau, target, t, *args)


@set_module_as('dendritex')
def rk2_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk2_tableau, target, t, *args)


@set_module_as('dendritex')
def heun2_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(heun2_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston2_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston2_tableau, target, t, *args)


@set_module_as('dendritex')
def rk3_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk3_tableau, target, t, *args)


@set_module_as('dendritex')
def heun3_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(heun3_tableau, target, t, *args)


@set_module_as('dendritex')
def ssprk3_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ssprk3_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston3_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston3_tableau, target, t, *args)


@set_module_as('dendritex')
def rk4_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(rk4_tableau, target, t, *args)


@set_module_as('dendritex')
def ralston4_step(target: DiffEqModule, t: bst.typing.ArrayLike, *args):
    _general_rk_step(ralston4_tableau, target, t, *args)
