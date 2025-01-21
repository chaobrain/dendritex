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


# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Dict, Sequence, Callable, NamedTuple, Tuple

import brainstate as bst
import numpy as np
from brainstate.mixin import _JointGenericAlias

from ._protocol import DiffEqModule
from ._misc import set_module_as, Container, TreeNode

__all__ = [
    'HHTypedNeuron',
    'IonChannel',
    'Ion',
    'MixIons',
    'Channel',
    'IonInfo',
]

'''
- HHTypedNeuron
  - SingleCompartment
- IonChannel
  - Ion
    - Calcium
    - Potassium
    - Sodium
  - MixIons
  - Channel
'''


class HHTypedNeuron(bst.nn.Dynamics, Container, DiffEqModule):
    """
    The base class for the Hodgkin-Huxley typed neuronal membrane dynamics.
    """
    __module__ = 'dendritex'
    _container_name = 'ion_channels'

    def __init__(
        self,
        size: bst.typing.Size,
        name: Optional[str] = None,
        **ion_channels
    ):
        # size
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise ValueError(f'size must be int, or a tuple/list of int. '
                                 f'But we got {type(size)}')
            if not isinstance(size[0], (int, np.integer)):
                raise ValueError('size must be int, or a tuple/list of int.'
                                 f'But we got {type(size)}')
            size = tuple(size)
        elif isinstance(size, (int, np.integer)):
            size = (size,)
        else:
            raise ValueError('size must be int, or a tuple/list of int.'
                             f'But we got {type(size)}')
        self.size = size
        assert len(size) >= 1, ('The size of the dendritic dynamics should be at '
                                'least 1D: (..., n_neuron, n_compartment).')
        self.pop_size: Tuple[int, ...] = size[:-1]
        self.n_compartment: int = size[-1]

        # initialize
        super().__init__(size, name=name)

        # attribute for ``Container``
        self.ion_channels = self._format_elements(IonChannel, **ion_channels)

    def current(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def pre_integral(self, *args, **kwargs):
        raise NotImplementedError

    def compute_derivative(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def post_integral(self, *args, **kwargs):
        """
        For the neuron model, the `post_integral()` is the `update()` function.
        """
        pass

    def init_state(self, batch_size=None):
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        TreeNode.check_hierarchies(self.__class__, *nodes)
        for channel in nodes:
            channel.init_state(self.V.value, batch_size=batch_size)

    def reset_state(self, batch_size=None):
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        for channel in nodes:
            channel.reset_state(self.V.value, batch_size=batch_size)

    def add_elem(self, **elements):
        """
        Add new elements.

        Args:
          elements: children objects.
        """
        TreeNode.check_hierarchies(type(self), **elements)
        self.ion_channels.update(self._format_elements(IonChannel, **elements))


class IonChannel(bst.graph.Node, TreeNode, DiffEqModule):
    """
    The base class for ion channel modeling.

    :py:class:`IonChannel` can be used to model the dynamics of an ion (instance of :py:class:`Ion`), or
    a mixture of ions (instance of :py:class:`MixIons`), or a channel (instance of :py:class:`Channel`).

    Particularly, an implementation of a :py:class:`IonChannel` should implement the following methods:

    - :py:meth:`current`: Calculate the current of the ion channel.
    - :py:meth:`before_integral`: Calculate the state variables before the integral.
    - :py:meth:`compute_derivative`: Calculate the derivative of the state variables.
    - :py:meth:`after_integral`: Calculate the state variables after the integral.
    - :py:meth:`init_state`: Initialize the state variables.
    - :py:meth:`reset_state`: Reset the state variables.

    """
    __module__ = 'dendritex'

    def __init__(
        self,
        size: bst.typing.Size,
        name: Optional[str] = None,
    ):
        # size
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise ValueError(f'size must be int, or a tuple/list of int. '
                                 f'But we got {type(size)}')
            if not isinstance(size[0], (int, np.integer)):
                raise ValueError('size must be int, or a tuple/list of int.'
                                 f'But we got {type(size)}')
            size = tuple(size)
        elif isinstance(size, (int, np.integer)):
            size = (size,)
        else:
            raise ValueError('size must be int, or a tuple/list of int.'
                             f'But we got {type(size)}')
        self.size = size
        assert len(size) >= 1, ('The size of the dendritic dynamics should be at '
                                'least 1D: (..., n_neuron, n_compartment).')
        self.name = name

    @property
    def varshape(self):
        """The shape of variables in the neuron group."""
        return self.size

    def current(self, *args, **kwargs):
        raise NotImplementedError

    def pre_integral(self, *args, **kwargs):
        pass

    def compute_derivative(self, *args, **kwargs):
        raise NotImplementedError

    def post_integral(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def reset_state(self, *args, **kwargs):
        pass

    def init_state(self, *args, **kwargs):
        pass


class IonInfo(NamedTuple):
    """
    The information of the ion.

    Attributes:
        E: The reversal potential.
        C: The ion concentration.
    """
    C: bst.typing.ArrayLike
    E: bst.typing.ArrayLike


class Ion(IonChannel, Container):
    """
    The base class for modeling the Ion dynamics.

    Args:
      size: The size of the simulation target.
      name: The name of the object.
    """
    __module__ = 'dendritex'
    _container_name = 'channels'

    # The type of the master object.
    root_type = HHTypedNeuron

    def __init__(
        self,
        size: bst.typing.Size,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name)
        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

        self._external_currents: Dict[str, Callable] = dict()

    def pre_integral(self, V):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.pre_integral(V, self.pack_info())

    def compute_derivative(self, V):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.compute_derivative(V, self.pack_info())

    def post_integral(self, V):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))
        for node in nodes.values():
            node.post_integral(V, self.pack_info())

    def current(self, V, include_external: bool = False):
        """
        Generate ion channel current.

        Args:
          V: The membrane potential.
          include_external: Include the external current.

        Returns:
          Current.
        """
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        ion_info = self.pack_info()
        current = None
        if len(nodes) > 0:
            for node in nodes:
                node: Channel
                new_current = node.current(V, ion_info)
                current = new_current if current is None else (current + new_current)
        if include_external:
            for key, node in self._external_currents.items():
                node: Callable
                current = current + node(V, ion_info)
        return current

    def init_state(self, V, batch_size: int = None):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(type(self), *tuple(nodes))
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.init_state(V, ion_info, batch_size)

    def reset_state(self, V, batch_size: int = None):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        ion_info = self.pack_info()
        for node in nodes:
            node: Channel
            node.reset_state(V, ion_info, batch_size)

    def register_external_current(self, key: str, fun: Callable):
        if key in self._external_currents:
            raise ValueError
        self._external_currents[key] = fun

    def pack_info(self) -> IonInfo:
        E = self.E
        E = E.value if isinstance(E, bst.State) else E
        C = self.C.value if isinstance(self.C, bst.State) else self.C
        return IonInfo(E=E, C=C)

    def add_elem(self, **elements):
        """
        Add new elements.

        Args:
          elements: children objects.
        """
        self.check_hierarchies(type(self), **elements)
        self.channels.update(self._format_elements(object, **elements))


class MixIons(IonChannel, Container):
    """
    Mixing Ions.

    Args:
      ions: Instances of ions. This option defines the master types of all children objects.
    """
    __module__ = 'dendritex'

    root_type = HHTypedNeuron
    _container_name = 'channels'

    def __init__(self, *ions, name: Optional[str] = None, **channels):
        # TODO: check "ions" should be independent from each other
        assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ions. '
        assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
        size = ions[0].size
        for ion in ions:
            assert ion.size == size, f'The size of all ions should be the same. But we got {ions}.'
        super().__init__(size=size, name=name)

        # Store the ion instances
        self.ions: Sequence['Ion'] = tuple(ions)
        self._ion_types = tuple([type(ion) for ion in self.ions])

        # Store the ion channel channels
        self.channels: Dict[str, Channel] = dict()
        self.channels.update(self._format_elements(Channel, **channels))

    def pre_integral(self, V):
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.pre_integral(V, *ion_infos)

    def compute_derivative(self, V):
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.compute_derivative(V, *ion_infos)

    def post_integral(self, V):
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            ion_infos = tuple([self._get_ion(ion).pack_info() for ion in node.root_type.__args__])
            node.post_integral(V, *ion_infos)

    def current(self, V):
        """Generate ion channel current.

        Args:
          V: The membrane potential.

        Returns:
          Current.
        """
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())

        if len(nodes) == 0:
            return 0.
        else:
            current = None
            for node in nodes:
                infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
                current = node.current(V, *infos) if current is None else (current + node.current(V, *infos))
            return current

    def init_state(self, V, batch_size: int = None):
        nodes = bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(self._ion_types, *tuple(nodes), check_fun=self._check_hierarchy)
        for node in nodes:
            node: Channel
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.init_state(V, *infos, batch_size)

    def reset_state(self, V, batch_size=None):
        nodes = tuple(bst.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values())
        for node in nodes:
            infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
            node.reset_state(V, *infos, batch_size)

    def _check_hierarchy(self, ions, leaf):
        # 'root_type' should be a brainpy.mixin.JointType
        self._check_root(leaf)
        for cls in leaf.root_type.__args__:
            if not any([issubclass(root, cls) for root in ions]):
                raise TypeError(
                    f'Type does not match. {leaf} requires a master with type '
                    f'of {leaf.root_type}, but the master type now is {ions}.'
                )

    def add_elem(self, **elements):
        """
        Add new elements.

        Args:
          elements: children objects.
        """
        self.check_hierarchies(self._ion_types, check_fun=self._check_hierarchy, **elements)
        self.channels.update(self._format_elements(Channel, **elements))
        for elem in tuple(elements.values()):
            elem: Channel
            for ion_root in elem.root_type.__args__:
                ion = self._get_ion(ion_root)
                ion.register_external_current(elem.name, self._get_ion_fun(ion, elem))

    def _get_ion_fun(self, ion: 'Ion', node: 'Channel'):
        def fun(V, ion_info):
            infos = tuple(
                [(ion_info if isinstance(ion, root) else self._get_ion(root).pack_info())
                 for root in node.root_type.__args__]
            )
            return node.current(V, *infos)

        return fun

    def _get_ion(self, cls):
        for ion in self.ions:
            if isinstance(ion, cls):
                return ion
        else:
            raise ValueError(f'No instance of {cls} is found.')

    def _check_root(self, leaf):
        if not isinstance(leaf.root_type, _JointGenericAlias):
            raise TypeError(
                f'{self.__class__.__name__} requires leaf nodes that have the root_type of '
                f'"brainpy.mixin.JointType". However, we got {leaf.root_type}'
            )


@set_module_as('dendritex')
def mix_ions(*ions) -> MixIons:
    """Create mixed ions.

    Args:
      ions: Ion instances.

    Returns:
      Instance of MixIons.
    """
    for ion in ions:
        assert isinstance(ion, Ion), f'Must be instance of {Ion.__name__}. But got {type(ion)}'
    assert len(ions) > 0, ''
    return MixIons(*ions)


class Channel(IonChannel):
    """
    The base class for modeling channel dynamics.
    """
    __module__ = 'dendritex'
