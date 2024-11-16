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

from typing import Callable

import brainstate as bst


def set_module_as(name: str):
    def decorator(module):
        module.__name__ = name
        return module

    return decorator


class Container(bst.mixin.Mixin):
    __module__ = 'dendritex'

    _container_name: str

    @staticmethod
    def _format_elements(child_type: type, **children_as_dict):
        res = dict()

        # add dict-typed components
        for k, v in children_as_dict.items():
            if not isinstance(v, child_type):
                raise TypeError(f'Should be instance of {child_type.__name__}. '
                                f'But we got {type(v)}')
            res[k] = v
        return res

    def __getitem__(self, item):
        """
        Overwrite the slice access (`self['']`).
        """
        children = self.__getattr__(self._container_name)
        if item in children:
            return children[item]
        else:
            raise ValueError(f'Unknown item {item}, we only found {list(children.keys())}')

    def __getattr__(self, item):
        """
        Overwrite the dot access (`self.`).
        """
        name = super().__getattribute__('_container_name')
        if item == '_container_name':
            return name
        children = super().__getattribute__(name)
        if item == name:
            return children
        if item in children:
            return children[item]
        else:
            return super().__getattribute__(item)

    def add_elem(self, *elems, **elements):
        """
        Add new elements.

        Args:
          elements: children objects.
        """
        raise NotImplementedError('Must be implemented by the subclass.')


class TreeNode(bst.mixin.Mixin):
    __module__ = 'dendritex'

    root_type: type

    @staticmethod
    def _root_leaf_pair_check(root: type, leaf: 'TreeNode'):
        if hasattr(leaf, 'root_type'):
            root_type = leaf.root_type
        else:
            raise ValueError('Child class should define "root_type" to '
                             'specify the type of the root node. '
                             f'But we did not found it in {leaf}')
        if not issubclass(root, root_type):
            raise TypeError(f'Type does not match. {leaf} requires a root with type '
                            f'of {leaf.root_type}, but the root now is {root}.')

    @staticmethod
    def check_hierarchies(root: type, *leaves, check_fun: Callable = None, **named_leaves):
        if check_fun is None:
            check_fun = TreeNode._root_leaf_pair_check

        for leaf in leaves:
            if isinstance(leaf, bst.graph.Node):
                check_fun(root, leaf)
            elif isinstance(leaf, (list, tuple)):
                TreeNode.check_hierarchies(root, *leaf, check_fun=check_fun)
            elif isinstance(leaf, dict):
                TreeNode.check_hierarchies(root, **leaf, check_fun=check_fun)
            else:
                raise ValueError(f'Do not support {type(leaf)}.')
        for leaf in named_leaves.values():
            if not isinstance(leaf, bst.graph.Node):
                raise ValueError(f'Do not support {type(leaf)}. Must be instance of {bst.graph.Node}')
            check_fun(root, leaf)
