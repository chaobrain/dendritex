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

import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt

import dendritex as dx

bst.environ.set(dt=0.01 * u.ms)


class HH(dx.neurons.SingleCompartment):
    def __init__(self, size):
        super().__init__(size)

        self.na = dx.ions.SodiumFixed(size, E=50. * u.mV)
        self.na.add_elem(INa=dx.channels.INa_HH1952(size))

        self.k = dx.ions.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add_elem(IK=dx.channels.IK_HH1952(size))

        self.IL = dx.channels.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))

    def step_fun(self, t):
        dx.rk2_step(self, t, 10 * u.nA / u.cm ** 2)
        # spike = self.update()
        return self.V.value


hh = HH([1, 1])
hh.init_state()

times = u.math.arange(10000) * bst.environ.get_dt()
vs = bst.compile.for_loop(hh.step_fun, times)

plt.plot(times, u.math.squeeze(vs))
plt.show()
