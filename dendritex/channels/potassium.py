# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent potassium channels.

"""

from __future__ import annotations

from typing import Union, Callable, Optional, Sequence

import brainstate as bst
import brainunit as bu

from .._base import Channel, IonInfo
from .._integrators import State4Integral
from ..ions import Potassium

__all__ = [
  'PotassiumChannel',
  'IKDR_Ba2002',
  'IK_TM1991',
  'IK_HH1952',
  'IKA1_HM1992',
  'IKA2_HM1992',
  'IKK2A_HM1992',
  'IKK2B_HM1992',
  'IKNI_Ya1989',
  'IK_Leak',
]


class PotassiumChannel(Channel):
  """Base class for sodium channel dynamics."""

  root_type = Potassium

  def update(self, V, K: IonInfo):
    raise NotImplementedError

  def current(self, V, K: IonInfo):
    raise NotImplementedError

  def reset_state(self, V, K: IonInfo, batch_size: int = None):
    raise NotImplementedError('Must be implemented by the subclass.')


class _IK_p4_markov(PotassiumChannel):
  r"""The delayed rectifier potassium channel of :math:`p^4`
  current which described with first-order Markov chain.

  This general potassium current model should have the form of

  .. math::

      \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p)
      \end{aligned}

  where :math:`\phi` is a temperature-dependent factor.

  Parameters
  ----------
  size: int, sequence of int
    The object size.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  phi : bst.typing.ArrayLike, Callable
    The temperature-dependent factor.
  name: Optional[str]
    The object name.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.phi = bst.init.param(phi, self.varshape, allow_none=False)

  def init_state(self, V, K: IonInfo, batch_size=None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, K: IonInfo, batch_size: int = None):
    alpha = self.f_p_alpha(V)
    beta = self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    if isinstance(batch_size, int):
      assert self.p.value.shape[0] == batch_size

  def derivative(self, p, t, V):
    return self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p) / bu.ms

  def update(self, V, K: IonInfo):
    self.p.value += self.derivative(self.p.value, bst.environ.get('t'), V) * bst.environ.get_dt()

  def current(self, V, K: IonInfo):
    return self.g_max * self.p.value ** 4 * (K.E - V)

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError


class IKDR_Ba2002(_IK_p4_markov):
  r"""The delayed rectifier potassium channel current.

  The potassium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

      \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &=\frac{0.032\left(V-V_{sh}-15\right)}{1-\exp \left(-\left(V-V_{sh}-15\right) / 5\right)} \\
      \beta_p &= 0.5 \exp \left(-\left(V-V_{sh}-10\right) / 40\right)
      \end{aligned}

  where :math:`\phi` is a temperature-dependent factor, which is given by
  :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

  Parameters
  ----------
  size: int, sequence of int
    The object size.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  T_base : float, ArrayType
    The brainpy_object of temperature factor.
  T : bst.typing.ArrayLike, Callable
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : bst.typing.ArrayLike, Callable
    The shift of the membrane potential to spike.
  name: Optional[str]
    The object name.

  References
  ----------
  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = -50. * bu.mV,
      T_base: bst.typing.ArrayLike = 3.,
      T: bst.typing.ArrayLike = 36.,
      phi: Optional[Union[bst.typing.ArrayLike, Callable]] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    phi = T_base ** ((T / bu.celsius - 36) / 10) if phi is None else phi
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi=phi,
      mode=mode
    )

    # parameters
    self.T = bst.init.param(T, self.varshape, allow_none=False)
    self.T_base = bst.init.param(T_base, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = (V - self.V_sh) / bu.mV
    tmp = V - 15.
    return 0.032 * tmp / (1. - bu.math.exp(-tmp / 5.))

  def f_p_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 0.5 * bu.math.exp(-(V - 10.) / 40.)


class IK_TM1991(_IK_p4_markov):
  r"""The potassium channel described by (Traub and Miles, 1991) [1]_.

  The dynamics of this channel is given by:

  .. math::

     \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &= 0.032 \frac{(15 - V + V_{sh})}{(\exp((15 - V + V_{sh}) / 5) - 1.)} \\
      \beta_p &= 0.5 * \exp((10 - V + V_{sh}) / 40)
      \end{aligned}

  where :math:`V_{sh}` is the membrane shift (default -63 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  name: Optional[str]
    The object name.

  References
  ----------
  .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
         Vol. 777. Cambridge University Press, 1991.

  See Also
  --------
  INa_TM1991
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      V_sh: Union[int, bst.typing.ArrayLike, Callable] = -60. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi=phi,
      g_max=g_max,
      mode=mode
    )
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    c = 15 + (- V + self.V_sh) / bu.mV
    return 0.032 * c / (bu.math.exp(c / 5) - 1.)

  def f_p_beta(self, V):
    V = (self.V_sh - V) / bu.mV
    return 0.5 * bu.math.exp((10 + V) / 40)


class IK_HH1952(_IK_p4_markov):
  r"""The potassium channel described by Hodgkin–Huxley model [1]_.

  The dynamics of this channel is given by:

  .. math::

     \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &= \frac{0.01 (V -V_{sh} + 10)}{1-\exp \left(-\left(V-V_{sh}+ 10\right) / 10\right)} \\
      \beta_p &= 0.125 \exp \left(-\left(V-V_{sh}+20\right) / 80\right)
      \end{aligned}

  where :math:`V_{sh}` is the membrane shift (default -45 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  name: Optional[str]
    The object name.

  References
  ----------
  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
         membrane current and its application to conduction and excitation in
         nerve." The Journal of physiology 117.4 (1952): 500.

  See Also
  --------
  INa_HH1952
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      phi: Union[bst.typing.ArrayLike, Callable] = 1.,
      V_sh: Union[int, bst.typing.ArrayLike, Callable] = -45. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi=phi,
      g_max=g_max,
      mode=mode
    )
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    V = (V - self.V_sh) / bu.mV
    temp = V + 10
    return 0.01 * temp / (1 - bu.math.exp(-temp / 10))

  def f_p_beta(self, V):
    V = (V - self.V_sh) / bu.mV
    return 0.125 * bu.math.exp(-(V + 20) / 80)


class _IKA_p4q_ss(PotassiumChannel):
  r"""
  The rapidly inactivating Potassium channel of :math:`p^4q`
  current which described with steady-state format.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [3]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V) / bu.ms

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V) / bu.ms

  def update(self, V, K: IonInfo):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get_dt()
    self.q.value += self.dq(self.q.value, bst.environ.get('t'), V) * bst.environ.get_dt()

  def current(self, V, K: IonInfo):
    return self.g_max * self.p.value ** 4 * self.q.value * (K.E - V)

  def init_state(self, V, K: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, K: IonInfo, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if isinstance(batch_size, int):
      assert self.p.value.shape[0] == batch_size
      assert self.q.value.shape[0] == batch_size

  def f_p_inf(self, V):
    raise NotImplementedError

  def f_p_tau(self, V):
    raise NotImplementedError

  def f_q_inf(self, V):
    raise NotImplementedError

  def f_q_tau(self, V):
    raise NotImplementedError


class IKA1_HM1992(_IKA_p4q_ss):
  r"""The rapidly inactivating Potassium channel (IA1) model proposed by (Huguenard & McCormick, 1992) [2]_.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [1]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 60)/8.5]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
          \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  See Also
  --------
  IKA2_HM1992
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 30. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_p=phi_p,
      phi_q=phi_q,
      mode=mode
    )

    # parameters
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 60.) / 8.5))

  def f_p_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (bu.math.exp((V + 35.8) / 19.7) +
                 bu.math.exp(-(V + 79.7) / 12.7)) + 0.37

  def f_q_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp((V + 78.) / 6.))

  def f_q_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return bu.math.where(
      V < -63,
      1. / (bu.math.exp((V + 46.) / 5.) +
            bu.math.exp(-(V + 238.) / 37.5)),
      19.
    )


class IKA2_HM1992(_IKA_p4q_ss):
  r"""The rapidly inactivating Potassium channel (IA2) model proposed by (Huguenard & McCormick, 1992) [2]_.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [1]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 36)/20.]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
          \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  See Also
  --------
  IKA1_HM1992
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 20. * (bu.mS / bu.cm ** 2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      g_max=g_max,
      phi_q=phi_q,
      phi_p=phi_p,
      mode=mode
    )

    # parameters
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 36.) / 20.))

  def f_p_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (bu.math.exp((V + 35.8) / 19.7) +
                 bu.math.exp(-(V + 79.7) / 12.7)) + 0.37

  def f_q_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp((V + 78.) / 6.))

  def f_q_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return bu.math.where(
      V < -63,
      1. / (bu.math.exp((V + 46.) / 5.) +
            bu.math.exp(-(V + 238.) / 37.5)),
      19.
    )


class _IKK2_pq_ss(PotassiumChannel):
  r"""The slowly inactivating Potassium channel of :math:`pq`
  current which described with steady-state format.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS / bu.cm ** 2),
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V)

  def update(self, V, K: IonInfo):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get_dt()
    self.q.value += self.dq(self.q.value, bst.environ.get('t'), V) * bst.environ.get_dt()

  def current(self, V, K: IonInfo):
    return self.g_max * self.p.value * self.q.value * (K.E - V)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))
    self.q = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, K: IonInfo, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if isinstance(batch_size, int):
      assert self.p.value.shape[0] == batch_size
      assert self.q.value.shape[0] == batch_size

  def f_p_inf(self, V):
    raise NotImplementedError

  def f_p_tau(self, V):
    raise NotImplementedError

  def f_q_inf(self, V):
    raise NotImplementedError

  def f_q_tau(self, V):
    raise NotImplementedError


class IKK2A_HM1992(_IKK2_pq_ss):
  r"""The slowly inactivating Potassium channel (IK2a) model proposed by (Huguenard & McCormick, 1992) [2]_.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
        \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
     & \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) + \exp((V -V_{sh}+130)/-7.1)} + 120 \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS * bu.cm ** -2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi_p=phi_p,
      phi_q=phi_q,
      g_max=g_max,
      mode=mode
    )

    # parameters
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 43.) / 17.))

  def f_p_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (bu.math.exp((V - 81.) / 25.6) +
                 bu.math.exp(-(V + 132) / 18.)) + 9.9

  def f_q_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp((V + 58.) / 10.6))

  def f_q_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (bu.math.exp((V - 1329.) / 200.) +
                 bu.math.exp(-(V + 130.) / 7.1))


class IKK2B_HM1992(_IKK2_pq_ss):
  r"""The slowly inactivating Potassium channel (IK2b) model proposed by (Huguenard & McCormick, 1992) [2]_.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
     \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) +
                      \exp((V -V_{sh}+130)/-7.1)} + 120 \quad V<(-70+V_{sh})\, mV  \\
          \tau_{q} = 8.9  \quad V \geq (-70 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 10. * (bu.mS * bu.cm ** -2),
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      phi_p=phi_p,
      phi_q=phi_q,
      g_max=g_max,
      mode=mode
    )

    # parameters
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 43.) / 17.))

  def f_p_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (bu.math.exp((V - 81.) / 25.6) +
                 bu.math.exp(-(V + 132) / 18.)) + 9.9

  def f_q_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp((V + 58.) / 10.6))

  def f_q_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    return bu.math.where(
      V < -70,
      1. / (bu.math.exp((V - 1329.) / 200.) +
            bu.math.exp(-(V + 130.) / 7.1)),
      8.9
    )


class IKNI_Ya1989(PotassiumChannel):
  r"""A slow non-inactivating K+ current described by Yamada et al. (1989) [1]_.

  This slow potassium current can effectively account for spike-frequency adaptation.

  .. math::

    \begin{aligned}
    &I_{M}=\bar{g}_{M} p\left(V-E_{K}\right) \\
    &\frac{\mathrm{d} p}{\mathrm{~d} t}=\left(p_{\infty}(V)-p\right) / \tau_{p}(V) \\
    &p_{\infty}(V)=\frac{1}{1+\exp [-(V-V_{sh}+35) / 10]} \\
    &\tau_{p}(V)=\frac{\tau_{\max }}{3.3 \exp [(V-V_{sh}+35) / 20]+\exp [-(V-V_{sh}+35) / 20]}
    \end{aligned}

  where :math:`\bar{g}_{M}` was :math:`0.004 \mathrm{mS} / \mathrm{cm}^{2}` and
  :math:`\tau_{\max }=4 \mathrm{~s}`, unless stated otherwise.

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  name: Optional[str]
    The object name.
  g_max : bst.typing.ArrayLike, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  tau_max: float, ArrayType, Callable, Initializer
    The :math:`tau_{\max}` parameter.

  References
  ----------
  .. [1] Yamada, Walter M. "Multiple channels and calcium dynamics." Methods in neuronal modeling (1989): 97-133.

  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[bst.typing.ArrayLike, Callable] = 0.004 * (bu.mS * bu.cm ** -2),
      phi_p: Union[bst.typing.ArrayLike, Callable] = 1.,
      phi_q: Union[bst.typing.ArrayLike, Callable] = 1.,
      tau_max: Union[bst.typing.ArrayLike, Callable] = 4e3 * bu.ms,
      V_sh: Union[bst.typing.ArrayLike, Callable] = 0. * bu.mV,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size,
      name=name,
      mode=mode
    )

    # parameters
    self.g_max = bst.init.param(g_max, self.varshape, allow_none=False)
    self.tau_max = bst.init.param(tau_max, self.varshape, allow_none=False)
    self.V_sh = bst.init.param(V_sh, self.varshape, allow_none=False)
    self.phi_p = bst.init.param(phi_p, self.varshape, allow_none=False)
    self.phi_q = bst.init.param(phi_q, self.varshape, allow_none=False)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def update(self, V, K: IonInfo):
    self.p.value += self.dp(self.p.value, bst.environ.get('t'), V) * bst.environ.get_dt()

  def current(self, V, K: IonInfo):
    return self.g_max * self.p.value * (K.E - V)

  def init_state(self, V, Ca: IonInfo, batch_size: int = None):
    self.p = State4Integral(bst.init.param(bu.math.zeros, self.varshape, batch_size))

  def reset_state(self, V, K: IonInfo, batch_size=None):
    self.p.value = self.f_p_inf(V)
    if isinstance(batch_size, int):
      assert self.p.value.shape[0] == batch_size

  def f_p_inf(self, V):
    V = (V - self.V_sh) / bu.mV
    return 1. / (1. + bu.math.exp(-(V + 35.) / 10.))

  def f_p_tau(self, V):
    V = (V - self.V_sh) / bu.mV
    temp = V + 35.
    return self.tau_max / (3.3 * bu.math.exp(temp / 20.) + bu.math.exp(-temp / 20.))


class IK_Leak(PotassiumChannel):
  """The potassium leak channel current.

  Parameters
  ----------
  g_max : float
    The potassium leakage conductance which is modulated by both
    acetylcholine and norepinephrine.
  """
  root_type = Potassium

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      g_max: Union[int, bst.typing.ArrayLike, Callable] = 0.005 * (bu.mS * bu.cm ** -2),
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
  ):
    super().__init__(
      size=size,
      name=name,
      mode=mode
    )
    self.g_max = bst.init.param(g_max, self.varshape)

  def reset_state(self, V, K: IonInfo, batch_size: int = None):
    pass

  def update(self, V, K: IonInfo):
    pass

  def current(self, V, K: IonInfo):
    return self.g_max * (K.E - V)
