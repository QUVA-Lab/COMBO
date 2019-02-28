from __future__ import division
import numpy

from openmdao.api import Component


class velcoupling(Component):
    """ Couples v, a, and M """

    def __init__(self):
        super(velcoupling, self).__init__()

        self.add_param('a', val=280.0)
        self.add_param('M', val=1.0)
        self.add_output('v', val=0.)
    
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['v'] = params['a']*params['M']

    def linearize(self, params, unknowns, resids):
        """ Jacobian for circulations."""

        jac = self.alloc_jacobian()
        jac['v', 'a'] = params['M']
        jac['v', 'M'] = params['a']

        return jac
