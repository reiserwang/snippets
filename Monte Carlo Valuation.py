#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
S0	=	100.
K	=	105.
T	=	1.0
r	=	0.05
sigma	=	0.2
I	=	100000      #number of simulations

z     =     np.random.standard_normal(I)
ST    =     S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*z)
hT    =     np.maximum(ST-K,0)
C0    =     np.exp(-r*T)*np.sum(hT)/I

print "value of the European Call Option %5.3f" %C0