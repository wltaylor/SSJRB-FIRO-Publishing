#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:59:38 2023

@author: williamtaylor
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

seeds = [-0.5568283040482762,
         -0.5473390985148735,
         -0.5518756614822883, 
         -0.556222843997829,
         -0.5539412673770513, 
         -0.5494754349627664, 
         -0.5471569966066341, 
         -0.5521837512659394, 
         -0.5527320895689278,
         -0.549681037004277]

seeds2 = [-0.5562882640103223,
          -0.5562993579162469,
          -0.5557847777662223,
          -0.5546911969578029,
          -0.5549898358130971,
          -0.5525988084976877,
          -0.5552557245431993,
          -0.5544316700162316,
          -0.5556774175503673,
          -0.5557353003855473]


plt.rcParams['figure.figsize'] = [8, 6]

plt.plot(seeds)
plt.plot(seeds2)
plt.title("Random Seed Test Results")
plt.xlabel("Random Seed")
plt.ylabel("Objective Function Value")
plt.gca().invert_yaxis()
plt.legend(['Old Model','New Model'])

print(np.mean(seeds))
print(np.std(seeds))

print(np.mean(seeds2))
print(np.std(seeds2))