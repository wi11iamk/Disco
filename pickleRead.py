#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:06:33 2024

@author: wi11iamk
"""

## Pickle retrieval 

import pickle

objects = []
with (open("/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/sample_labelsets.p", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break