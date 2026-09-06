# SPDX-FileCopyrightText: Copyright (C) 2025-2026, Antoine Basset
# SPDX-PackageSourceInfo: https://github.com/kabasset/azulero
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from azulero.image import color


def test_sharpening_border_conditions():
    data = np.ones([3, 3, 3], dtype=float)
    data[1, 1, 1] = 2
    data = color.sharpen(data, [1, 1, 1], 1)
    assert np.all(data[0] == 1)
    assert data[1, 1, 1] > 2
    assert data[1, 0, 0] < 1
    assert data[1, 1, 0] < 1
    assert data[1, 1, 0] < data[1, 0, 0]
    assert np.all(data[2] == 1)


def test_abmag_to_value():
    assert color.abmag_to_value(0, 0) == 1
    assert color.abmag_to_value(2.5, 0) == 0.1
    assert color.abmag_to_value(0, 2.5) == 10


def test_iyj_to_lbgr():
    iyjh = np.array([[[1]], [[2]], [[4]], [[8]]], dtype=float)
    no_h_transform = color.Transform(nir_to_l=0, i_to_b=1, y_to_g=1, j_to_r=1)
    lbgr = color.iyjh_to_lbgr(iyjh, no_h_transform)
    assert lbgr[0, 0, 0] == 1
    assert lbgr[0, 0, 1] == 1
    assert lbgr[0, 0, 2] == 2
    assert lbgr[0, 0, 3] == 4


def test_yjh_to_lbgr():
    iyjh = np.array([[[1]], [[2]], [[4]], [[8]]], dtype=float)
    no_i_transform = color.Transform(nir_to_l=1, i_to_b=0, y_to_g=0, j_to_r=0)
    lbgr = color.iyjh_to_lbgr(iyjh, no_i_transform)
    assert lbgr[0, 0, 0] == 4
    assert lbgr[0, 0, 1] == 2
    assert lbgr[0, 0, 2] == 4
    assert lbgr[0, 0, 3] == 8
