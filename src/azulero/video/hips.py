# SPDX-FileCopyrightText: Copyright (C) 2026, Antoine Basset
# SPDX-PackageSourceInfo: https://github.com/kabasset/azulero
# SPDX-License-Identifier: Apache-2.0

from astropy.coordinates import Angle
from astropy.io import fits
import numpy as np
from pathlib import Path
import requests

# from astroquery.hips2fits import hips2fits  # Requires matplotlib for cmap

from azulero.video.sequence import Frame
from azulero.providers.tiling import Target


def endpoint():
    urls = [
        "http://alasky.cds.unistra.fr/hips-image-services/hips2fits",
        "http://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
    ]
    for u in urls:
        r = requests.get(u)
        if r.status_code == 200:
            return u
    raise requests.HTTPError("No working endpoint found.")


def capture_frame(
    hips_id: str,
    video_format: tuple[int, int],
    frame: Frame,
    path: Path | None = None,
):
    radec = frame.center_in_radec_degrees()
    params = {
        "hips": hips_id,
        "projection": "SIN",
        "width": video_format[0],
        "height": video_format[1],
        "ra": radec[0],
        "dec": radec[1],
        "fov": frame.hfov_in_degrees(),
        "rotation_angle": frame.roll_in_degrees(),
        "format": "fits" if path is None else path.suffix[1:],
    }

    r = requests.get(endpoint(), params=params)
    r.raise_for_status()
    if path is None:
        hdul = fits.HDUList.fromstring(r.content)
        return np.array([channel.data for channel in reversed(hdul)])
    with open(path, "wb") as f:
        f.write(r.content)


def download_cutout(name: str, path: Path, target: Target):
    assert target.radius is not None
    assert target.coord is not None
    hfov = target.radius * 2
    width = int(hfov.to_value("arcsec") * 10 + 0.5)  # type: ignore
    frame = Frame(
        0,
        Angle([target.coord.ra, target.coord.dec]),
        hfov=hfov,
        roll=Angle(0, unit="deg"),
    )
    return capture_frame(name, (width, width), frame, path)
