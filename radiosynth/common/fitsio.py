# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal FITS image I/O (primary HDU, 2D/3D images) without astropy.

astropy.io.fits is used when installed; otherwise these small routines read and
write simple single-HDU images (BITPIX 8/16/32/-32/-64, BSCALE/BZERO), which
covers Nobeyama/MWA/LOFAR-style solar maps and radiosynth's own output.
"""
import numpy as np

_BLOCK = 2880
_DT = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


def _parse_value(v):
    v = v.split("/")[0].strip() if not v.strip().startswith("'") else v.strip()
    if v.startswith("'"):
        return v[1:v.find("'", 1)].strip()
    if v in ("T", "F"):
        return v == "T"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v.replace("D", "E"))
        except ValueError:
            return v


def read_fits(path, hdu=0):
    """Return (data, header_dict) of an image HDU."""
    try:
        from astropy.io import fits
        with fits.open(path) as hl:
            return np.asarray(hl[hdu].data, float), dict(hl[hdu].header)
    except ImportError:
        pass
    raw = open(path, "rb").read()
    pos = 0
    for k in range(hdu + 1):
        hdr = {}
        while True:
            block = raw[pos:pos + _BLOCK].decode("ascii", "replace")
            pos += _BLOCK
            done = False
            for i in range(0, _BLOCK, 80):
                card = block[i:i + 80]
                key = card[:8].strip()
                if key == "END":
                    done = True
                    break
                if card[8:10] == "= ":
                    hdr[key] = _parse_value(card[10:])
            if done:
                break
        naxis = hdr.get("NAXIS", 0)
        shape = [hdr[f"NAXIS{i}"] for i in range(1, naxis + 1)]
        nbytes = abs(hdr["BITPIX"]) // 8 * int(np.prod(shape)) if naxis else 0
        if k == hdu:
            data = np.frombuffer(raw, dtype=_DT[hdr["BITPIX"]], count=int(np.prod(shape)),
                                 offset=pos).reshape(shape[::-1]).astype(float)
            data = data * hdr.get("BSCALE", 1.0) + hdr.get("BZERO", 0.0)
            return data, hdr
        pos += -(-nbytes // _BLOCK) * _BLOCK
    raise IndexError("HDU not found")


def write_fits(path, data, header=None):
    """Write a float32 image with the given header keywords."""
    data = np.asarray(data, dtype=np.float32)
    try:
        from astropy.io import fits
        h = fits.Header()
        for k, v in (header or {}).items():
            h[k] = v
        fits.PrimaryHDU(data, header=h).writeto(path, overwrite=True)
        return
    except ImportError:
        pass

    def card(k, v):
        if isinstance(v, bool):
            s = "T" if v else "F"
            return f"{k:<8}= {s:>20}".ljust(80)
        if isinstance(v, (int, np.integer)):
            return f"{k:<8}= {int(v):>20}".ljust(80)
        if isinstance(v, (float, np.floating)):
            return f"{k:<8}= {float(v):>20.12E}".ljust(80)
        s = "'" + str(v).replace("'", "''")[:66].ljust(8) + "'"
        return f"{k:<8}= {s:<20}".ljust(80)

    cards = [card("SIMPLE", True), card("BITPIX", -32), card("NAXIS", data.ndim)]
    cards += [card(f"NAXIS{i + 1}", n) for i, n in enumerate(data.shape[::-1])]
    for k, v in (header or {}).items():
        if k not in ("SIMPLE", "BITPIX", "NAXIS") and not k.startswith("NAXIS"):
            cards.append(card(k[:8].upper(), v))
    cards.append("END".ljust(80))
    h = "".join(cards)
    h += " " * (-len(h) % _BLOCK)
    body = data.astype(">f4").tobytes()
    body += b"\0" * (-len(body) % _BLOCK)
    with open(path, "wb") as fh:
        fh.write(h.encode("ascii"))
        fh.write(body)
