# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Access to the public "Ultimate fast gyrosynchrotron codes".

The gyrosynchrotron emission is computed by the fast GS codes of Fleishman &
Kuznetsov (2010, ApJ 721, 1127) and Kuznetsov & Fleishman (2021, ApJ 922, 103),
the engine also used by GX Simulator:

    https://github.com/kuznetsov-radio/gyrosynchrotron      (GPL-3.0)

radiosynth does not include that code.  It loads the compiled library
(MWTransferArr.so / .dll) with ctypes and calls its two Python entry points,
pyGET_MW (one line of sight) and pyGET_MW_SLICE (many lines of sight, OpenMP).
The calling conventions are those of the code's CallingConventions.pdf.

Where the library is looked for (first match wins):
  1. the ``path`` argument of load_library() / make_gs_image(library=...);
  2. the environment variable RADIOSYNTH_GSCODES (file, or directory that
     contains the library);
  3. ~/.radiosynth/gyrosynchrotron/binaries/ (where build_library() puts it).

build_library() clones the repository and compiles it with make (g++ and
OpenMP; on macOS "brew install libomp").  If compilation fails it can fall
back to the prebuilt binaries shipped in the repository.
"""
import ctypes
import os
import platform
import shutil
import subprocess

import numpy as np
from numpy.ctypeslib import ndpointer

REPO_URL = "https://github.com/kuznetsov-radio/gyrosynchrotron.git"
DEFAULT_HOME = os.path.join(os.path.expanduser("~"), ".radiosynth", "gyrosynchrotron")

_CACHE = {}


def _candidate_names():
    sysname = platform.system()
    mach = platform.machine().lower()
    if sysname == "Windows":
        return ["MWTransferArr64.dll" if platform.architecture()[0] == "64bit"
                else "MWTransferArr32.dll"]
    if sysname == "Darwin":
        return (["MWTransferArr_arm64.so"] if mach in ("arm64", "aarch64")
                else ["MWTransferArr_x86_64.so", "MWTransferArr.so"])
    if mach in ("arm64", "aarch64"):
        return ["MWTransferArr.so", "MWTransferArr_arm64.so"]
    return ["MWTransferArr.so"]


def find_library(path=None):
    """Return the path of the GS-codes shared library, or raise FileNotFoundError
    with instructions."""
    cands = []
    for base in (path, os.environ.get("RADIOSYNTH_GSCODES"),
                 os.path.join(DEFAULT_HOME, "binaries")):
        if not base:
            continue
        base = os.path.expanduser(base)
        if os.path.isfile(base):
            return base
        for d in (base, os.path.join(base, "binaries")):
            for n in _candidate_names():
                cands.append(os.path.join(d, n))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "The fast gyrosynchrotron codes (Kuznetsov & Fleishman 2021) were not found.\n"
        "Build them once with\n"
        "    python -m radiosynth.gyrosynchrotron.library --build\n"
        "(clones " + REPO_URL + " into ~/.radiosynth/gyrosynchrotron and runs make),\n"
        "or set RADIOSYNTH_GSCODES to the compiled MWTransferArr library / its folder.\n"
        "Looked for: " + ", ".join(cands))


class GSLibrary:
    """ctypes binding of pyGET_MW and pyGET_MW_SLICE."""

    def __init__(self, path):
        self.path = path
        self._lib = ctypes.CDLL(path)
        ip = ndpointer(dtype=np.int32, flags="F")
        dp = ndpointer(dtype=np.float64, flags="F")
        # on Linux/macOS the exported functions return double, on Windows int
        rt = ctypes.c_int if platform.system() == "Windows" else ctypes.c_double
        for name in ("pyGET_MW", "pyGET_MW_SLICE"):
            f = getattr(self._lib, name)
            f.argtypes = [ip, dp, dp, dp, dp, dp, dp]
            f.restype = rt
        self._mw = self._lib.pyGET_MW
        self._slice = self._lib.pyGET_MW_SLICE

    def __repr__(self):
        return f"GSLibrary({self.path})"

    @staticmethod
    def _arrays(*a):
        return [np.asfortranarray(x) for x in a]

    def get_mw(self, Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL):
        """One line of sight (see CallingConventions.pdf).  RL is modified in place."""
        Lparms = np.asfortranarray(Lparms, dtype=np.int32)
        Rparms, Parms, E_arr, mu_arr, f_arr = (np.asfortranarray(x, dtype=np.float64)
                                               for x in (Rparms, Parms, E_arr, mu_arr, f_arr))
        if not (RL.flags.f_contiguous and RL.dtype == np.float64):
            raise ValueError("RL must be a Fortran-ordered float64 array")
        res = int(self._mw(Lparms, Rparms, Parms, E_arr, mu_arr, f_arr, RL))
        if res != 0:
            raise RuntimeError(f"GET_MW returned error code {res}")
        return RL

    def get_mw_slice(self, Lparms_M, Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M, RL_M):
        """Many lines of sight (multi-threaded).  RL_M is modified in place."""
        Lparms_M = np.asfortranarray(Lparms_M, dtype=np.int32)
        Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M = (
            np.asfortranarray(x, dtype=np.float64)
            for x in (Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M))
        if not (RL_M.flags.f_contiguous and RL_M.dtype == np.float64):
            raise ValueError("RL_M must be a Fortran-ordered float64 array")
        res = int(self._slice(Lparms_M, Rparms_M, Parms_M, E_arr, mu_arr, f_arr_M, RL_M))
        if res != 0:
            raise RuntimeError(f"GET_MW_SLICE returned error code {res}")
        return RL_M


def load_library(path=None):
    """Load (once) and return the GSLibrary."""
    p = find_library(path)
    if p not in _CACHE:
        _CACHE[p] = GSLibrary(p)
    return _CACHE[p]


def build_library(dest=DEFAULT_HOME, repo=REPO_URL, ref=None, use_prebuilt=True,
                  verbose=True):
    """Clone the fast GS codes into ``dest`` and compile them with make.

    ref          : git branch/tag/commit to check out (default: latest)
    use_prebuilt : if compilation fails, use the binary shipped in the
                   repository for this platform (if any)
    Returns the path of the library."""
    dest = os.path.expanduser(dest)
    if not os.path.isdir(os.path.join(dest, "source")):
        if shutil.which("git") is None:
            raise RuntimeError("git is needed to download the GS codes "
                               f"(or download {repo} by hand into {dest})")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cmd = ["git", "clone", repo, dest] if ref else ["git", "clone", "--depth", "1", repo, dest]
        if verbose:
            print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        if ref:
            subprocess.run(["git", "-C", dest, "checkout", ref], check=True)
    built = False
    if platform.system() != "Windows" and shutil.which("make"):
        if verbose:
            print(f"compiling in {os.path.join(dest, 'source')} ...")
        r = subprocess.run(["make"], cwd=os.path.join(dest, "source"),
                           capture_output=not verbose)
        built = r.returncode == 0
    if not built:
        if not use_prebuilt:
            raise RuntimeError("compilation of the GS codes failed")
        if verbose:
            print("compilation not possible/failed: using the prebuilt binary of the repository")
    path = find_library(dest)
    _CACHE.pop(path, None)
    if verbose:
        print("GS codes library:", path)
    return path


def library_version(path=None):
    """git commit of the library checkout (if it is a git clone), else None."""
    p = find_library(path)
    d = os.path.dirname(os.path.dirname(p))
    if shutil.which("git") and os.path.isdir(os.path.join(d, ".git")):
        r = subprocess.run(["git", "-C", d, "log", "-1", "--format=%h %cd", "--date=short"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    return None


if __name__ == "__main__":                                  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Fast gyrosynchrotron codes for radiosynth")
    ap.add_argument("--build", action="store_true", help="download and compile the codes")
    ap.add_argument("--dest", default=DEFAULT_HOME)
    ap.add_argument("--ref", default=None, help="git branch/tag/commit")
    a = ap.parse_args()
    if a.build:
        build_library(a.dest, ref=a.ref)
    else:
        print(find_library())
