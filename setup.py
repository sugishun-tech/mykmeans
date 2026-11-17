"""Portable build; local CPU tuning is an explicit opt-in.

MYKMEANS_OPENMP=0 disables OpenMP (the native serial kernel still builds).
MYKMEANS_OPENMP=1 enables it; default: Linux/Windows on, macOS off.
MYKMEANS_NATIVE=1 enables -march=native on GCC/Clang, for local builds only.
No fast-math: finite-value checks and conservative bounds matter.
"""
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

is_windows = sys.platform == "win32"
is_macos = sys.platform == "darwin"
use_openmp = os.environ.get("MYKMEANS_OPENMP", "0" if is_macos else "1") == "1"
native = os.environ.get("MYKMEANS_NATIVE", "0") == "1"
compile_args = ["/O2"] if is_windows else ["-O3", "-std=c99"]
link_args = []
if use_openmp:
    if is_windows:
        compile_args += ["/openmp"]
    elif is_macos:
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args += ["-lomp"]
    else:
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]
if native and not is_windows:
    compile_args += ["-march=native"]

extensions = [Extension(
    "mykmeans." + module, ["src/mykmeans/" + module + ".pyx"],
    include_dirs=[np.get_include(), "src/mykmeans"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                   ("MYKMEANS_NATIVE_BUILD", "1" if native else "0")],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
) for module in ("_core", "_sparse")]
setup(ext_modules=cythonize(
    extensions,
    compiler_directives={"language_level": 3, "boundscheck": False,
                         "wraparound": False, "initializedcheck": False,
                         "cdivision": True},
))
