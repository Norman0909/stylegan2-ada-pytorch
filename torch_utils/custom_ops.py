# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import torch
import torch.utils.cpp_extension
import importlib
import hashlib
import shutil
import sys
from pathlib import Path

from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

def _find_ninja_bindir():
    patterns = [
        os.path.join(os.path.dirname(sys.executable), 'Scripts'),
        os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'),
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja',
    ]
    for pattern in patterns:
        for bindir in reversed(sorted(glob.glob(pattern))):
            if os.path.isfile(os.path.join(bindir, 'ninja.exe')):
                return bindir
    return None

def _prepend_path(bindir):
    current = os.environ.get('PATH', '')
    entries = current.split(os.pathsep) if current else []
    if bindir not in entries:
        os.environ['PATH'] = os.pathsep.join([bindir] + entries)

def _setup_windows_build_environment():
    if os.name != 'nt':
        return

    # PyTorch's Windows extension builder needs both Ninja and MSVC.  The
    # Conda interpreter may be launched without its Scripts directory on PATH.
    if shutil.which('cl.exe') is None:
        compiler_bindir = _find_compiler_bindir()
        if compiler_bindir is None:
            raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
        _prepend_path(compiler_bindir)
    if shutil.which('ninja.exe') is None:
        ninja_bindir = _find_ninja_bindir()
        if ninja_bindir is None:
            raise RuntimeError(f'Could not find ninja.exe for the active Python environment. Check _find_ninja_bindir() in "{__file__}".')
        _prepend_path(ninja_bindir)

    # torch.utils.cpp_extension._run_ninja_build() otherwise reconstructs its
    # own VC environment and can overwrite PATH with the launcher PATH.  Keep
    # the standard INCLUDE/LIB variables from that environment, but preserve
    # the corrected PATH assembled above.
    if 'VSCMD_ARG_TGT_ARCH' not in os.environ:
        vc_env = torch.utils.cpp_extension._get_vc_env('x64')  # pylint: disable=protected-access
        for key, value in vc_env.items():
            if key.upper() != 'PATH':
                os.environ[key.upper()] = value
        os.environ['VSCMD_ARG_TGT_ARCH'] = 'x64'

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler and build binaries.
        _setup_windows_build_environment()

        # Compile and load.
        verbose_build = (verbosity == 'full')

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        source_dirs_set = set(os.path.dirname(source) for source in sources)
        if len(source_dirs_set) == 1 and ('TORCH_EXTENSIONS_DIR' in os.environ):
            all_source_files = sorted(list(x for x in Path(list(source_dirs_set)[0]).iterdir() if x.is_file()))

            # Compute a combined hash digest for all source files in the same
            # custom op directory (usually .cu, .cpp, .py and .h files).
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            digest_build_dir = os.path.join(build_dir, hash_md5.hexdigest())

            if not os.path.isdir(digest_build_dir):
                os.makedirs(digest_build_dir, exist_ok=True)
                baton = FileBaton(os.path.join(digest_build_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        for src in all_source_files:
                            shutil.copyfile(src, os.path.join(digest_build_dir, os.path.basename(src)))
                    finally:
                        baton.release()
                else:
                    # Someone else is copying source files under the digest dir,
                    # wait until done and continue.
                    baton.wait()
            digest_sources = [os.path.join(digest_build_dir, os.path.basename(x)) for x in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,
                verbose=verbose_build, sources=digest_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
