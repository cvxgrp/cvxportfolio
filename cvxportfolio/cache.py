# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Caching functions used by :class:`MarketSimulator`."""

import logging
import os
import pickle
from io import BytesIO
from pathlib import Path

# this global dict is used to store raw file buffers if base_location is set
# to None; it will be copied in full to child processes if parallelizing;
# however, any edit done in a child process (e.g., estimated covariances with
# some HP changed) will be lost as the child process is terminated
from .data.symbol_data import _IN_MEMORY_FILE_STORE

logger = logging.getLogger(__name__)

def _mp_init(l):
    """Shared lock to disk access for multiprocessing."""
    # pylint: disable=global-variable-undefined
    global LOCK # pragma: no cover
    LOCK = l # pragma: no cover

# def _hash_universe(universe):
#     """Hash given universe"""
#     return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()

def cache_name(signature, base_location):
    """Cache name.

    :param signature: Signature of the market data server.
    :type signature: str
    :param base_location: Base storage location. Use ``None`` for
        in-memory storage.
    :type base_location: pathlib.Path or None

    :returns: Storage location.
    :rtype: pathlib.Path
    """
    if base_location is not None: # pragma: no cover
        return (base_location / 'backtest_cache') / (signature + '.pkl')
    else:
        return 'backtest_cache___' + signature

def _load_cache(signature, base_location):
    """Load cache from disk."""
    if signature is None:
        logger.info('Market data has no signature!')
        return {}
    name = cache_name(signature, base_location)
    if 'LOCK' in globals():
        logger.debug( # pragma: no cover
            'Acquiring cache lock from process %s', os.getpid())
        LOCK.acquire() # pragma: no cover
    if not isinstance(name, Path) and not name in _IN_MEMORY_FILE_STORE:
        return {}
    try:
        with (open(name, 'rb') if isinstance(name, Path)
                else BytesIO(_IN_MEMORY_FILE_STORE[name])) as f:
            res = pickle.load(f)
            logger.info('Loaded cache %s', name)
            return res
    except FileNotFoundError:
        logger.info('Cache not found!')
        return {}
    except EOFError: # pragma: no cover
        logger.warning(
            'Cache file %s is corrupt! Discarding it.',
                name) # pragma: no cover
        return {} # pragma: no cover
    finally:
        if 'LOCK' in globals():
            logger.debug( # pragma: no cover
                'Releasing cache lock from process %s', os.getpid())
            LOCK.release() # pragma: no cover

def _store_cache(cache, signature, base_location):
    """Store cache to disk."""
    if signature is None:
        logger.info('Market data has no signature!')
        return
    name = cache_name(signature, base_location)
    if 'LOCK' in globals():
        logger.debug( # pragma: no cover
            'Acquiring cache lock from process %s', os.getpid())
        LOCK.acquire() # pragma: no cover
    if isinstance(name, Path): # pragma: no cover
        name.parent.mkdir(exist_ok=True)
    with (open(name, 'wb') if isinstance(name, Path)
            else BytesIO()) as f:
        logger.info('Storing cache %s', name)
        pickle.dump(cache, f)
        if not isinstance(name, Path):
            _IN_MEMORY_FILE_STORE[name] = f.getvalue()
    if 'LOCK' in globals():
        logger.debug( # pragma: no cover
            'Releasing cache lock from process %s', os.getpid())
        LOCK.release() # pragma: no cover
