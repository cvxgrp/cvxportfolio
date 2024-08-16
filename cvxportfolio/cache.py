# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""Caching functions used by :class:`MarketSimulator`."""

import logging
import os
import pickle

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
    :param base_location: Base storage location.
    :type base_location: pathlib.Path

    :returns: Storage location.
    :rtype: pathlib.Path
    """
    return (base_location / 'backtest_cache') / (signature + '.pkl')

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
    try:
        with open(name, 'rb') as f:
            res = pickle.load(f)
            logger.info('Loaded cache %s', name)
            return res
    except FileNotFoundError:
        logger.info('Cache not found!')
        return {}
    except (EOFError, ModuleNotFoundError): # pragma: no cover
        logger.warning(
            'Cache file %s is corrupt or un-readable! Discarding it.',
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
    name.parent.mkdir(exist_ok=True)
    with open(name, 'wb') as f:
        logger.info('Storing cache %s', name)
        pickle.dump(cache, f)
    if 'LOCK' in globals():
        logger.debug( # pragma: no cover
            'Releasing cache lock from process %s', os.getpid())
        LOCK.release() # pragma: no cover
