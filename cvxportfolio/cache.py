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

import copy
import hashlib
import logging
import os
import pickle


def _mp_init(l):
    """Shared lock to disk access for multiprocessing."""
    global LOCK
    LOCK = l

# def _hash_universe(universe):
#     """Hash given universe"""
#     return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()

def cache_name(signature, base_location):
    """Cache name."""
    return (base_location / 'backtest_cache') / (signature + '.pkl')

def _load_cache(signature, base_location):
    """Load cache from disk."""
    if signature is None:
        logging.info(f'Market data has no signature!')
        return {}
    name = cache_name(signature, base_location)
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    try:
        with open(name, 'rb') as f:
            res = pickle.load(f)
            logging.info(f'Loaded cache {name}')
            return res
    except FileNotFoundError:
        logging.info(f'Cache not found!')
        return {}
    finally:
        if 'LOCK' in globals():
            logging.debug(f'Releasing cache lock from process {os.getpid()}')
            LOCK.release()

def _store_cache(cache, signature, base_location):
    """Store cache to disk."""
    if signature is None:
        logging.info(f'Market data has no signature!')
        return {}
    name = cache_name(signature, base_location)
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    name.parent.mkdir(exist_ok=True)
    with open(name, 'wb') as f:
        logging.info(f'Storing cache {name}')
        pickle.dump(cache, f)
    if 'LOCK' in globals():
        logging.debug(f'Releasing cache lock from process {os.getpid()}')
        LOCK.release()