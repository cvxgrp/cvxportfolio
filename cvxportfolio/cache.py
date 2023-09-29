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
import logging


def _mp_init(l):
    """Shared lock to disk access for multiprocessing."""
    global LOCK
    LOCK = l

def _hash_universe(universe):
    """Hash given universe"""
    return hashlib.sha256(bytes(str(tuple(universe)), 'utf-8')).hexdigest()

def _load_cache(universe, trading_frequency, base_location):
    """Load cache from disk."""
    folder = base_location / (
        f'hash(universe)={_hash_universe(universe)},'
        + f'trading_frequency={trading_frequency}')
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    try:
        with open(folder/'cache.pkl', 'rb') as f:
            logging.info(
                f'Loading cache for universe = {universe}'
                f' and trading_frequency = {trading_frequency}')
            return pickle.load(f)
    except FileNotFoundError:
        logging.info(f'Cache not found!')
        return {}
    else:
        logging.info(f'Cache found!')
    finally:
        if 'LOCK' in globals():
            logging.debug(f'Releasing cache lock from process {os.getpid()}')
            LOCK.release()


def _store_cache(cache, universe, trading_frequency, base_location):
    """Store cache to disk."""
    folder = base_location / (
        f'hash(universe)={_hash_universe(universe)},'
        f'trading_frequency={trading_frequency}')
    if 'LOCK' in globals():
        logging.debug(f'Acquiring cache lock from process {os.getpid()}')
        LOCK.acquire()
    folder.mkdir(exist_ok=True)
    with open(folder/'cache.pkl', 'wb') as f:
        logging.info(
            f'Storing cache for universe = {universe} and trading_frequency = {trading_frequency}')
        pickle.dump(cache, f)
    if 'LOCK' in globals():
        logging.debug(f'Releasing cache lock from process {os.getpid()}')
        LOCK.release()