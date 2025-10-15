import logging
from typing import Tuple
import psutil
import os
import tracemalloc

import numpy
from mpi4py.MPI import Comm

from httomo.data.mpiutil import alltoall, alltoall_ring
from httomo.data.hdf._utils import chunk
from httomo.utils import log_once


def _get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _get_array_size_mb(arr):
    """Get array size in MB."""
    return arr.nbytes / 1024 / 1024


def _get_tracemalloc_mb():
    """Get current tracemalloc memory usage in MB."""
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024, peak / 1024 / 1024
    return 0.0, 0.0


def reslice(
    data: numpy.ndarray,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
) -> Tuple[numpy.ndarray, int, int]:
    """Reslice data by using in-memory MPI directives (using numpy.split).

    Parameters
    ----------
    data : numpy.ndarray
        The data to be re-sliced.
    current_slice_dim : int
        The dimension along which the data is currently sliced.
    next_slice_dim : int
        The dimension along which the data should be sliced after re-chunking
        and saving.
    comm : Comm
        The MPI communicator to be used.

    Returns:
    tuple[numpy.ndarray, int, int]:
        A tuple containing the resliced data, the dimension along which it is
        now sliced, and the starting index in slicing dimension for the current process.
    """
    log_once(
        "<-------Reslicing/rechunking the data (using numpy.split)-------->",
        level=logging.DEBUG,
    )

    rank = comm.rank
    
    # Start tracemalloc if not already running
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    
    tracemalloc.reset_peak()
    mem_start = _get_memory_usage_mb()
    trace_start, trace_peak_start = _get_tracemalloc_mb()

    log_once(
        f"reslice: Starting - psutil memory: {mem_start:.2f} MB, "
        f"tracemalloc current: {trace_start:.2f} MB, peak: {trace_peak_start:.2f} MB",
        level=logging.DEBUG,
    )

    # No need to reslice anything if there is only one process
    if comm.size == 1:
        log_once(
            f"reslice: Not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    # Get shape of full/unsplit data
    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size

    data_size_mb = _get_array_size_mb(data)
    log_once(
        f"reslice: Input data shape={data.shape}, size={data_size_mb:.2f} MB, dtype={data.dtype}",
        level=logging.DEBUG,
    )

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(1, nprocs)]

    log_once(
        f"reslice: Splitting dimension {next_slice_dim} of length {length} into {nprocs} parts",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: Split indices: [0] + {split_indices} + [{length}]",
        level=logging.DEBUG,
    )

    # Prepare list for alltoall using numpy.split
    log_once(
        f"reslice: Calling numpy.split to create {nprocs} slices",
        level=logging.DEBUG,
    )
    
    mem_before_split = _get_memory_usage_mb()
    trace_before_split, _ = _get_tracemalloc_mb()
    
    to_scatter = numpy.split(data, split_indices, axis=next_slice_dim - 1)
    
    mem_after_split = _get_memory_usage_mb()
    trace_after_split, trace_peak_after_split = _get_tracemalloc_mb()

    total_scatter_size_mb = sum(_get_array_size_mb(arr) for arr in to_scatter)
    
    log_once(
        f"reslice: numpy.split completed, created {len(to_scatter)} arrays, "
        f"total size={total_scatter_size_mb:.2f} MB",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: After split - psutil: {mem_after_split:.2f} MB (delta: +{mem_after_split - mem_before_split:.2f} MB), "
        f"tracemalloc: {trace_after_split:.2f} MB (delta: +{trace_after_split - trace_before_split:.2f} MB), "
        f"peak: {trace_peak_after_split:.2f} MB",
        level=logging.DEBUG,
    )
    
    # Log individual slice shapes
    for i, arr in enumerate(to_scatter):
        log_once(
            f"reslice: Slice {i} shape={arr.shape}, size={_get_array_size_mb(arr):.2f} MB",
            level=logging.DEBUG,
        )

    log_once(
        f"reslice: Freeing original data array (size={data_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    
    mem_before_del = _get_memory_usage_mb()
    trace_before_del, _ = _get_tracemalloc_mb()
    
    del data
    
    mem_after_del = _get_memory_usage_mb()
    trace_after_del, trace_peak_after_del = _get_tracemalloc_mb()
    
    log_once(
        f"reslice: Original data freed - psutil: {mem_after_del:.2f} MB (delta: {mem_after_del - mem_before_del:.2f} MB), "
        f"tracemalloc: {trace_after_del:.2f} MB (delta: {trace_after_del - trace_before_del:.2f} MB), "
        f"peak: {trace_peak_after_del:.2f} MB",
        level=logging.DEBUG,
    )

    # All-to-all communication with direct concatenation
    concat_axis = current_slice_dim - 1

    log_once(
        f"reslice: Starting alltoall_ring communication with concat_axis={concat_axis}",
        level=logging.DEBUG,
    )

    mem_before_mpi = _get_memory_usage_mb()
    trace_before_mpi, _ = _get_tracemalloc_mb()
    
    new_data = alltoall_ring(to_scatter, comm, concat_axis=concat_axis)
    
    mem_after_mpi = _get_memory_usage_mb()
    trace_after_mpi, trace_peak_after_mpi = _get_tracemalloc_mb()

    new_data_size_mb = _get_array_size_mb(new_data)
    log_once(
        f"reslice: alltoall_ring completed, received concatenated array "
        f"shape={new_data.shape}, size={new_data_size_mb:.2f} MB",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: After MPI - psutil: {mem_after_mpi:.2f} MB (delta: +{mem_after_mpi - mem_before_mpi:.2f} MB), "
        f"tracemalloc: {trace_after_mpi:.2f} MB (delta: +{trace_after_mpi - trace_before_mpi:.2f} MB), "
        f"peak: {trace_peak_after_mpi:.2f} MB",
        level=logging.DEBUG,
    )

    # Free scatter list
    log_once(
        f"reslice: Freeing to_scatter list (size={total_scatter_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    
    mem_before_free = _get_memory_usage_mb()
    trace_before_free, _ = _get_tracemalloc_mb()
    
    del to_scatter
    
    mem_after_free = _get_memory_usage_mb()
    trace_after_free, trace_peak_after_free = _get_tracemalloc_mb()
    
    log_once(
        f"reslice: to_scatter freed - psutil: {mem_after_free:.2f} MB (delta: {mem_after_free - mem_before_free:.2f} MB), "
        f"tracemalloc: {trace_after_free:.2f} MB (delta: {trace_after_free - trace_before_free:.2f} MB), "
        f"peak: {trace_peak_after_free:.2f} MB",
        level=logging.DEBUG,
    )

    start_idx = 0 if comm.rank == 0 else split_indices[comm.rank - 1]

    mem_end = _get_memory_usage_mb()
    trace_end, trace_peak_end = _get_tracemalloc_mb()
    
    log_once(
        f"reslice: Completed, returning data with slice_dim={next_slice_dim}, start_idx={start_idx}",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: Final memory - psutil: {mem_end:.2f} MB (total delta: {mem_end - mem_start:.2f} MB), "
        f"tracemalloc current: {trace_end:.2f} MB, peak: {trace_peak_end:.2f} MB "
        f"(peak delta from start: {trace_peak_end - trace_peak_start:.2f} MB)",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: Peak memory during operation: "
        f"psutil={max(mem_after_split, mem_after_mpi):.2f} MB, "
        f"tracemalloc={trace_peak_end:.2f} MB",
        level=logging.DEBUG,
    )

    return new_data, next_slice_dim, start_idx