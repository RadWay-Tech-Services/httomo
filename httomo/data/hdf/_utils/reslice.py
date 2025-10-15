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
    """Reslice data by using in-memory MPI directives (using numpy.take).

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
        "<-------Reslicing/rechunking the data (using numpy.take)-------->",
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

    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size

    data_size_mb = _get_array_size_mb(data)
    log_once(
        f"reslice: Input data shape={data.shape}, size={data_size_mb:.2f} MB, dtype={data.dtype}",
        level=logging.DEBUG,
    )

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(nprocs + 1)]

    log_once(
        f"reslice: Splitting dimension {next_slice_dim} of length {length} into {nprocs} parts",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: Split indices: {split_indices}",
        level=logging.DEBUG,
    )

    # Prepare list for alltoall using numpy.take
    log_once(
        f"reslice: Creating {nprocs} slices using numpy.take",
        level=logging.DEBUG,
    )

    mem_before_take = _get_memory_usage_mb()
    trace_before_take, _ = _get_tracemalloc_mb()
    
    to_scatter = []
    total_scatter_size_mb = 0
    
    for i in range(nprocs):
        mem_before_slice = _get_memory_usage_mb()
        trace_before_slice, _ = _get_tracemalloc_mb()
        
        start = split_indices[i]
        end = split_indices[i + 1]
        
        # Use numpy.take instead of split to avoid intermediate array
        sliced = numpy.take(data, range(start, end), axis=next_slice_dim - 1)
        
        mem_after_slice = _get_memory_usage_mb()
        trace_after_slice, trace_peak_after_slice = _get_tracemalloc_mb()
        
        slice_size_mb = _get_array_size_mb(sliced)
        total_scatter_size_mb += slice_size_mb
        to_scatter.append(sliced)
        
        log_once(
            f"reslice: Created slice {i} for rank {i}, shape={sliced.shape}, size={slice_size_mb:.2f} MB, "
            f"psutil delta: +{mem_after_slice - mem_before_slice:.2f} MB, "
            f"tracemalloc delta: +{trace_after_slice - trace_before_slice:.2f} MB, "
            f"tracemalloc peak: {trace_peak_after_slice:.2f} MB",
            level=logging.DEBUG,
        )

    mem_after_take = _get_memory_usage_mb()
    trace_after_take, trace_peak_after_take = _get_tracemalloc_mb()
    
    log_once(
        f"reslice: All slices created, total size={total_scatter_size_mb:.2f} MB",
        level=logging.DEBUG,
    )
    log_once(
        f"reslice: After take - psutil: {mem_after_take:.2f} MB (delta: +{mem_after_take - mem_before_take:.2f} MB), "
        f"tracemalloc: {trace_after_take:.2f} MB (delta: +{trace_after_take - trace_before_take:.2f} MB), "
        f"peak: {trace_peak_after_take:.2f} MB",
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

    start_idx = split_indices[rank]

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
        f"psutil={max(mem_after_take, mem_after_mpi):.2f} MB, "
        f"tracemalloc={trace_peak_end:.2f} MB",
        level=logging.DEBUG,
    )

    return new_data, next_slice_dim, start_idx