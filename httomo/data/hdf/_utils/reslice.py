import logging
from typing import Tuple
import psutil
import os

import numpy
from mpi4py.MPI import Comm

from httomo.data.mpiutil import alltoall
from httomo.data.hdf._utils import chunk
from httomo.utils import log_once


def _get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _get_array_size_mb(arr):
    """Get array size in MB."""
    return arr.nbytes / 1024 / 1024


def reslice(
    data: numpy.ndarray,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
) -> Tuple[numpy.ndarray, int, int]:
    """Reslice data by using in-memory MPI directives.

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
        "<-------Reslicing/rechunking the data-------->",
        level=logging.DEBUG,
    )

    rank = comm.rank
    mem_start = _get_memory_usage_mb()

    log_once(
        f"[Rank {rank}] reslice: Starting memory usage: {mem_start:.2f} MB (using original alltoall method)",
        level=logging.DEBUG,
    )

    # No need to reclice anything if there is only one process
    if comm.size == 1:
        log_once(
            f"[Rank {rank}] reslice: Not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size

    data_size_mb = _get_array_size_mb(data)
    log_once(
        f"[Rank {rank}] reslice: Input data shape={data.shape}, size={data_size_mb:.2f} MB, dtype={data.dtype}",
        level=logging.DEBUG,
    )

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(nprocs + 1)]

    log_once(
        f"[Rank {rank}] reslice: Splitting dimension {next_slice_dim} of length {length} into {nprocs} parts",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] reslice: Split indices: {split_indices}",
        level=logging.DEBUG,
    )

    # Prepare list for alltoall
    log_once(
        f"[Rank {rank}] reslice: Preparing {nprocs} slices to scatter",
        level=logging.DEBUG,
    )

    to_scatter = []
    total_scatter_size_mb = 0
    for i in range(nprocs):
        start = split_indices[i]
        end = split_indices[i + 1]
        # Use slicing instead of split to avoid intermediate array
        sliced = numpy.take(data, range(start, end), axis=next_slice_dim - 1)
        slice_size_mb = _get_array_size_mb(sliced)
        total_scatter_size_mb += slice_size_mb
        to_scatter.append(sliced)
        log_once(
            f"[Rank {rank}] reslice: Prepared slice {i} for rank {i}, shape={sliced.shape}, size={slice_size_mb:.2f} MB",
            level=logging.DEBUG,
        )

    mem_after_scatter = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: to_scatter list created, total size={total_scatter_size_mb:.2f} MB, "
        f"memory usage={mem_after_scatter:.2f} MB (delta: +{mem_after_scatter - mem_start:.2f} MB)",
        level=logging.DEBUG,
    )

    # Free original data if possible (Can we?)
    log_once(
        f"[Rank {rank}] reslice: Freeing original data array (size={data_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    del data

    mem_after_del = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: Original data freed, memory usage={mem_after_del:.2f} MB "
        f"(delta: {mem_after_del - mem_after_scatter:.2f} MB)",
        level=logging.DEBUG,
    )

    # All-to-all communication
    log_once(
        f"[Rank {rank}] reslice: Starting standard alltoall communication, "
        f"memory before MPI={mem_after_del:.2f} MB",
        level=logging.DEBUG,
    )

    mem_before_mpi = _get_memory_usage_mb()
    received = alltoall(to_scatter, comm)
    mem_after_mpi = _get_memory_usage_mb()

    received_size_mb = sum(_get_array_size_mb(arr) for arr in received)
    log_once(
        f"[Rank {rank}] reslice: alltoall completed, received {len(received)} arrays, "
        f"total size={received_size_mb:.2f} MB, "
        f"memory usage={mem_after_mpi:.2f} MB (delta: +{mem_after_mpi - mem_before_mpi:.2f} MB)",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] reslice: Received array shapes: {[arr.shape for arr in received]}",
        level=logging.DEBUG,
    )

    # Free scatter list
    log_once(
        f"[Rank {rank}] reslice: Freeing to_scatter list (size={total_scatter_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    del to_scatter

    mem_after_free_scatter = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: to_scatter freed, memory usage={mem_after_free_scatter:.2f} MB "
        f"(delta: {mem_after_free_scatter - mem_after_mpi:.2f} MB)",
        level=logging.DEBUG,
    )

    # Concatenate received chunks
    concat_axis = current_slice_dim - 1
    expected_shape = list(received[0].shape)
    expected_shape[concat_axis] = sum(arr.shape[concat_axis] for arr in received)
    expected_size_mb = (numpy.prod(expected_shape) * received[0].itemsize) / 1024 / 1024

    log_once(
        f"[Rank {rank}] reslice: Starting numpy.concatenate along axis {concat_axis}, "
        f"expected output shape={tuple(expected_shape)}, size={expected_size_mb:.2f} MB, "
        f"memory before concat={mem_after_free_scatter:.2f} MB",
        level=logging.DEBUG,
    )

    mem_before_concat = _get_memory_usage_mb()
    new_data = numpy.concatenate(received, axis=concat_axis)
    mem_after_concat = _get_memory_usage_mb()

    new_data_size_mb = _get_array_size_mb(new_data)
    log_once(
        f"[Rank {rank}] reslice: numpy.concatenate completed, output shape={new_data.shape}, "
        f"size={new_data_size_mb:.2f} MB, "
        f"memory usage={mem_after_concat:.2f} MB (delta: +{mem_after_concat - mem_before_concat:.2f} MB)",
        level=logging.DEBUG,
    )

    # Free received list
    log_once(
        f"[Rank {rank}] reslice: Freeing received list (size={received_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    del received

    mem_after_free_received = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: received list freed, memory usage={mem_after_free_received:.2f} MB "
        f"(delta: {mem_after_free_received - mem_after_concat:.2f} MB)",
        level=logging.DEBUG,
    )

    start_idx = split_indices[rank]

    mem_end = _get_memory_usage_mb()
    peak_mem = max(mem_after_scatter, mem_after_mpi, mem_after_concat)
    log_once(
        f"[Rank {rank}] reslice: Completed, returning data with slice_dim={next_slice_dim}, "
        f"start_idx={start_idx}, final memory={mem_end:.2f} MB "
        f"(total delta: {mem_end - mem_start:.2f} MB, peak memory: {peak_mem:.2f} MB, "
        f"peak increase: {peak_mem - mem_start:.2f} MB)",
        level=logging.DEBUG,
    )

    return new_data, next_slice_dim, start_idx