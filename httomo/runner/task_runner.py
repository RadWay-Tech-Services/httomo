from itertools import islice
import logging
import time
from typing import Any, Dict, Literal, Optional, List, Tuple, Union
import os

import tqdm
from mpi4py import MPI

import httomo.globals
from httomo.data.dataset_store import DataSetStoreWriter
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper
from httomo.runner.dataset_store_backing import determine_store_backing
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import (
    DataSetSink,
    DataSetSource,
    DummySink,
    ReadableDataSetSink,
)
from httomo.runner.gpu_utils import get_available_gpu_memory, gpumem_cleanup
from httomo.runner.monitoring_interface import MonitoringInterface
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import (
    Section,
    determine_minimum_block_length,
    determine_section_padding,
    sectionize,
)
from httomo.utils import (
    Pattern,
    _get_slicing_dim,
    catchtime,
    log_exception,
    log_once,
    log_rank,
)
import numpy as np

import psutil
import os

def _get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_size_recursive(obj, seen=None):
    """Recursively calculate the size of an object including referenced objects."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    
    # Handle different container types
    if isinstance(obj, dict):
        size += sum(get_size_recursive(k, seen) + get_size_recursive(v, seen) 
                   for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size_recursive(item, seen) for item in obj)
    
    return size


def list_all_variables_all_namespaces():
    """List ALL variables from ALL accessible namespaces with memory usage."""
    
    all_vars = {}
    
    log_once("=" * 100, level=logging.DEBUG)
    log_once("ALL VARIABLES IN ALL NAMESPACES", level=logging.DEBUG)
    log_once("=" * 100, level=logging.DEBUG)
    
    # 1. GLOBAL NAMESPACE
    log_once("\n1. GLOBAL NAMESPACE", level=logging.DEBUG)
    log_once("-" * 100, level=logging.DEBUG)
    
    global_vars = []
    for name, obj in globals().items():
        if not name.startswith('_') and not inspect.ismodule(obj) and not inspect.isfunction(obj) and not inspect.isclass(obj):
            try:
                shallow_size = sys.getsizeof(obj)
                deep_size = get_size_recursive(obj)
                global_vars.append({
                    'name': name,
                    'namespace': 'global',
                    'type': type(obj).__name__,
                    'shallow_size': shallow_size,
                    'deep_size': deep_size,
                    'obj': obj
                })
            except Exception as e:
                pass
    
    if global_vars:
        log_once(f"{'Variable':<25} {'Type':<15} {'Shallow Size':<15} {'Deep Size':<15}", level=logging.DEBUG)
        log_once("-" * 100, level=logging.DEBUG)
        for var in sorted(global_vars, key=lambda x: x['deep_size'], reverse=True):
            log_once(f"{var['name']:<25} {var['type']:<15} {var['shallow_size']:>12,} B  {var['deep_size']:>12,} B", level=logging.DEBUG)
        total = sum(v['deep_size'] for v in global_vars)
        log_once("-" * 100, level=logging.DEBUG)
        log_once(f"Total: {len(global_vars)} variables, {total:,} bytes (deep)", level=logging.DEBUG)
    else:
        log_once("No variables found in global namespace", level=logging.DEBUG)
    
    all_vars['global'] = global_vars
    
    # 2. LOCAL NAMESPACE
    log_once("\n2. LOCAL NAMESPACE", level=logging.DEBUG)
    log_once("-" * 100, level=logging.DEBUG)
    
    frame = inspect.currentframe()
    local_vars = []
    
    for name, obj in frame.f_locals.items():
        if not name.startswith('_') and name not in ['frame', 'all_vars', 'global_vars']:
            try:
                shallow_size = sys.getsizeof(obj)
                deep_size = get_size_recursive(obj)
                local_vars.append({
                    'name': name,
                    'namespace': 'local',
                    'type': type(obj).__name__,
                    'shallow_size': shallow_size,
                    'deep_size': deep_size,
                    'obj': obj
                })
            except Exception as e:
                pass
    
    if local_vars:
        log_once(f"{'Variable':<25} {'Type':<15} {'Shallow Size':<15} {'Deep Size':<15}", level=logging.DEBUG)
        log_once("-" * 100, level=logging.DEBUG)
        for var in sorted(local_vars, key=lambda x: x['deep_size'], reverse=True):
            log_once(f"{var['name']:<25} {var['type']:<15} {var['shallow_size']:>12,} B  {var['deep_size']:>12,} B", level=logging.DEBUG)
        total = sum(v['deep_size'] for v in local_vars)
        log_once("-" * 100, level=logging.DEBUG)
        log_once(f"Total: {len(local_vars)} variables, {total:,} bytes (deep)", level=logging.DEBUG)
    else:
        log_once("No variables found in local namespace", level=logging.DEBUG)
    
    all_vars['local'] = local_vars
    
    # 3. SCAN OUTER SCOPES
    log_once("\n3. OUTER SCOPES (if any)", level=logging.DEBUG)
    log_once("-" * 100, level=logging.DEBUG)
    
    outer_vars = []
    current_frame = frame.f_back
    scope_level = 1
    
    while current_frame is not None:
        for name, obj in current_frame.f_locals.items():
            if (not name.startswith('_') and 
                not inspect.ismodule(obj) and 
                not inspect.isfunction(obj) and 
                not inspect.isclass(obj)):
                try:
                    shallow_size = sys.getsizeof(obj)
                    deep_size = get_size_recursive(obj)
                    outer_vars.append({
                        'name': name,
                        'namespace': f'outer_scope_{scope_level}',
                        'type': type(obj).__name__,
                        'shallow_size': shallow_size,
                        'deep_size': deep_size,
                        'obj': obj
                    })
                except Exception as e:
                    pass
        current_frame = current_frame.f_back
        scope_level += 1
    
    if outer_vars:
        log_once(f"{'Variable':<25} {'Scope':<20} {'Type':<15} {'Deep Size':<15}", level=logging.DEBUG)
        log_once("-" * 100, level=logging.DEBUG)
        for var in sorted(outer_vars, key=lambda x: x['deep_size'], reverse=True):
            log_once(f"{var['name']:<25} {var['namespace']:<20} {var['type']:<15} {var['deep_size']:>12,} B", level=logging.DEBUG)
        total = sum(v['deep_size'] for v in outer_vars)
        log_once("-" * 100, level=logging.DEBUG)
        log_once(f"Total: {len(outer_vars)} variables, {total:,} bytes (deep)", level=logging.DEBUG)
    else:
        log_once("No outer scopes found", level=logging.DEBUG)
    
    all_vars['outer'] = outer_vars
    
    # 4. ALL OBJECTS IN MEMORY (using garbage collector)
    log_once("\n4. ALL OBJECTS IN MEMORY (via garbage collector)", level=logging.DEBUG)
    log_once("-" * 100, level=logging.DEBUG)
    log_once("Scanning all objects in memory... (this may take a moment)", level=logging.DEBUG)
    
    type_summary = {}
    gc_objects = gc.get_objects()
    
    for obj in gc_objects:
        obj_type = type(obj).__name__
        try:
            size = sys.getsizeof(obj)
            if obj_type not in type_summary:
                type_summary[obj_type] = {'count': 0, 'total_size': 0}
            type_summary[obj_type]['count'] += 1
            type_summary[obj_type]['total_size'] += size
        except:
            pass
    
    log_once(f"\n{'Type':<25} {'Count':<15} {'Total Size':<20}", level=logging.DEBUG)
    log_once("-" * 100, level=logging.DEBUG)
    
    sorted_types = sorted(type_summary.items(), 
                         key=lambda x: x[1]['total_size'], 
                         reverse=True)[:20]  # Top 20 types
    
    for obj_type, info in sorted_types:
        log_once(f"{obj_type:<25} {info['count']:>12,}   {info['total_size']:>15,} B", level=logging.DEBUG)
    
    log_once("-" * 100, level=logging.DEBUG)
    log_once(f"Total object types: {len(type_summary)}", level=logging.DEBUG)
    log_once(f"Total objects in memory: {len(gc_objects):,}", level=logging.DEBUG)
    
    # SUMMARY
    log_once("\n" + "=" * 100, level=logging.DEBUG)
    log_once("SUMMARY - ALL NAMESPACES COMBINED", level=logging.DEBUG)
    log_once("=" * 100, level=logging.DEBUG)
    
    combined = []
    for namespace, vars_list in all_vars.items():
        combined.extend(vars_list)
    
    # Remove duplicates (same object in multiple namespaces)
    unique_vars = {}
    for var in combined:
        obj_id = id(var['obj'])
        if obj_id not in unique_vars:
            unique_vars[obj_id] = var
    
    unique_list = list(unique_vars.values())
    unique_list.sort(key=lambda x: x['deep_size'], reverse=True)
    
    if unique_list:
        log_once(f"\n{'Variable':<25} {'Namespace':<15} {'Type':<15} {'Deep Size':<15}", level=logging.DEBUG)
        log_once("-" * 100, level=logging.DEBUG)
        for var in unique_list:
            log_once(f"{var['name']:<25} {var['namespace']:<15} {var['type']:<15} {var['deep_size']:>12,} B", level=logging.DEBUG)
        
        grand_total = sum(v['deep_size'] for v in unique_list)
        log_once("-" * 100, level=logging.DEBUG)
        log_once(f"Total unique variables: {len(unique_list)}", level=logging.DEBUG)
        log_once(f"Total memory (deep): {grand_total:,} bytes ({grand_total/1024/1024:.2f} MB)", level=logging.DEBUG)
    else:
        log_once("\nNo user-defined variables found in any namespace.", level=logging.DEBUG)
        log_once("\nTip: Create some variables first:", level=logging.DEBUG)
        log_once("  my_list = [1, 2, 3]", level=logging.DEBUG)
        log_once("  my_data = {'key': 'value'}", level=logging.DEBUG)
    
    log_once("=" * 100, level=logging.DEBUG)

class TaskRunner:
    """Handles the execution of a pipeline"""

    def __init__(
        self,
        pipeline: Pipeline,
        reslice_dir: os.PathLike,
        comm: MPI.Comm,
        memory_limit_bytes: int = 0,
        monitor: Optional[MonitoringInterface] = None,
    ):
        self.pipeline = pipeline
        self.reslice_dir = reslice_dir
        self.comm = comm
        self.monitor = monitor

        self.side_outputs: Dict[str, Any] = dict()
        self.source: Optional[DataSetSource] = None
        self.sink: Optional[Union[DataSetSink, ReadableDataSetSink]] = None

        self._memory_limit_bytes = memory_limit_bytes

        self._sections = self._sectionize()

    def execute(self) -> None:
        with catchtime() as t:

            self._prepare()
            for i, section in enumerate(self._sections):
                self._execute_section(section, i)
                gpumem_cleanup()

        self._log_pipeline(f"Pipeline finished. Took {t.elapsed:.3f}s")
        if self.monitor is not None:
            self.monitor.report_total_time(t.elapsed)

    def _sectionize(self) -> List[Section]:
        sections = sectionize(self.pipeline)
        self._log_pipeline(f"Pipeline has been separated into {len(sections)} sections")
        num_stores = len(sections) - 1
        if num_stores > 3:
            log_once(
                f"WARNING: Data saving or/and reslicing operation will be performed {num_stores} times. This increases the total run time, especially if the saving is performed through the disk.",
            )

        return sections

    def _execute_section(self, section: Section, section_index: int = 0):
        self._setup_source_sink(section, section_index)
        assert self.source is not None, "Dataset has not been loaded yet"
        assert self.sink is not None, "Sink setup failed"

        self._log_pipeline(
            f"Section {section_index} (pattern={section.methods[0].pattern.name}) with the following methods:",
            level=logging.INFO,
        )
        methods_info = [
            f"    {method.method_name} ({method.package_name})\n"
            for method in section.methods
        ]
        methods_info[-1] = methods_info[-1].rstrip("\n")
        self._log_pipeline(methods_info, level=logging.INFO)

        slicing_dim_section: Literal[0, 1] = _get_slicing_dim(section.pattern) - 1  # type: ignore
        self.determine_max_slices(section, slicing_dim_section)

        # Account for potential padding in number of max slices
        padding = determine_section_padding(section)
        section.max_slices -= padding[0] + padding[1]

        self._log_pipeline(
            f"Maximum amount of slices is {section.max_slices} for section {section_index}",
            level=logging.DEBUG,
        )

        self._pass_min_block_length_to_intermediate_data_wrapper(section)

        splitter = BlockSplitter(self.source, section.max_slices)
        start_source = time.perf_counter_ns()
        no_of_blocks = len(splitter)

        list_all_variables_all_namespaces()

        # Redirect tqdm progress bar output to /dev/null, and instead manually write block
        # processing progress to logfile within loop
        progress = tqdm.tqdm(
            iterable=splitter,
            file=open(os.devnull, "w"),
            unit="block",
            ascii=True,
        )
        for idx, block in enumerate(progress):

            mem_start = _get_memory_usage_mb()
            log_once(
                f"For cycle idx={idx}, block={block.global_index} memory usage start={mem_start:.2f} MB",
                level=logging.DEBUG,
            )

            end_source = time.perf_counter_ns()
            if self.monitor is not None:
                self.monitor.report_source_block(
                    f"sec_{section_index}",
                    section.methods[0].task_id if len(section) > 0 else "",
                    _get_slicing_dim(section.pattern) - 1,
                    block.shape,
                    block.chunk_index,
                    block.global_index,
                    (end_source - start_source) * 1e-9,
                )

            log_once(f"   {str(progress)}", level=logging.INFO)
            block = self._execute_section_block(section, block)
            log_rank(
                f"    Finished processing block {idx + 1} of {no_of_blocks}",
                comm=self.comm,
            )

            mem_end = _get_memory_usage_mb()
            log_once(
                f"For cycle idx={idx}, block={block} memory usage end={mem_end:.2f} MB",
                level=logging.DEBUG,
            )

            start_sink = time.perf_counter_ns()
            self.sink.write_block(block)
            end_sink = time.perf_counter_ns()
            if self.monitor is not None:
                self.monitor.report_sink_block(
                    f"sec_{section_index}",
                    section.methods[-1].task_id if len(section) > 0 else "",
                    _get_slicing_dim(section.pattern) - 1,
                    block.shape,
                    block.chunk_index,
                    block.global_index,
                    (end_sink - start_sink) * 1e-9,
                )

            # remove the reference pointing to the CuPy array before
            # calling the clean-up rountine
            del block.data
            gpumem_cleanup()

            start_source = time.perf_counter_ns()

        self._log_pipeline(
            "    Finished processing last block",
            level=logging.INFO,
        )

    def _setup_source_sink(self, section: Section, idx: int):
        assert self.source is not None, "Dataset has not been loaded yet"

        slicing_dim_section: Literal[0, 1] = _get_slicing_dim(section.pattern) - 1  # type: ignore

        if self.sink is not None:
            # we have a store-based sink from the last section - use that to determine
            # the source for this one
            assert isinstance(self.sink, ReadableDataSetSink)
            self.source = self.sink.make_reader(slicing_dim_section)

        store_backing = determine_store_backing(
            comm=self.comm,
            sections=self._sections,
            memory_limit_bytes=self._memory_limit_bytes,
            dtype=self.source.dtype,
            global_shape=self.source.global_shape,
            section_idx=idx,
        )

        if section.is_last:
            # we don't need to store the results - this sink just discards it
            self.sink = DummySink(slicing_dim_section)
        else:
            self.sink = DataSetStoreWriter(
                slicing_dim_section,
                self.comm,
                self.reslice_dir,
                store_backing=store_backing,
            )

    def _execute_section_block(
        self, section: Section, block: DataSetBlock
    ) -> DataSetBlock:
        for method in section:
            self.set_side_inputs(method)
            block = self._execute_method(method, block)
        return block

    def _log_pipeline(self, msg: Any, level: int = logging.INFO):
        log_once(msg, level=level)

    def _prepare(self):
        self._log_pipeline(
            f"See the full log file at: {httomo.globals.run_out_dir}/user.log",
        )
        self._check_params_for_sweep()
        self._load_datasets()

    def _load_datasets(self):
        start_time = self._log_task_start(
            "loader",
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
        )
        loader_padding = determine_section_padding(self._sections[0])
        self.source = self.pipeline.loader.make_data_source(padding=loader_padding)
        self._log_task_end(
            "loader",
            start_time,
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
            self.pipeline.loader.package_name,
        )

    def _execute_method(
        self, method: MethodWrapper, block: DataSetBlock
    ) -> DataSetBlock:
        start = time.perf_counter_ns()
        block = method.execute(block)
        if block.is_last_in_chunk:
            self.append_side_outputs(method.get_side_output())
        end = time.perf_counter_ns()
        if self.monitor is not None:
            self.monitor.report_method_block(
                method.method_name,
                method.module_path,
                method.task_id,
                _get_slicing_dim(method.pattern) - 1,
                block.shape,
                block.chunk_index,
                block.global_index,
                (end - start) * 1e-9,
                method.gpu_time.kernel,
                method.gpu_time.host2device,
                method.gpu_time.device2host,
            )
        return block

    def append_side_outputs(self, side_outputs: Dict[str, Any]):
        """Appends to the side outputs that are available for the next section(s)"""
        if len(side_outputs) == 0:
            return

        self.side_outputs |= side_outputs

    def set_side_inputs(self, method: MethodWrapper):
        """Sets the parameters that reference side outputs for this method."""
        for k, v in self.side_outputs.items():
            if k in method.parameters:
                method[k] = v

    def _log_task_start(self, id: str, pattern: Pattern, name: str) -> int:
        log_once(
            f"Running {id} (pattern={pattern.name}): {name}...",
            level=logging.INFO,
        )
        return time.perf_counter_ns()

    def _log_task_end(
        self,
        id: str,
        start_time: int,
        pattern: Pattern,
        name: str,
        package: str = "httomo",
    ):
        output_str_list = [
            f"    Finished {id}: {name} (",
            package,
            f") Took {float(time.perf_counter_ns() - start_time)*1e-6:.2f}ms",
        ]
        log_once(output_str_list)

    def _check_params_for_sweep(self):
        # Check pipeline for the number of parameter sweeps present. If one is
        # defined, raise an error, due to not supporting parameter sweeps in a
        # "performance" run of httomo
        params = [m.config_params for m in self.pipeline]
        no_of_sweeps = sum(map(self._count_tuple_values, params))
        if no_of_sweeps > 0:
            err_str = (
                f"There exists {no_of_sweeps} parameter sweep(s) in the "
                "pipeline, but parameter sweeps are not supported in "
                "`httomo performance`. Please either:\n  1) Remove the parameter "
                "sweeps.\n  2) Use `httomo preview` to run this pipeline."
            )
            log_exception(err_str)
            raise ValueError(err_str)

    def _count_tuple_values(self, d: Dict[str, Any]) -> int:
        return sum(1 for v in d.values() if isinstance(v, tuple))

    def determine_max_slices(self, section: Section, slicing_dim: int):
        assert self.source is not None
        assert len(section) > 0, "Section should contain at least 1 method"

        data_shape = self.source.chunk_shape
        max_slices = data_shape[slicing_dim]
        # loop over all methods in section
        has_gpu = False
        for idx, m in enumerate(section):
            if m.implementation in ["gpu", "gpu_cupy"] or m.is_gpu:
                has_gpu = True

        # if section consists of all cpu method then MAX_CPU_SLICES defines the block size
        if not has_gpu:
            section.max_slices = min(httomo.globals.MAX_CPU_SLICES, max_slices)
            return

        nsl_dim_l = list(data_shape)
        nsl_dim_l.pop(slicing_dim)
        non_slice_dims_shape = (nsl_dim_l[0], nsl_dim_l[1])

        available_memory = get_available_gpu_memory(10.0)
        available_memory_in_GB = round(available_memory / (1024**3), 2)
        memory_str = (
            f"The amount of the available GPU memory is {available_memory_in_GB} GB"
        )
        log_rank(memory_str, comm=self.comm)
        if self._memory_limit_bytes != 0:
            available_memory = min(available_memory, self._memory_limit_bytes)
            log_once(
                f"The memory has been limited to {available_memory / (1024**3):4.2f} GB",
                level=logging.DEBUG,
            )

        max_slices_methods = [max_slices] * len(section)

        SOURCE_DTYPE = np.dtype("float32")
        # NOTE: as the convertion of the raw data from uint16 to float32 happens after the data gets loaded,
        # we should consider self.source.dtype to be float for memory estimators.
        # see https://github.com/DiamondLightSource/httomo/issues/440

        # loop over all methods in section
        for idx, m in enumerate(section):
            if m.memory_gpu is None:
                max_slices_methods[idx] = max_slices
                continue

            output_dims = m.calculate_output_dims(non_slice_dims_shape)
            (slices_estimated, available_memory) = m.calculate_max_slices(
                SOURCE_DTYPE,  # self.source.dtype,
                non_slice_dims_shape,
                available_memory,
            )
            max_slices_methods[idx] = min(max_slices, slices_estimated)
            non_slice_dims_shape = output_dims

        section.max_slices = min(max_slices_methods)

    def _pass_min_block_length_to_intermediate_data_wrapper(self, section: Section):
        assert self.source is not None
        for method in section.methods:
            if isinstance(method, SaveIntermediateFilesWrapper):
                min_block_len = determine_minimum_block_length(
                    self.source.chunk_shape[self.source.slicing_dim], section.max_slices
                )
                method.append_config_params({"minimum_block_length": min_block_len})
