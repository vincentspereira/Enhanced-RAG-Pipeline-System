# Hardware Profiling Guide for RAG System

This guide provides recommendations and basic instructions for profiling the RAG system to identify performance bottlenecks related to CPU, GPU, memory, and I/O. Effective profiling is crucial for optimizing resource utilization and improving overall system performance.

## General Profiling Principles

1.  **Profile in a Production-Like Environment:** Whenever possible, profile on hardware that closely resembles your production setup.
2.  **Profile Realistic Workloads:** Use realistic data sizes, query complexities, and concurrency levels that match your expected usage patterns.
3.  **Isolate Components:** Profile individual components or operations first (e.g., embedding generation, a specific API endpoint, document processing) before attempting end-to-end profiling.
4.  **Iterate:** Profiling is an iterative process. Identify a bottleneck, optimize, and then re-profile to measure the impact and find the next bottleneck.
5.  **Baseline:** Always establish a baseline performance measurement before making changes.

## Python CPU Profilers

These tools help identify CPU-bound bottlenecks in your Python code.

### 1. `cProfile` (Built-in)

`cProfile` is a deterministic profiler built into Python. It provides detailed statistics on function calls, time spent in each function, and call counts.

**Usage:**

*   **Profiling a script:**
    ```bash
    python -m cProfile -o output.prof Scripts/your_script_to_profile.py [args]
    ```
*   **Analyzing results:** Use `pstats` module or visualizers like `snakeviz`.
    ```python
    # In Python interpreter
    import pstats
    from pstats import SortKey

    p = pstats.Stats('output.prof')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20) # Print top 20 cumulative time
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)     # Print top 20 self time
    ```
*   **Visualizing with `snakeviz`:**
    ```bash
    pip install snakeviz
    snakeviz output.prof
    ```
    This opens an interactive HTML view in your browser.

*   **Profiling specific functions within code:**
    ```python
    import cProfile, pstats, io

    def my_function_to_profile():
        # ... your code ...
        pass

    pr = cProfile.Profile()
    pr.enable()
    my_function_to_profile()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    ```

### 2. `Pyinstrument`

`Pyinstrument` is a statistical profiler that has lower overhead than `cProfile` and often gives a clearer picture of wall-clock time spent.

**Usage:**

*   **Installation:** `pip install pyinstrument`
*   **Profiling a script:**
    ```bash
    pyinstrument Scripts/your_script_to_profile.py [args]
    ```
    This will output an interactive HTML report or print to console.
*   **Profiling code blocks:**
    ```python
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    # ... code to profile ...

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    # or profiler.open_in_browser()
    ```

### 3. `Scalene`

Scalene is a high-performance CPU and memory profiler for Python that can also highlight parallelism issues.

**Usage:**

*   **Installation:** `pip install scalene`
*   **Profiling a script:**
    ```bash
    scalene Scripts/your_script_to_profile.py [args]
    ```
    Outputs an HTML report.

### 4. `line_profiler`

For line-by-line analysis of specific functions to see where time is spent within them.

**Usage:**

*   **Installation:** `pip install line_profiler`
*   **Decorate functions:**
    ```python
    # In your Python script (e.g., embedding_generator.py)
    # Add @profile decorator to functions you want to profile
    # This decorator will be available when running with kernprof.
    # If not running with kernprof, it will raise NameError unless handled.
    try:
        # This is for line_profiler
        profile # type: ignore
    except NameError:
        profile = lambda func: func # No-op decorator if not running line_profiler

    @profile
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # ... existing code ...
        pass
    ```
*   **Run with `kernprof`:**
    ```bash
    kernprof -l -v Scripts/your_script_to_profile.py [args]
    ```

## Python Memory Profilers

These tools help identify memory leaks or areas of high memory consumption.

### 1. `memory_profiler`

Provides line-by-line memory usage for functions.

**Usage:**

*   **Installation:** `pip install memory_profiler`
*   **Decorate functions:**
    ```python
    from memory_profiler import profile

    @profile
    def my_memory_intensive_function():
        # ... code ...
        pass
    ```
*   **Run script:**
    ```bash
    python -m memory_profiler Scripts/your_script_to_profile.py [args]
    ```

### 2. `Pympler`

Useful for tracking the size and number of Python objects in memory.

**Usage:**
Refer to Pympler documentation for detailed usage. It can help identify which types of objects are consuming the most memory.

## GPU Profiling (NVIDIA)

If your RAG system utilizes NVIDIA GPUs (e.g., for embeddings or local LLMs), these tools are essential.

### 1. PyTorch Profiler with TensorBoard

PyTorch has a built-in profiler that can trace CPU and GPU operations and visualize them in TensorBoard.

**Usage (within your PyTorch code, e.g., `EmbeddingGenerator`):**
```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# ... inside your function ...
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"): # Label a block of code
        # Your model inference code (e.g., self.model(**encoded_input))
        pass # Replace with actual code

# Print results to console
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to TensorBoard (after running your script)
# prof.export_chrome_trace("trace.json") # For Chrome tracing
# To view in TensorBoard, you might need to export in a format it supports or use its direct plugin.
# Check current PyTorch documentation for TensorBoard plugin usage with the profiler.
# Example:
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/profiler_experiment")
# prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
# writer.add_text('profiler_stacks', prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total"))
# writer.close()
# Then run: tensorboard --logdir=runs/profiler_experiment
```
Refer to the latest PyTorch Profiler documentation for detailed examples and TensorBoard integration.

### 2. NVIDIA Nsight Systems (`nsys`)

`Nsight Systems` is a system-wide performance analysis tool for deep learning applications. It helps visualize application algorithms, their interaction with the GPU, and system bottlenecks.

**Usage:**

*   **Profile an application:**
    ```bash
    nsys profile -o report_name python Scripts/your_gpu_script.py [args]
    ```
*   **View results:** Open `report_name.qdrep` file with the Nsight Systems UI.

### 3. NVIDIA Nsight Compute (`ncu`)

`Nsight Compute` is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and analysis for CUDA kernels.

**Usage:**

*   **Profile an application focusing on CUDA kernels:**
    ```bash
    ncu -o report_name python Scripts/your_gpu_script.py [args]
    ```
*   **View results:** Open `report_name.ncu-rep` file with the Nsight Compute UI.

## Profiling Specific Components of this RAG System

*   **Embedding Generation (`Scripts/embedding_generator.py`):**
    *   Use `cProfile`/`Pyinstrument` for Python-level bottlenecks.
    *   If GPU is used, use PyTorch Profiler or Nsight tools (`nsys`, `ncu`) to analyze CUDA kernel performance and GPU utilization.
    *   Use `memory_profiler` if high memory usage is suspected.
*   **LLM Inference (Local HuggingFace Models in `Scripts/llm/service.py`):**
    *   Similar to embedding generation, use Python CPU profilers and NVIDIA GPU profilers.
*   **API Endpoints (`Scripts/enhanced_api.py`):**
    *   Wrap endpoint functions (or parts of them) with `cProfile` or `Pyinstrument` decorators/context managers for targeted profiling under load.
    *   Use load testing tools (e.g., k6, Locust, Apache Bench) to generate traffic while profiling.
    *   Check logs for response times added by the middleware.
*   **Document Processing (`Scripts/document_processor.py`):**
    *   Profile the `process_directory` or individual file processing methods using `cProfile` or `Pyinstrument`.
    *   If I/O bound (reading many files), look at I/O wait times.
*   **Data Ingestion into Vector Store (`Scripts/rag_pipeline.py`):**
    *   Profile the `process_documents` method in `RAGPipeline`, paying attention to time spent in embedding generation vs. Qdrant client calls.

## Interpreting Results

*   **CPU Bound:** Look for functions with high "self time" (time spent in the function itself, not sub-calls) or high "cumulative time" (total time including sub-calls).
*   **I/O Bound:** If CPU utilization is low but the application is slow, it might be waiting for disk I/O or network I/O. Tools like `iotop` (Linux) or resource monitor can help. Python profilers might show time spent in I/O-related library calls.
*   **Memory Bound:** High memory usage, frequent garbage collection (can be seen with some profilers or `gc` module). Use memory profilers to find culprits.
*   **GPU Bound:** High GPU utilization. Use Nsight tools to see if kernels are efficient, or if there's too much CPU-GPU data transfer. Low GPU utilization when expected to be high indicates a CPU bottleneck feeding the GPU or inefficient GPU usage.

This guide provides a starting point. Effective profiling often requires a combination of these tools and an understanding of the system's architecture. Remember to consult the specific documentation for each profiling tool for advanced features and options.
