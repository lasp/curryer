# SPICE Path Length Handling

Curryer provides automatic path shortening to handle SPICE tools' 80-character path limit, using a simple two-strategy approach that prioritizes zero-overhead symlinks before falling back to file copying.

## Overview

SPICE tools (MSOPCK, MKSPK) enforce an **80-character per-line limit** for all string values in setup files, including file paths. This causes kernel generation to fail when:

- Kernels are located in Conda/pip environments (deep directory structures)
- System temp directories have long base paths
- Docker containers use long mount paths
- AWS deployments use network storage with long paths

### Why SPICE's PATH_SYMBOLS Don't Always Work

`PATH_SYMBOLS` and `PATH_VALUES` are **only** interpreted by `furnsh_c` in meta-kernels (`.tm` files). MSOPCK and MKSPK setup files are text kernels but **not** meta-kernels—they require literal file paths and do not support symbolic substitution.

## Automatic Path Shortening

Curryer automatically shortens long paths using a **two-strategy approach**. Strategies are tried in a fixed order, and the first one that successfully shortens the path is used.

### Strategy 1: Symlink (Preferred - Always Tried First)

**How it works:** Creates a symbolic link in a short temp directory with simple naming: `curryer_{filename}`

**Advantages:**

- Zero file copying
- Zero storage overhead
- Fastest option
- No data duplication

**Platform Support:**

- **Works:** Linux, macOS, Unix, EC2, most containers
- **May fail:** Windows (requires admin/dev mode), restricted containers (seccomp policies)

**Automatic Fallback:** If symlink creation fails, curryer automatically tries the copy strategy

**Example:**

```
Original: /var/folders/3r/2r3w66hn4zdbtyfcw2d5b_ww00cdsj/T/kernels/naif0012.tls (85 chars)
Symlink:  /tmp/curryer_naif0012.tls (25 chars)
```

### Strategy 2: File Copy (Fallback option)

**How it works:** Copies file to short temp directory using `tempfile.mkstemp()` for unique naming

**Advantages:**

- **Works across platforms**
- No platform dependencies
- Handles any path length

**Disadvantages:**

- I/O overhead (must copy entire file)
- Doubles storage during operation
- Potential Cloud/AWS costs if copying from EFS/network storage

**Cleanup:** Temp files are **tracked** during creation. The calling code (e.g., `AbstractKernelWriter`) automatically deletes them in a `finally` block after kernel generation completes.

**Important:** Files created by `copy_to_short_path()` are NOT automatically cleaned up by Python's `tempfile` module. They must be manually deleted by the application. Curryer's kernel writers handle this automatically.

**Example:**

```
Original: /very/long/path/to/kernel.bsp (127 chars)
Copy:     /tmp/curryer_abc12345.bsp (25 chars)
```

## Configuration

Both strategies are enabled by default (symlink first, then copy as fallback). Configure behavior through environment variables.

### Environment Variables

Only **2 environment variables** are supported for simplicity:

#### 1. `CURRYER_TEMP_DIR` - Custom Temporary Directory

Set a custom short temp directory for maximum filename space:

```bash
# Use a very short custom path
export CURRYER_TEMP_DIR="/tmp"

# Or any other short path (must be ≤50 chars)
export CURRYER_TEMP_DIR="/opt/tmp"
```

**Validation:**

- Path must be ≤50 characters (raises `ValueError` if longer)
- Path will be created if it doesn't exist

**Default behavior (if not set):**

- **Unix/macOS:** Tries `/tmp` first (4 chars - maximum filename space!)
- **Windows:** Tries `C:\Temp` first (7 chars)
- **Fallback:** Uses `tempfile.gettempdir()` (with warning if >20 chars)

#### 2. `CURRYER_DISABLE_COPY` - Disable File Copying (AWS/Cloud)

Disable the file copy fallback to avoid storage costs in cloud environments:

```bash
# AWS/Cloud: Try symlinks only (avoid copying from EFS/network storage)
export CURRYER_DISABLE_COPY="true"
```

**When to use:**

- AWS deployments with EFS or network storage
- Cloud environments with metered storage
- When you want to ensure zero file copying

**Important:** If symlinks fail and copy is disabled, paths will remain long and may cause errors.

### Configuration Examples

#### Default Behavior (Recommended for Most Users)

```bash
# No configuration needed!
# - Tries /tmp on Unix (4 chars - 75 chars for filename)
# - Symlink first, copy fallback
# - Automatic cleanup
```

#### AWS/Cloud: Avoid File Copying Costs

```bash
# Disable copy to prevent EFS transfer costs
export CURRYER_DISABLE_COPY="true"

# Optional: Use short custom temp directory
export CURRYER_TEMP_DIR="/tmp"
```

#### Custom Short Temp Directory

```bash
# Use custom short path for all temp files
export CURRYER_TEMP_DIR="/data/tmp"
```

#### Extremely Long Temp Directory (macOS Users)

If you see warnings about long temp directories:

```bash
# Override macOS TMPDIR (which can be ~49 chars)
export CURRYER_TEMP_DIR="/tmp"
```

### Logging

Curryer logs path-shortening operations at INFO level for visibility:

**Successful symlink:**

```
INFO: Path exceeds 80 chars (102 chars): naif0012.tls
INFO:   → Using symlink: /tmp/curryer_naif0012.tls
```

**Fallback to copy:**

```
INFO: Path exceeds 80 chars (127 chars): large_kernel.bsp
DEBUG: Symlink creation failed: Operation not permitted
INFO:   → Using copy: /tmp/curryer_abc12345.bsp
```

**Both strategies fail:**

```
INFO: Path exceeds 80 chars (150 chars): very_long_file.txt
DEBUG: Symlink creation failed: Operation not permitted
DEBUG: Copy failed: Permission denied
WARNING: Failed to shorten path: very_long_file.txt (150 chars)
```

## Platform-Specific Behavior

**Recommended configurations:**

### **Tips:**

1. **Enable symlinks**

   - Symlinks work by default on Linux/macOS
   - Check logs to see if symlinks are failing

2. **Disable copy fallback** on AWS/cloud environments:

   ```bash
   export CURRYER_DISABLE_COPY="true"
   ```

3. **Use local storage** for `CURRYER_TEMP_DIR`:
   ```bash
   export CURRYER_TEMP_DIR="/tmp"  # Local SSD, not network storage
   ```

**Note:** Curryer automatically prefers `/tmp` when available, but logs a warning if falling back to longer paths.

## Temporary File Cleanup

### Important: Manual Cleanup Required

Temporary files created by the **copy strategy** are NOT automatically cleaned up by Python's `tempfile` module. This is because:

1. **We use `tempfile.mkstemp()`** - Creates the file but doesn't auto-delete it
2. **Files must persist** - They need to exist during SPICE tool execution
3. **Manual tracking** - The `update_invalid_paths()` function returns a list of temp files

### How Cleanup Works

#### When Using Kernel Writer Classes (Automatic)

If you're using Curryer's kernel writer classes, cleanup is **automatic**:

```python
from curryer.kernels.ephemeris import EphemerisKernel

# Cleanup happens automatically in finally block
kernel = EphemerisKernel(properties_dict)
kernel.write_kernel(output_path)  # Cleanup handled automatically
```

The `AbstractKernelWriter` class:

- Tracks temp files in `self._temp_kernel_files`
- Calls `_cleanup_temp_kernel_files()` in a `finally` block
- Deletes files even if kernel generation fails

#### When Using `update_invalid_paths()` Directly (Manual)

If you call `update_invalid_paths()` directly, **you must clean up manually**:

```python
from curryer.kernels.path_utils import update_invalid_paths
import os

temp_files = []
try:
    result, temp_files = update_invalid_paths(config, max_len=80)

    # Use the shortened paths...

finally:
    # REQUIRED: Clean up temp files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up: {temp_file}")
        except OSError as e:
            print(f"Warning: Could not clean up {temp_file}: {e}")
```

### What Gets Cleaned Up

| Strategy | Creates Temp Files?     | Cleanup Required?        |
| -------- | ----------------------- | ------------------------ |
| Symlink  | Yes (symlink, not data) | Yes (remove symlink)     |
| Copy     | Yes (full file copy)    | Yes (remove copied file) |

**Both strategies** create files in `/tmp` that need cleanup:

- **Symlinks** - Remove the symlink file (doesn't delete source)
- **Copies** - Remove the copied file (source remains)

### Cleanup Best Practices

1. **Always use `finally` blocks** - Ensures cleanup even if errors occur
2. **Check file exists** - File may already be deleted
3. **Catch OSError** - Don't let cleanup failures interrupt your program
4. **Log cleanup failures** - For debugging
5. **Use kernel writer classes** - They handle cleanup automatically

### What Happens If You Don't Clean Up?

Temp files accumulate in `/tmp` (or `CURRYER_TEMP_DIR`):

- **Disk space usage** - Each copied kernel file remains
- **System cleanup** - Most systems auto-clean `/tmp` on reboot
- **Not critical** - But wastes disk space during long-running processes

### Best Practices for Production Pipelines

1. **Test symlinks in your deployment environment:**

   ```bash
   # Quick test
   ln -s /path/to/source /tmp/test_symlink
   ```

2. **Use local storage** for `CURRYER_TEMP_DIR`:

   - Avoid network-mounted directories (NFS, EFS, SMB)
   - Use instance-local SSD when available

3. **Monitor logs** for copy operations:

   - Look for patterns of symlink failures
   - Address root causes (permissions, paths)

4. **For AWS/cloud deployments:**

   ```bash
   # Disable copy to avoid potential network transfer costs
   export CURRYER_DISABLE_COPY="true"
   ```

5. **Use shortest possible temp directory:**
   ```bash
   # Maximum filename space
   export CURRYER_TEMP_DIR="/tmp"  # 4 chars on Unix
   ```

## API Usage

For programmatic control, use the `update_invalid_paths()` function:

```python
from curryer.kernels.path_utils import update_invalid_paths
import os

# Basic usage with defaults
config = {"properties": {"kernel_path": "/very/long/path/to/kernel.bsp"}}

result, temp_files = update_invalid_paths(
    config,
    max_len=80,       # Maximum path length
    try_copy=True,    # Enable copy fallback (default: True)
    parent_dir=None,  # Parent directory for relative paths
    temp_dir=None,    # Custom temp directory (default: auto-detect)
)

# Use shortened path
shortened_path = result["properties"]["kernel_path"]

# ... use the shortened path for kernel generation ...

# IMPORTANT: Clean up temp files when done (required!)
for temp_file in temp_files:
    if os.path.exists(temp_file):
        os.remove(temp_file)
```

**Parameters:**

- `config` (dict): Configuration dictionary to update
- `max_len` (int): Maximum path length (default: 80 for SPICE)
- `try_copy` (bool): Enable copy strategy fallback (default: True)
- `parent_dir` (Path): Parent directory for resolving relative paths
- `temp_dir` (Path): Base directory for temp files (default: auto-detect)

**Returns:**

- `tuple`: (updated_config_dict, list_of_temp_files)
  - `updated_config_dict`: Config with shortened paths
  - `list_of_temp_files`: List of temporary file paths that need cleanup

**Cleanup Behavior:**

- Symlink strategy is always tried first (zero cost)
- The `try_copy` parameter controls whether the copy fallback is used if symlink fails
- **YOU MUST MANUALLY DELETE** the temp files returned in `list_of_temp_files`
- Temp files are NOT automatically cleaned by Python's `tempfile` module
- Use a `finally` block to ensure cleanup even if errors occur:

```python
temp_files = []
try:
    result, temp_files = update_invalid_paths(config, max_len=80)
    # ... use result ...
finally:
    # Always cleanup, even if errors occurred
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass  # Best effort cleanup
```

**Note:** When using Curryer's kernel writer classes (e.g., `AbstractKernelWriter`), cleanup is handled automatically via `_cleanup_temp_kernel_files()` in a `finally` block.

## See Also

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [SPICE Kernel Format Specification](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html)
- [Curryer API Reference](../api/kernels.rst)
