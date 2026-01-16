# SPICE Path Length Handling

Curryer provides automatic path shortening to handle SPICE tools' 80-character path limit, using a simple two-strategy approach that prioritizes zero-overhead symlinks before falling back to file copying.

## Background

SPICE tools (MSOPCK, MKSPK) enforce an **80-character per-line limit** for all string values in setup files, including file paths. This causes kernel generation to fail when:

- Kernels are located in Conda/pip environments (deep directory structures)
- System directories have long base paths
- Docker containers or AWS deployments have long mount paths

### Why SPICE's PATH_SYMBOLS Don't Always Work

`PATH_SYMBOLS` and `PATH_VALUES` are **only** interpreted by `furnsh_c` in meta-kernels (`.tm` files). MSOPCK and MKSPK setup files are text kernels but **not** meta-kernels; they require literal file paths limited to 80 characters.

---

## Automatic Path Shortening Overview

Curryer automatically shortens long paths using a **two-strategy approach**.

### Strategy 1: Symlink (Preferred and Always Tried First)

**How it works:** Creates a symbolic link in a short temp directory with simple naming: `curryer_{filename}`

**Advantages:**

- Zero file copying
- Zero storage overhead
- Fastest option
- No data duplication

**Platform Support:**

- **Works:** Linux, macOS, Unix, EC2, most containers
- **May fail:** Windows (requires admin/dev mode), restricted permissions containers.

**Automatic Fallback:** If symlink creation fails, curryer automatically tries the copy strategy

**Example:**

```
Original: /var/folders/3r/2r3w66hn4zdbtyfcw2d5b_ww00cdsj/T/kernels/naif0012.tls (85 chars)
Symlink:  /tmp/curryer_naif0012.tls (25 chars)
```

### Strategy 2: File Copy (Backup option)

**How it works:** Copies file to short temp directory using `tempfile.mkstemp()` for unique naming

**Advantage:**

- **Works consistently across platforms**

**Disadvantages:**

- I/O overhead (copying entire file)
- Doubles storage temporarily
- Potential Cloud costs if copying from network storage

### **File Cleanup:**

Temp files created during these strategies are **tracked** during creation. The calling code (e.g., `AbstractKernelWriter`) automatically deletes them in a `finally` block after kernel generation completes.

**Important:** Files created by `copy_to_short_path()` are NOT automatically cleaned up by Python's `tempfile` module. They must be manually deleted. Use Curryer's kernel writers to handle this automatically.

**Example:**

```
Original: /very/long/path/to/kernel.bsp (127 chars)
Copy:     /tmp/curryer_abc12345.bsp (25 chars)
```

---

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

**Default behavior (if env var not set):**

- **Unix/macOS:** Tries `/tmp` first (4 chars - maximum filename space!)
- **Windows:** Tries `C:\Temp` first (7 chars)
- **Fallback:** Uses `tempfile.gettempdir()` (with warning if >20 chars)

#### 2. `CURRYER_DISABLE_COPY` - Disable File Copying (AWS/Cloud)

Disable the file copy fallback to avoid storage costs in cloud environments:

```bash
# AWS/Cloud: Try symlinks only (avoid transfer)
export CURRYER_DISABLE_COPY="true"
```

**When to use:**

- AWS deployments with EFS or network storage
- Cloud environments with metered storage
- When you want to ensure zero file copying

**Important:** If symlinks fail and copy is disabled, paths will remain long and may cause errors.

### Configuration Examples

#### Default Behavior (Recommended)

```bash
# No configuration needed
# - Tries /tmp on Unix
# - Symlink first, copy fallback
# - Automatic cleanup
```

#### AWS/Cloud: Avoid File Copying Costs

```bash
# Disable copy to avoid costs
export CURRYER_DISABLE_COPY="true"

# Optional: Specify short custom temp directory
export CURRYER_TEMP_DIR="/tmp"
```

---

### Logging

Curryer logs path-shortening operations for transparency:

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

---

### How Cleanup Works

#### Using Kernel Writer Classes (Automatic Cleanup)

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

**Both strategies** create files in `/tmp` that need cleanup:

- **Symlinks** - Remove the symlink file (doesn't delete source)
- **Copies** - Remove the copied file (source remains)

**NOTE**: If you call `update_invalid_paths()` directly, **you must clean up manually**
