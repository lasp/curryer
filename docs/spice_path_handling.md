# SPICE Path Length Handling

Curryer provides automatic path shortening to handle SPICE tools' 80-character path limit, using a multi-strategy approach that prioritizes zero-overhead solutions before falling back to file copying.

## Overview

SPICE tools (MSOPCK, MKSPK) enforce an **80-character per-line limit** for all string values in setup files, including file paths. This causes kernel generation to fail when:

- Kernels are installed in Conda/pip environments (deep directory structures)
- System temp directories have long base paths
- Docker containers use long mount paths
- AWS deployments use network storage with long paths

### Why PATH_SYMBOLS Don't Work

`PATH_SYMBOLS` and `PATH_VALUES` are **only** interpreted by `furnsh_c` in meta-kernels (`.tm` files). MSOPCK and MKSPK setup files are text kernels but **not** meta-kernels—they require literal file paths and do not support symbolic substitution.

## Strategy Chain

Curryer automatically shortens paths using a **priority-ordered strategy chain**. Each strategy is tried in sequence, and the first one that successfully shortens the path is used. This ensures optimal performance while guaranteeing success.

### Strategy 1: Symlink (Preferred)

**How it works:** Creates a symbolic link in a short temp directory (e.g., `/tmp/spice/abc123.tls`)

**Advantages:**
- Zero file copying (no I/O overhead)
- Zero storage overhead
- Fastest option
- No data duplication

**Limitations:**
- Platform-dependent
- ✅ **Works:** Linux, macOS, most containers
- ⚠️ **May fail:** Windows (requires admin/dev mode), restricted containers (seccomp policies)

**Fallback:** Automatically tries next strategy if symlink creation fails

**Example:**
```
Original: /very/long/conda/env/lib/python3.10/site-packages/data/kernels/naif0012.tls (85 chars)
Symlink:  /tmp/spice/naif0012_a1b2c3d4.tls (33 chars)
```

### Strategy 2: Continuation Character (`+`)

**How it works:** Wraps path across multiple lines using `+` character

**Advantages:**
- No file operations
- Works if all path segments ≤ 79 chars
- Zero overhead

**Limitations:**
- Limited applicability—fails if any single directory name exceeds 79 chars
- Not all SPICE tools may support this (though most do)

**Example:**
```
Original: /very/long/path/to/many/nested/directories/kernel.tls

Wrapped:  '/very/long/path/to/many/+
          nested/directories/kernel.tls'
```

### Strategy 3: Relative Path

**How it works:** Converts absolute path to relative path (if shorter)

**Advantages:**
- No file operations
- Zero overhead

**Limitations:**
- Requires controlled working directory
- Limited applicability (only useful when relative path is significantly shorter)
- May not work across different mount points

**Note:** Only used if resulting path is < 80 chars

### Strategy 4: File Copy (Bulletproof Fallback)

**How it works:** Copies file to short temp directory

**Advantages:**
- **Always works** (guaranteed success)
- No platform dependencies
- Handles any path length

**Disadvantages:**
- I/O overhead (kernels can be many MB)
- Doubles storage during operation
- AWS costs if copying from EFS/network storage
- Warnings logged for files exceeding threshold (default: 10 MB)

**Cleanup:** Temp files automatically deleted after kernel generation

**Example:**
```
Original: /very/long/path/to/kernel.bsp (127 chars, 125 MB)
Copy:     /tmp/kernel_abc123.bsp (24 chars)
Warning:  ⚠ Copying large file: kernel.bsp (125.0 MB) - consider using symlinks
```

## Configuration

### Environment Variables

Control path shortening behavior through environment variables:

```bash
# Strategy priority order (comma-separated)
# Default: "symlink,wrap,relative,copy"
export CURRYER_PATH_STRATEGY="symlink,wrap,relative,copy"

# Disable symlinks (e.g., for Windows, restricted containers)
# Default behavior: symlinks are enabled when CURRYER_DISABLE_SYMLINKS is unset
export CURRYER_DISABLE_SYMLINKS="true"

# Custom short temp directory
# Default: /tmp (Unix), C:\Temp (Windows)
export CURRYER_TEMP_DIR="/tmp/spice"

# Warn when copying large files
# Default: true
export CURRYER_WARN_ON_COPY="true"

# Size threshold in MB to trigger warning
# Default: 10 MB
export CURRYER_WARN_COPY_THRESHOLD="10"
```

### Custom Strategy Order

You can customize which strategies are used and in what order:

```bash
# Only use non-I/O strategies (skip copy)
export CURRYER_PATH_STRATEGY="symlink,wrap,relative"

# Prioritize copy over symlinks (e.g., for debugging)
export CURRYER_PATH_STRATEGY="copy,symlink"

# Only use symlinks (fail if symlinks don't work)
export CURRYER_PATH_STRATEGY="symlink"
```

### Logging

Curryer logs every path-shortening attempt with structured output:

**Successful symlink:**
```
INFO: Path exceeds 80 chars (125 chars): kernel.tls
DEBUG:   Full path: /very/long/conda/env/path/to/kernel.tls
INFO:   Attempting symlink strategy...
INFO:   ✓ Created symlink: /tmp/spice/kernel_abc123.tls (30 chars)
```

**Strategy chain fallback:**
```
INFO: Path exceeds 80 chars (135 chars): large_kernel.bsp
INFO:   Attempting symlink strategy...
WARNING:   ✗ Symlink creation failed (OSError: Operation not permitted)
INFO:   Attempting continuation character (+) wrapping...
WARNING:   ✗ Wrapping not possible: segment exceeds 79 chars
INFO:   Attempting relative path optimization...
DEBUG:   ✗ Relative path still too long (120 chars)
INFO:   Attempting file copy strategy...
WARNING:   ⚠ Copying large file: large_kernel.bsp (125.5 MB) - consider using symlinks
INFO:   ✓ Copied to: /tmp/large_kern_xyz789.bsp (32 chars)
```

## Platform-Specific Behavior

### Linux/macOS

- **All strategies work**
- Symlinks preferred (fastest, no overhead)
- Default temp directory: `/tmp`

**Recommended configuration:**
```bash
# Use defaults (symlinks preferred)
# No configuration needed
```

### Windows

- **Symlinks require administrator privileges or Developer Mode**
- If symlinks fail, automatically falls back to copy
- Default temp directory: `C:\Temp`

**Recommended configuration:**
```bash
# Disable symlinks if not running as admin
export CURRYER_DISABLE_SYMLINKS="true"
export CURRYER_PATH_STRATEGY="wrap,relative,copy"
```

**Enabling Developer Mode for Symlinks:**
1. Settings → Update & Security → For Developers
2. Enable "Developer Mode"
3. Restart terminal/IDE

### Docker/Containers

- **Most containers support symlinks**
- Some security policies (seccomp) may restrict symlink creation
- Curryer automatically falls back to copy if restricted

**Recommended configuration:**
```bash
# Use short temp directory inside container
export CURRYER_TEMP_DIR="/tmp/k"

# If symlinks fail in your container
export CURRYER_DISABLE_SYMLINKS="true"
```

### AWS (EC2, Lambda, ECS)

- **Symlinks work on EC2 instances**
- File copy may have I/O costs if reading from EFS/network storage
- Consider symlinks to minimize data transfer costs

**Recommended configuration:**
```bash
# Use local temp directory (not EFS)
export CURRYER_TEMP_DIR="/tmp/spice"

# Set high threshold for large files
export CURRYER_WARN_COPY_THRESHOLD="100"  # 100 MB
```

## Troubleshooting

### "Path still exceeds 80 characters after shortening"

**Cause:** Temp directory itself is too long

**Solution:**
```bash
export CURRYER_TEMP_DIR="/tmp/k"  # Use very short base directory
```

### "Symlink creation failed"

**Cause:** Insufficient permissions or restricted environment

**Solutions:**

1. **Enable symlinks** (if platform supports):
   - Windows: Enable Developer Mode
   - Container: Check seccomp profile

2. **Disable symlinks** (use other strategies):
   ```bash
   export CURRYER_DISABLE_SYMLINKS="true"
   ```

### "Copying large files is slow"

**Cause:** File copy strategy used for large kernels

**Solutions:**

1. **Enable symlinks** (if platform supports):
   ```bash
   unset CURRYER_DISABLE_SYMLINKS  # or set to "false"
   ```

2. **Use faster local storage** for `CURRYER_TEMP_DIR`:
   ```bash
   export CURRYER_TEMP_DIR="/tmp"  # Local SSD, not network storage
   ```

3. **Check logs** to see why earlier strategies failed:
   - Look for "✗" markers in log output
   - Address root cause (e.g., enable symlinks, shorten paths)

### "Warnings about large file copies"

**Cause:** Files exceed `CURRYER_WARN_COPY_THRESHOLD`

**Solutions:**

1. **Use symlinks** (zero overhead):
   ```bash
   export CURRYER_DISABLE_SYMLINKS="false"
   export CURRYER_PATH_STRATEGY="symlink,wrap,relative,copy"
   ```

2. **Increase threshold** (if warnings are not helpful):
   ```bash
   export CURRYER_WARN_COPY_THRESHOLD="100"  # 100 MB
   ```

3. **Disable warnings** (not recommended):
   ```bash
   export CURRYER_WARN_ON_COPY="false"
   ```

## Performance Considerations

| Strategy | I/O Overhead | Storage Overhead | Speed | Success Rate |
|----------|--------------|------------------|-------|--------------|
| Symlink | None | None | Fastest | High (platform-dependent) |
| Wrap | None | None | Fast | Medium (limited by segment length) |
| Relative | None | None | Fast | Low (limited applicability) |
| Copy | High (2× file size) | 2× file size | Slowest | 100% (always works) |

### Best Practices for Production Pipelines

1. **Test symlinks in your deployment environment:**
   ```bash
   # Quick test
   ln -s /path/to/source /tmp/test_symlink
   ```

2. **Use local storage** for `CURRYER_TEMP_DIR`:
   - Avoid network-mounted directories (NFS, EFS, SMB)
   - Use instance-local SSD when available

3. **Monitor logs** for unnecessary copy operations:
   - Look for patterns of symlink failures
   - Address root causes (permissions, paths)

4. **Set appropriate warning thresholds:**
   ```bash
   # For large kernel processing
   export CURRYER_WARN_COPY_THRESHOLD="50"  # 50 MB
   ```

5. **Use custom strategy order** for specific needs:
   ```bash
   # Skip symlinks in restrictive environments
   export CURRYER_PATH_STRATEGY="wrap,relative,copy"
   ```

## API Usage

For programmatic control, use the `update_invalid_paths()` function:

```python
from curryer.kernels.writer import update_invalid_paths

# Custom configuration
config = {"properties": {"kernel_path": "/very/long/path/to/kernel.bsp"}}

result, temp_files = update_invalid_paths(
    config,
    max_len=80,
    try_symlink=True,   # Enable symlink strategy
    try_wrap=True,      # Enable wrapping strategy
    try_relative=True,  # Enable relative path strategy
    try_copy=True,      # Enable copy strategy (fallback)
)

# Use shortened path
shortened_path = result["properties"]["kernel_path"]

# Clean up temp files when done
for temp_file in temp_files:
    os.remove(temp_file)
```

## See Also

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [SPICE Kernel Format Specification](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html)
- [Curryer API Reference](../api/kernels.rst)
