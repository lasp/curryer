# SPICE Path Length Handling

## The Problem

SPICE tools (MSOPCK, MKSPK) enforce an **80-character per-line limit** for all string values in setup files, including file paths. This causes kernel generation to fail when:

- Kernels installed in Conda/pip environments (deep directory structures)
- System temp directories have long base paths
- Docker containers use long mount paths
- AWS deployments use network storage with long paths

### Why PATH_SYMBOLS Don't Work

`PATH_SYMBOLS` and `PATH_VALUES` are **only** interpreted by `furnsh_c` in meta-kernels (`.tm` files). MSOPCK and MKSPK setup files are text kernels but **not** meta-kernels—they require literal file paths and do not support symbolic substitution.

## Curryer's Solution

Curryer automatically shortens paths using a **priority-ordered strategy chain**:

### 1. Symlink (Preferred)
- **How**: Creates symlink in short temp dir (e.g., `/tmp/spice/abc123.tls`)
- **Pros**: Zero file copying, zero storage overhead, fast
- **Cons**: Platform-dependent
  - ✅ Works: Linux, macOS, most containers
  - ⚠️ May fail: Windows (requires admin/dev mode), restricted containers
- **Fallback**: Automatically tries next strategy if symlink creation fails

### 2. Continuation Character (`+`)
- **How**: Wraps path across multiple lines using `+` character
- **Pros**: No file operations, works if all path segments ≤80 chars
- **Cons**: Limited—fails if any single directory name exceeds 80 chars
- **Example**:
  ```
  '/very/long/path/+
  to/kernel.tls'
  ```

### 3. Relative Path
- **How**: Converts absolute path to relative (if shorter)
- **Pros**: No file operations
- **Cons**: Requires controlled working directory, limited applicability
- **Note**: Only used if result is <80 chars

### 4. File Copy (Bulletproof Fallback)
- **How**: Copies file to short temp directory
- **Pros**: Always works, guaranteed success
- **Cons**: 
  - I/O overhead (kernels can be many MB)
  - Doubles storage during operation
  - AWS costs if copying from EFS/network storage
- **Cleanup**: Temp files automatically deleted after kernel generation

## Configuration

### Environment Variables

Control path shortening behavior:

```bash
# Strategy priority order (comma-separated)
export CURRYER_PATH_STRATEGY="symlink,wrap,relative,copy"  # Default

# Disable symlinks (e.g., for Windows, restricted containers)
export CURRYER_DISABLE_SYMLINKS="false"  # Default

# Custom short temp directory
export CURRYER_TEMP_DIR="/tmp/spice"  # Default: /tmp (Unix), C:/Temp (Windows)

# Warn when copying large files
export CURRYER_WARN_ON_COPY="true"  # Default
```

### Logging

Curryer logs every path-shortening attempt:

```
INFO: Path exceeds 80 chars (125 chars): kernel.tls
DEBUG:   Full path: /very/long/conda/env/path/to/kernel.tls
INFO:   Attempting symlink strategy...
INFO:   ✓ Created symlink: /tmp/spice/kernel_abc123.tls (30 chars)
```

Or on failure:

```
WARNING:   ✗ Symlink creation failed (OSError: Operation not permitted)
INFO:   Attempting continuation character (+) wrapping...
WARNING:   ✗ Wrapping not possible: segment exceeds 80 chars
INFO:   Attempting file copy strategy...
INFO:   ✓ Copied to: /tmp/spice/kernel_xyz789.tls (32 chars)
```

## Platform-Specific Behavior

### Linux/macOS
- All strategies work
- Symlinks preferred (fastest, no overhead)

### Windows
- Symlinks require administrator privileges or Developer Mode
- If symlinks fail, automatically falls back to copy
- Use `CURRYER_DISABLE_SYMLINKS=true` to skip symlink attempts

### Docker/Containers
- Most containers support symlinks
- Some security policies (seccomp) may restrict symlink creation
- Curryer automatically falls back to copy if restricted

### AWS (EC2, Lambda, ECS)
- Symlinks work on EC2 instances
- File copy may have I/O costs if reading from EFS/network storage
- Use `CURRYER_TEMP_DIR` to control temp file location

## Troubleshooting

### "Path still exceeds 80 characters after shortening"

**Cause**: Temp directory itself is too long

**Solution**:
```bash
export CURRYER_TEMP_DIR="/tmp/k"  # Use very short base directory
```

### "Symlink creation failed"

**Cause**: Insufficient permissions or restricted environment

**Solution**: Curryer automatically falls back to file copying. To suppress symlink attempts:
```bash
export CURRYER_DISABLE_SYMLINKS="true"
```

### "Copying large files is slow"

**Cause**: File copy strategy used for large kernels

**Solutions**:
- Enable symlinks (if platform supports)
- Use faster local storage for `CURRYER_TEMP_DIR`
- Check logs to see why earlier strategies failed

## Performance Considerations

| Strategy | I/O Overhead | Storage Overhead | Speed |
|----------|--------------|------------------|-------|
| Symlink | None | None | Fastest |
| Wrap | None | None | Fast |
| Relative | None | None | Fast |
| Copy | High (2× file size read/write) | 2× file size | Slowest |

For production pipelines processing many/large kernels:
1. Ensure symlinks work (test in deployment environment)
2. Use local (not network) storage for `CURRYER_TEMP_DIR`
3. Monitor logs for unnecessary copy operations
