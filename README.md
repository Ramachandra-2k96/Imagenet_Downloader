# ImageNet Downloader - Usage Guide

Professional-grade multi-threaded ImageNet dataset downloader with robust error handling and connection pooling.

## Features

✅ **Multi-threaded parallel downloads** - Download multiple classes simultaneously  
✅ **Connection pooling** - Efficient HTTP connection reuse  
✅ **Automatic retry logic** - Handles transient network failures  
✅ **Smart skipping** - Avoids re-downloading existing data  
✅ **Thread-safe operations** - No race conditions or conflicts  
✅ **Progress tracking** - Real-time progress bars for each download  
✅ **Comprehensive logging** - Detailed logs and error tracking  
✅ **Auto tar cleanup** - Optional deletion of tar files after extraction  

---

## Installation

```bash
# Install dependencies
uv sync
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fname` | str | `data.csv` | CSV file with ImageNet class IDs (must have "wnid" column) |
| `--workers` | int | `10` | Number of parallel download threads |
| `--target` | str | `cwd` | Target directory for downloads (default: current directory) |
| `--delete-tar` | flag | `False` | Delete tar files after extraction |
| `--url` | str | ImageNet URL | Base URL for ImageNet downloads |

---

## Usage Examples

### 1. Basic Usage (with defaults)
```bash
uv run main.py
```
- Uses `data.csv` in current directory
- 10 parallel workers
- Downloads to current directory
- Keeps tar files

### 2. Custom CSV and Target Directory
```bash
uv run main.py --fname my_classes.csv --target /path/to/imagenet
```

### 3. Maximum Speed (20 workers + delete tar files)
```bash
uv run main.py --workers 20 --delete-tar --target /media/external/imagenet
```

### 4. Conservative Mode (fewer workers)
```bash
uv run main.py --workers 5
```

### 5. Production Setup
```bash
uv run main.py \
    --fname production_classes.csv \
    --workers 15 \
    --target /data/imagenet \
    --delete-tar
```

---

## CSV File Format

Your CSV file must contain a `wnid` column with ImageNet class IDs:

```csv
wnid,label
n02124075,Egyptian cat
n02352591,dog
n02084071,mammal
```

**Note:** The `label` column is optional and will be ignored. Only `wnid` is used.

---

## Output Structure

```
target_path/
├── n02124075/              # Extracted images for class 1
│   ├── image_001.JPEG
│   ├── image_002.JPEG
│   └── ...
├── n02352591/              # Extracted images for class 2
│   ├── image_001.JPEG
│   └── ...
├── failed_downloads.log    # Failed downloads log
└── imagenet_downloader.log # Detailed application log
```

---

## Performance Tips

### Optimal Worker Count
- **Fast internet (100+ Mbps)**: `--workers 15-20`
- **Medium internet (50-100 Mbps)**: `--workers 10-15`
- **Slow internet (<50 Mbps)**: `--workers 5-10`

### Disk Space Management
- Use `--delete-tar` to save disk space (tar files are deleted after extraction)
- Each class tar file is ~50-150MB, extracted ~50-200MB
- Plan accordingly for large datasets

### Network Optimization
- Connection pooling is automatically enabled
- Automatic retry on transient failures (429, 500, 502, 503, 504)
- Keep-alive connections reduce overhead

---

## Error Handling

### Failed Downloads
All failed downloads are logged to `failed_downloads.log`:
```
n02124075 - HTTPError: 404 Client Error
n02352591 - Timeout: Request timeout after 120s
```

### Retry Failed Downloads
To retry only failed downloads, create a new CSV from the log:
```bash
# Extract failed IDs
grep -o 'n[0-9]*' target_path/failed_downloads.log > failed_retry.csv
# Add header
sed -i '1iwnid' failed_retry.csv
# Retry
uv run main.py --fname failed_retry.csv --target /path/to/imagenet
```

---

## Monitoring

### Real-time Progress
Each worker shows its own progress bar:
```
[00] n02124075: 100%|████████| 45.2M/45.2M [00:23<00:00, 1.95MB/s]
[01] n02352591: 67%|██████▋  | 32.1M/48.0M [00:15<00:08, 2.13MB/s]
```

### Log Files
- `imagenet_downloader.log`: Detailed application logs
- `failed_downloads.log`: Failed download records

### Statistics Summary
After completion:
```
================================================================================
✓ Success: 95
⊘ Skipped: 3
✗ Failed:  2
Total:     100
================================================================================
```

---

## Troubleshooting

### Issue: "CSV file must contain 'wnid' column"
**Solution:** Ensure your CSV has a header row with `wnid` column.

### Issue: Downloads are slow
**Solution:** Increase `--workers` count or check your internet speed.

### Issue: Out of disk space
**Solution:** Use `--delete-tar` flag to auto-delete tar files.

### Issue: Connection timeouts
**Solution:** The script automatically retries. Check your network connection.

---

## Advanced: Programmatic Usage

```python
from main import download_batch, load_part_ids_from_csv

# Load IDs from CSV
part_ids = load_part_ids_from_csv('data.csv')

# Download with custom settings
stats = download_batch(
    part_ids=part_ids,
    target_path='/path/to/imagenet',
    num_workers=15,
    delete_tar=True
)

print(f"Success: {stats['success']}, Failed: {stats['failed']}")
```

---

## License

MIT License - Feel free to use in your projects!
