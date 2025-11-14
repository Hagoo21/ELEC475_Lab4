"""
Dataset Cleanup Script

This script removes downloaded images, captions, and cached embeddings.
Run this before re-downloading with different TRAIN_SAMPLES or VAL_SAMPLES settings.

Usage:
    python cleanup_dataset.py [options]

Options:
    --all           Delete everything including annotation cache
    --images-only   Delete only images and captions (keep embeddings)
    --cache-only    Delete only embeddings cache (keep images)
    --fiftyone      Also clean up FiftyOne database and cache
    --dry-run       Show what would be deleted without actually deleting
"""

import os
import shutil
import argparse
from pathlib import Path
import config

# Try to import FiftyOne to get its paths
try:
    import fiftyone as fo
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False


def get_dir_size(path):
    """Get total size of directory in MB."""
    if not os.path.exists(path):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB


def delete_directory(path, dry_run=False):
    """Delete a directory and return whether it existed."""
    if not os.path.exists(path):
        return False, 0
    
    size_mb = get_dir_size(path)
    
    if dry_run:
        print(f"  [DRY RUN] Would delete: {path} (~{size_mb:.1f} MB)")
        return True, size_mb
    else:
        print(f"  Deleting: {path} (~{size_mb:.1f} MB)")
        shutil.rmtree(path)
        print(f"  ✓ Deleted: {path}")
        return True, size_mb


def get_fiftyone_paths():
    """Get FiftyOne database and dataset cache paths."""
    paths = {}
    
    if not FIFTYONE_AVAILABLE:
        return paths
    
    try:
        # Try multiple ways to get FiftyOne paths (API varies by version)
        
        # Config directory (database)
        if hasattr(fo.config, 'config_dir'):
            paths['config'] = fo.config.config_dir
        elif hasattr(fo.config, 'default_config_dir'):
            paths['config'] = fo.config.default_config_dir
        else:
            # Fallback to default location
            from pathlib import Path
            paths['config'] = str(Path.home() / '.fiftyone')
        
        # Zoo datasets directory
        if hasattr(fo.config, 'dataset_zoo_dir'):
            paths['zoo'] = fo.config.dataset_zoo_dir
        elif hasattr(fo.config, 'default_dataset_dir'):
            paths['zoo'] = fo.config.default_dataset_dir
        else:
            # Fallback to default location
            from pathlib import Path
            paths['zoo'] = str(Path.home() / 'fiftyone')
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not get FiftyOne paths: {e}")
        # Try fallback paths
        from pathlib import Path
        paths['config'] = str(Path.home() / '.fiftyone')
        paths['zoo'] = str(Path.home() / 'fiftyone')
    
    return paths


def cleanup_fiftyone(dry_run=False):
    """Clean up FiftyOne database and cached datasets."""
    print("\n" + "="*70)
    print("CLEANUP: FIFTYONE DATABASE & CACHE")
    print("="*70)
    
    if not FIFTYONE_AVAILABLE:
        print("  ⚠ FiftyOne not installed or not available")
        return
    
    fiftyone_paths = get_fiftyone_paths()
    
    if not fiftyone_paths:
        print("  ⚠ Could not determine FiftyOne paths")
        return
    
    total_size = 0
    deleted_count = 0
    
    # Delete COCO-specific temporary datasets first
    if not dry_run:
        try:
            print("  Deleting FiftyOne temporary datasets...")
            for dataset_name in fo.list_datasets():
                if 'coco-2014' in dataset_name.lower() and 'temp' in dataset_name.lower():
                    print(f"    - Deleting dataset: {dataset_name}")
                    fo.delete_dataset(dataset_name)
        except Exception as e:
            print(f"  ⚠ Warning: Error deleting datasets: {e}")
    
    # Clean config directory (.fiftyone)
    if 'config' in fiftyone_paths:
        config_path = fiftyone_paths['config']
        print(f"\n  FiftyOne config/database: {config_path}")
        existed, size_mb = delete_directory(config_path, dry_run)
        if existed:
            deleted_count += 1
            total_size += size_mb
    
    # Clean zoo datasets directory
    if 'zoo' in fiftyone_paths:
        zoo_path = fiftyone_paths['zoo']
        print(f"\n  FiftyOne zoo datasets: {zoo_path}")
        existed, size_mb = delete_directory(zoo_path, dry_run)
        if existed:
            deleted_count += 1
            total_size += size_mb
    
    if deleted_count == 0:
        print("  ℹ No FiftyOne data found to delete")
    else:
        action = "Would free" if dry_run else "Freed"
        print(f"\n{'='*70}")
        print(f"✓ {action} approximately {total_size:.1f} MB from FiftyOne cache")
        print(f"{'='*70}")
    
    if not dry_run and deleted_count > 0:
        print("\n⚠ Note: FiftyOne will recreate these directories on next use")


def cleanup_all(dry_run=False):
    """Delete all dataset files including annotation cache."""
    print("\n" + "="*70)
    print("CLEANUP: ALL DATA (images, captions, embeddings, annotations)")
    print("="*70)
    
    total_size = 0
    deleted_count = 0
    
    dirs_to_delete = [
        config.DATA_DIR,              # coco_subset/
        config.CACHE_DIR,             # cache/
        config.ANNOTATIONS_CACHE_DIR, # coco_annotations_cache/
        config.EDA_OUTPUT_DIR         # eda_results/
    ]
    
    for dir_path in dirs_to_delete:
        existed, size_mb = delete_directory(dir_path, dry_run)
        if existed:
            deleted_count += 1
            total_size += size_mb
    
    if deleted_count == 0:
        print("  ℹ No data found to delete")
    else:
        action = "Would free" if dry_run else "Freed"
        print(f"\n{'='*70}")
        print(f"✓ {action} approximately {total_size:.1f} MB")
        print(f"{'='*70}")
    
    if not dry_run and deleted_count > 0:
        print("\n⚠ Note: You'll need to re-download annotation cache (~19MB) next time")


def cleanup_images_only(dry_run=False):
    """Delete only images and captions, keep embeddings and annotations."""
    print("\n" + "="*70)
    print("CLEANUP: IMAGES & CAPTIONS ONLY (keep embeddings & annotations)")
    print("="*70)
    
    total_size = 0
    existed, size_mb = delete_directory(config.DATA_DIR, dry_run)
    if existed:
        total_size += size_mb
        action = "Would free" if dry_run else "Freed"
        print(f"\n{'='*70}")
        print(f"✓ {action} approximately {total_size:.1f} MB")
        print(f"{'='*70}")
    else:
        print("  ℹ No image data found to delete")
    
    if not dry_run and existed:
        print("\n⚠ Note: Embeddings are still cached - delete cache/ if changing sample sizes")


def cleanup_cache_only(dry_run=False):
    """Delete only embedding cache, keep images and annotations."""
    print("\n" + "="*70)
    print("CLEANUP: EMBEDDING CACHE ONLY (keep images & annotations)")
    print("="*70)
    
    total_size = 0
    deleted_count = 0
    
    dirs_to_delete = [
        config.CACHE_DIR,      # cache/
        config.EDA_OUTPUT_DIR  # eda_results/
    ]
    
    for dir_path in dirs_to_delete:
        existed, size_mb = delete_directory(dir_path, dry_run)
        if existed:
            deleted_count += 1
            total_size += size_mb
    
    if deleted_count == 0:
        print("  ℹ No cache found to delete")
    else:
        action = "Would free" if dry_run else "Freed"
        print(f"\n{'='*70}")
        print(f"✓ {action} approximately {total_size:.1f} MB")
        print(f"{'='*70}")
    
    if not dry_run and deleted_count > 0:
        print("\n💡 Tip: Images are still there - embeddings will be regenerated on next run")


def cleanup_standard(dry_run=False):
    """Standard cleanup: Delete images and embeddings, keep annotation cache."""
    print("\n" + "="*70)
    print("CLEANUP: STANDARD (images, captions, embeddings)")
    print("="*70)
    print("⚠ Keeping annotation cache (~19MB) for reuse")
    
    total_size = 0
    deleted_count = 0
    
    dirs_to_delete = [
        config.DATA_DIR,       # coco_subset/
        config.CACHE_DIR,      # cache/
        config.EDA_OUTPUT_DIR  # eda_results/
    ]
    
    for dir_path in dirs_to_delete:
        existed, size_mb = delete_directory(dir_path, dry_run)
        if existed:
            deleted_count += 1
            total_size += size_mb
    
    if deleted_count == 0:
        print("  ℹ No data found to delete")
    else:
        action = "Would free" if dry_run else "Freed"
        print(f"\n{'='*70}")
        print(f"✓ {action} approximately {total_size:.1f} MB")
        print(f"{'='*70}")
    
    if not dry_run and deleted_count > 0:
        print("\n✅ Ready for fresh download! Run: python datasets/prepare_coco_clip_dataset.py")


def show_current_state():
    """Show what's currently on disk."""
    print("\n" + "="*70)
    print("CURRENT DATASET STATE")
    print("="*70)
    
    items = [
        ("Images & Captions", config.DATA_DIR),
        ("Embedding Cache", config.CACHE_DIR),
        ("Annotation Cache", config.ANNOTATIONS_CACHE_DIR),
        ("EDA Results", config.EDA_OUTPUT_DIR),
    ]
    
    total_size = 0
    for name, path in items:
        if os.path.exists(path):
            size_mb = get_dir_size(path)
            total_size += size_mb
            print(f"  ✓ {name:20s} {size_mb:8.1f} MB  ({path})")
        else:
            print(f"  ✗ {name:20s} {'---':>8s}     (not found)")
    
    # Show FiftyOne info
    print(f"\n  FiftyOne Cache:")
    if FIFTYONE_AVAILABLE:
        fiftyone_paths = get_fiftyone_paths()
        fiftyone_total = 0
        
        if 'config' in fiftyone_paths and os.path.exists(fiftyone_paths['config']):
            size_mb = get_dir_size(fiftyone_paths['config'])
            fiftyone_total += size_mb
            print(f"  ✓ {'  Config/Database':20s} {size_mb:8.1f} MB  ({fiftyone_paths['config']})")
        else:
            print(f"  ✗ {'  Config/Database':20s} {'---':>8s}")
        
        if 'zoo' in fiftyone_paths and os.path.exists(fiftyone_paths['zoo']):
            size_mb = get_dir_size(fiftyone_paths['zoo'])
            fiftyone_total += size_mb
            print(f"  ✓ {'  Zoo Datasets':20s} {size_mb:8.1f} MB  ({fiftyone_paths['zoo']})")
        else:
            print(f"  ✗ {'  Zoo Datasets':20s} {'---':>8s}")
        
        total_size += fiftyone_total
    else:
        print(f"  ⚠ FiftyOne not available")
    
    print(f"  {'─'*70}")
    print(f"  {'TOTAL:':20s} {total_size:8.1f} MB")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up COCO dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_dataset.py                    # Standard cleanup (keep annotations)
  python cleanup_dataset.py --all              # Delete everything
  python cleanup_dataset.py --images-only      # Keep embeddings
  python cleanup_dataset.py --cache-only       # Keep images
  python cleanup_dataset.py --fiftyone         # Also clean FiftyOne cache
  python cleanup_dataset.py --all --fiftyone   # Delete everything including FiftyOne
  python cleanup_dataset.py --dry-run          # Preview without deleting
  python cleanup_dataset.py --status           # Show current disk usage
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Delete everything including annotation cache')
    parser.add_argument('--images-only', action='store_true',
                        help='Delete only images and captions')
    parser.add_argument('--cache-only', action='store_true',
                        help='Delete only embedding cache')
    parser.add_argument('--fiftyone', action='store_true',
                        help='Also clean up FiftyOne database and cache')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without deleting')
    parser.add_argument('--status', action='store_true',
                        help='Show current dataset state and exit')
    
    args = parser.parse_args()
    
    # Show status and exit
    if args.status:
        show_current_state()
        return
    
    # Show current state first
    show_current_state()
    
    # Count how many cleanup options selected
    cleanup_options = sum([args.all, args.images_only, args.cache_only])
    
    if cleanup_options > 1:
        print("\n❌ Error: Please select only one cleanup option")
        print("   (--all, --images-only, or --cache-only)")
        return
    
    # Show warning for non-dry-run
    if not args.dry_run:
        print("\n⚠️  WARNING: This will delete files from disk!")
        response = input("   Continue? [y/N]: ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return
    
    # Perform cleanup based on selected option
    if args.all:
        cleanup_all(args.dry_run)
    elif args.images_only:
        cleanup_images_only(args.dry_run)
    elif args.cache_only:
        cleanup_cache_only(args.dry_run)
    else:
        # Default: standard cleanup
        cleanup_standard(args.dry_run)
    
    # Also clean FiftyOne if requested
    if args.fiftyone:
        cleanup_fiftyone(args.dry_run)
    
    if args.dry_run:
        print("\n💡 Tip: Remove --dry-run to actually delete files")


if __name__ == "__main__":
    main()

