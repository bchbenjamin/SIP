#!/usr/bin/env python3
"""
Autonomous weapon detection training script for Raspberry Pi 5.
Designed to run as a systemd service — autostart on boot, crash recovery, Telegram notifications.
Maximizes CPU and memory usage.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

import yaml

HOME = os.path.expanduser("~")
VENV_PYTHON = os.path.join(HOME, "edge-ai", "pi", ".venv", "bin", "python3")
UNIFIED_DATASET = os.path.join(HOME, "edge-ai", "unified_weapon_dataset", "data.yaml")
FALLBACK_DATASET = os.path.join(HOME, "gun-and-knife-detection-1", "data.yaml")
CHECKPOINT = os.path.join(HOME, "runs", "detect", "train", "weights", "last.pt")
NOTIFY_SCRIPT = os.path.join(HOME, "edge-ai", "telegram_notify.py")
MANIFEST_PATH = os.path.join(HOME, "edge-ai", "datasets_manifest.json")

# Expected merged dataset configuration
EXPECTED_CLASSES = ["gun", "knife", "dangerous_animal"]
EXPECTED_NC = len(EXPECTED_CLASSES)

# Add edge-ai to path for telegram_notify
sys.path.insert(0, os.path.join(HOME, "edge-ai"))


def notify(cmd, *args):
    """Send a Telegram notification."""
    try:
        subprocess.run(
            [VENV_PYTHON, NOTIFY_SCRIPT, cmd] + list(args),
            timeout=30, check=False,
            cwd=HOME
        )
    except Exception:
        pass


def load_manifest():
    """Load and return the datasets_manifest.json, or None if missing."""
    if not os.path.exists(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def load_data_yaml(path):
    """Load and return a data.yaml, or None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def verify_dataset_manifest(manifest):
    """
    Verify the dataset manifest matches the expected merged dataset config.
    Returns (True, reason) if valid, (False, reason) if incompatible.
    """
    if manifest is None:
        return False, "No datasets_manifest.json found — run download_and_merge.py first"

    # Check classes match
    manifest_classes = manifest.get("classes", [])
    if manifest_classes != EXPECTED_CLASSES:
        return False, (
            f"Dataset manifest has wrong classes: {manifest_classes} "
            f"(expected {EXPECTED_CLASSES}). "
            "Run download_and_merge.py to regenerate the unified dataset."
        )

    # Check stats exist
    stats = manifest.get("stats", {})
    if not stats:
        return False, "Dataset manifest has no stats — dataset may be incomplete"

    total_images = sum(s.get("images", 0) for s in stats.values())
    if total_images < 100:
        return False, (
            f"Dataset manifest has only {total_images} images — insufficient for training. "
            "Wait for download_and_merge.py to complete."
        )

    return True, f"Manifest verified: {total_images} images across {list(stats.keys())}"


def verify_checkpoint_compatibility(checkpoint_path, expected_nc):
    """
    Verify that a YOLO checkpoint is compatible with the expected number of classes.
    YOLO checkpoints embed the class count in their metadata.
    Returns (True, metadata) if compatible, (False, reason) if not.
    """
    if not os.path.exists(checkpoint_path):
        return False, "No checkpoint found"

    try:
        # Load checkpoint metadata without loading full model
        from ultralytics import YOLO
        model = YOLO(checkpoint_path)
        ckpt_data = model.ckpt

        # Check embedded class count
        nc = ckpt_data.get("nc", None)
        if nc is None:
            # Try reading from model info
            return True, "Checkpoint class count not in metadata, assuming compatible"

        if nc != expected_nc:
            return False, (
                f"Checkpoint trained for {nc} classes but current dataset has {expected_nc}. "
                "Cannot resume — class counts must match. "
                "Remove {0} to start fresh with the new merged dataset.".format(checkpoint_path)
            )

        # Check model type matches
        model_name = ckpt_data.get("model_name", "")
        if model_name and "yolov8" not in model_name.lower():
            return False, f"Checkpoint is {model_name}, expected YOLOv8"

        # Verify class names if available
        names = ckpt_data.get("names", {})
        if names and len(names) == expected_nc:
            names_list = [names.get(i, "") for i in range(len(names))]
            if names_list != EXPECTED_CLASSES:
                return False, (
                    f"Checkpoint class names {names_list} don't match expected {EXPECTED_CLASSES}"
                )

        return True, f"Checkpoint compatible: {nc} classes, model={model_name}"

    except Exception as e:
        # If we can't read the checkpoint metadata, be conservative and warn but allow use
        return True, f"Could not verify checkpoint metadata ({e}), attempting to use it"


def verify_data_yaml(data_yaml_path, expected_nc):
    """
    Verify the data.yaml has the correct number of classes.
    Returns (True, reason) if valid, (False, reason) if incompatible.
    """
    data = load_data_yaml(data_yaml_path)
    if data is None:
        return False, f"data.yaml not found at {data_yaml_path}"

    nc = data.get("nc", 0)
    names = data.get("names", [])

    if nc != expected_nc:
        return False, (
            f"data.yaml has nc={nc} but expected {expected_nc}. "
            "The unified dataset may have been rebuilt with different classes. "
            "Remove {0} and run download_and_merge.py to regenerate.".format(data_yaml_path)
        )

    if names != EXPECTED_CLASSES:
        return False, (
            f"data.yaml class names {names} don't match expected {EXPECTED_CLASSES}"
        )

    return True, f"data.yaml verified: nc={nc}, names={names}"


def should_use_checkpoint(checkpoint_path, manifest_path, data_yaml_path):
    """
    Decide whether to resume from the existing checkpoint or start fresh.
    Returns (True, reason) to resume, (False, reason) to start fresh.
    """
    # 1. Verify dataset manifest
    manifest = load_manifest()
    manifest_ok, manifest_reason = verify_dataset_manifest(manifest)
    print(f"[verify] Manifest: {manifest_reason}")

    if not manifest_ok:
        return False, f"Manifest invalid — starting fresh: {manifest_reason}"

    # 2. Verify data.yaml matches expected classes
    yaml_ok, yaml_reason = verify_data_yaml(data_yaml_path, EXPECTED_NC)
    print(f"[verify] data.yaml: {yaml_reason}")

    if not yaml_ok:
        return False, f"data.yaml invalid — starting fresh: {yaml_reason}"

    # 3. Verify checkpoint compatibility
    if not os.path.exists(checkpoint_path):
        return False, "No checkpoint found — starting fresh"

    ckpt_ok, ckpt_reason = verify_checkpoint_compatibility(checkpoint_path, EXPECTED_NC)
    print(f"[verify] Checkpoint: {ckpt_reason}")

    if not ckpt_ok:
        # Move the incompatible checkpoint aside instead of deleting
        backup_path = checkpoint_path.replace(".pt", "_incompatible.pt")
        try:
            shutil.move(checkpoint_path, backup_path)
            print(f"[verify] Moved incompatible checkpoint to {backup_path}")
        except Exception:
            pass
        return False, f"Checkpoint incompatible — starting fresh: {ckpt_reason}"

    return True, f"Checkpoint verified and compatible"


def prepare_datasets():
    """Download and merge datasets if unified dataset doesn't exist."""
    if os.path.exists(UNIFIED_DATASET):
        # Verify the existing dataset is compatible before using
        yaml_ok, yaml_reason = verify_data_yaml(UNIFIED_DATASET, EXPECTED_NC)
        if yaml_ok:
            print(f"Unified dataset verified: {yaml_reason}")
            return UNIFIED_DATASET
        else:
            print(f"Existing data.yaml incompatible: {yaml_reason}")
            print("Re-running download_and_merge.py to regenerate...")

    print("Unified dataset not found or invalid. Running download_and_merge.py...")
    merge_script = os.path.join(HOME, "edge-ai", "download_and_merge.py")
    if os.path.exists(merge_script):
        try:
            subprocess.run(
                [VENV_PYTHON, merge_script],
                timeout=7200,  # 2 hour timeout
                check=True,
                cwd=HOME
            )
        except Exception as e:
            print(f"Merge script failed: {e}")

    if os.path.exists(UNIFIED_DATASET):
        return UNIFIED_DATASET
    elif os.path.exists(FALLBACK_DATASET):
        print(f"Falling back to original dataset: {FALLBACK_DATASET}")
        return FALLBACK_DATASET
    else:
        raise FileNotFoundError(
            "No dataset found! Run download_and_merge.py to create the unified dataset."
        )


def train():
    """Run YOLOv8 training with maximum resource usage."""
    from ultralytics import YOLO

    # Determine dataset
    data_yaml = prepare_datasets()
    print(f"Using dataset: {data_yaml}")

    # Verify checkpoint compatibility before deciding to resume
    resume_checkpoint, resume_reason = should_use_checkpoint(
        CHECKPOINT, MANIFEST_PATH, UNIFIED_DATASET
    )
    print(f"[train] Checkpoint decision: {resume_reason}")

    if resume_checkpoint:
        print(f"Resuming from checkpoint: {CHECKPOINT}")
        model = YOLO(CHECKPOINT)
        resume = True
    else:
        print("Starting fresh with yolov8n.pt")
        model = YOLO("yolov8n.pt")
        resume = False

    notify("start", f"Using dataset: {data_yaml}\nResume: {resume}")

    # Train with MAXIMUM resource usage
    # RPi5 has 4 cores and 4GB RAM
    try:
        model.train(
            data=data_yaml,
            epochs=150,
            batch=4,            # Doubled from 2 — uses more RAM
            imgsz=416,          # Larger images for better accuracy
            device="cpu",
            workers=0,          # Reduced from 4 for ARM CPU (0 to prevent crash)
            save_period=5,      
            project="runs/detect",
            name="train",
            exist_ok=True,
            resume=resume,
            patience=0,        
            cache=False,        # Disable cache to prevent OOM
            plots=True,        # Generate results.png
            verbose=True,
        )
        notify("complete")
        print("Training complete!")

    except AssertionError as e:
        if "nothing to resume" in str(e):
            print("Previous training run reached its epoch limit.")
            print("Backing up checkpoint and starting a new training run to continue training...")
            notify("start", "Starting a new training run from the finished checkpoint to reach higher epochs.")

            backup_ckpt = os.path.join(HOME, "last_backup.pt")
            shutil.copy(CHECKPOINT, backup_ckpt)

            # Re-verify manifest before starting a new run
            manifest = load_manifest()
            manifest_ok, manifest_reason = verify_dataset_manifest(manifest)
            if not manifest_ok:
                raise RuntimeError(
                    f"Cannot start extended training: {manifest_reason}"
                )

            model = YOLO(backup_ckpt)
            model.train(
                data=data_yaml,
                epochs=150,
                batch=4,
                imgsz=416,
                device="cpu",
                workers=0,
                save_period=5,
                project="runs/detect",
                name="train",
                exist_ok=True,
                resume=False,
                patience=0,
                cache=False,
                plots=True,
                verbose=True,
            )
            notify("complete", "New extended training run complete!")
            return
        raise

    except (RuntimeError, MemoryError, OSError) as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower() or isinstance(e, (MemoryError, OSError)):
            print(f"OOM or MemoryError with initial settings, falling back to batch=2/imgsz=320/workers=0...")
            notify("crash", f"OOM/MemoryError — retrying with smaller batch.\n{str(e)[:200]}")

            # Retry with smaller settings
            if os.path.exists(CHECKPOINT):
                model = YOLO(CHECKPOINT)
                resume = True
            else:
                model = YOLO("yolov8n.pt")
                resume = False

            try:
                model.train(
                    data=data_yaml,
                    epochs=150,
                    batch=2,
                    imgsz=320,
                    device="cpu",
                    workers=0,
                    save_period=5,
                    project="runs/detect",
                    name="train",
                    exist_ok=True,
                    resume=resume,
                    patience=0,
                    cache=False,
                    plots=True,
                    verbose=True,
                )
                notify("complete")
            except AssertionError as e:
                if "nothing to resume" in str(e):
                    print("Previous training run reached its epoch limit.")
                    print("Backing up checkpoint and starting a new fallback training run...")
                    notify("start", "Fallback: Starting a new training run from the finished checkpoint.")

                    backup_ckpt = os.path.join(HOME, "last_backup.pt")
                    if os.path.exists(CHECKPOINT):
                        shutil.copy(CHECKPOINT, backup_ckpt)
                        model = YOLO(backup_ckpt)
                    
                    model.train(
                        data=data_yaml,
                        epochs=150,
                        batch=2,
                        imgsz=320,
                        device="cpu",
                        workers=0,
                        save_period=5,
                        project="runs/detect",
                        name="train",
                        exist_ok=True,
                        resume=False,
                        patience=0,
                        cache=False,
                        plots=True,
                        verbose=True,
                    )
                    notify("complete", "New fallback training run complete!")
                    return
                raise
        else:
            raise


def main():
    try:
        train()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"FATAL: {e}\n{tb}", file=sys.stderr)
        notify("crash", tb[:500])
        sys.exit(1)


if __name__ == "__main__":
    main()
