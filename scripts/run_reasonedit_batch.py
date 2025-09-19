import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def list_task_ids(category_dir: Path) -> List[str]:
    ids = []
    for img_path in sorted(category_dir.glob("*.png")):
        stem = img_path.stem
        if stem.isdigit():
            mask_path = category_dir / f"{stem}_mask.jpg"
            if mask_path.exists():
                ids.append(stem)
    return ids


def sample_ids(ids: List[str], k: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if k <= 0 or k >= len(ids):
        return ids
    return sorted(rng.sample(ids, k))


def create_workspace_root(base_workspace: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = base_workspace / timestamp / "ReasonEdit"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def load_instructions(debug_root: Path) -> Dict[str, Dict[str, str]]:
    """Load instruction texts for categories. Keyed by category -> id(str) -> instruction."""
    mapping: Dict[str, Dict[str, str]] = {}
    files = {
        "3-Mirror": debug_root / "3-Mirror" / "Mirror_text.txt",
        "4-Color": debug_root / "4-Color" / "Color_text.txt",
    }
    for category, file_path in files.items():
        id2instr: Dict[str, str] = {}
        if file_path.exists():
            lines = file_path.read_text().splitlines()
            # Assume 1-indexed lines map to 001.. format
            for idx, line in enumerate(lines, start=1):
                key = f"{idx:03d}"
                id2instr[key] = line.strip()
        mapping[category] = id2instr
    return mapping


def prepare_task(category: str, task_id: str, source_dir: Path, dest_root: Path, instruction: Optional[str]) -> Path:
    dest_dir = dest_root / category / task_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_img = source_dir / f"{task_id}.png"
    src_mask = source_dir / f"{task_id}_mask.jpg"
    dst_img = dest_dir / src_img.name
    dst_mask = dest_dir / src_mask.name
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_mask, dst_mask)

    # simple manifest for traceability
    manifest = dest_dir / "manifest.txt"
    manifest.write_text(
        f"category: {category}\n"\
        f"task_id: {task_id}\n"\
        f"image: {dst_img}\n"\
        f"mask: {dst_mask}\n"\
        f"instruction: {instruction or ''}\n"
    )

    # Save instruction into a file for easy reading
    if instruction:
        (dest_dir / "instruction.txt").write_text(instruction)
    return dest_dir


def resolve_config_path(repo_root: Path, raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    p = Path(raw_path)
    # Absolute path
    if p.is_absolute() and p.exists():
        return p
    # As given (relative to CWD at call site)
    if p.exists():
        return p
    # Try relative to repo root
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return candidate
    # Handle optional leading NeoAnalogist/ prefix variations
    parts = list(p.parts)
    if parts and parts[0].lower() == "neoanalogist":
        candidate2 = (repo_root / Path(*parts[1:])).resolve()
        if candidate2.exists():
            return candidate2
    return None


def build_command(main_py: Path, work_dir: Path, config_path: Optional[Path], task_text: str | None, repo_root: Path) -> List[str]:
    cmd = [
        "python",
        str(main_py),
        "--work-dir", str(work_dir),
    ]
    if config_path and config_path.exists():
        # Use absolute path to avoid issues when working directory changes
        cmd.extend(["--config", str(config_path.resolve())])
    if task_text:
        cmd.extend(["--task", task_text])
    return cmd


def maybe_run_task(run: bool, cmd: List[str], cwd: Path) -> int:
    if not run:
        return 0
    import subprocess
    env = os.environ.copy()
    # ensure deterministic torch if desired, user can customize
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def main():
    # Resolve repository root as the parent of the scripts directory
    repo_root = Path(__file__).resolve().parents[1]

    # Compute sane, relative defaults
    default_debug_root = repo_root / "debug" / "ReasonEdit"
    default_workspace = repo_root / "workspace"
    default_config = repo_root / "config.json"
    default_main_py = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Batch prepare and run ReasonEdit tasks")
    parser.add_argument("--debug-root", type=str, default=str(default_debug_root),
                        help="Root directory of ReasonEdit dataset (contains 3-Mirror, 4-Color)")
    parser.add_argument("--workspace", type=str, default=str(default_workspace),
                        help="Base workspace directory where timestamped folder will be created")
    parser.add_argument("--num-per-category", type=int, default=1, help="Number of tasks to sample per category (default 1 for per-task runs)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--run", action="store_true", help="If set, invoke main.py for each task")
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Config file passed to main.py if --run is set")
    parser.add_argument("--task-template", type=str, default=None,
                        help="Optional task text passed to main.py --task. Supports {category}, {task_id}, {instruction} placeholders.")
    parser.add_argument("--task-from-instruction", action="store_true",
                        help="If set, use instruction text directly as --task (overridden by --task-template if given)")
    parser.add_argument("--separate-by-category", action="store_true",
                        help="Create a separate timestamped workspace for each category (if multiple tasks)")
    parser.add_argument("--grouped", action="store_true",
                        help="Group all selected tasks into a single timestamped workspace (override default per-task mode)")
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    workspace_base = Path(args.workspace)
    main_py = default_main_py
    resolved_config = resolve_config_path(repo_root, args.config) if args.config else None

    categories: List[Tuple[str, Path]] = [
        ("3-Mirror", debug_root / "3-Mirror"),
        ("4-Color", debug_root / "4-Color"),
    ]

    # Load instructions map once
    instr_map = load_instructions(debug_root)

    if not args.grouped:
        # Default: per-task workspaces (each task gets its own timestamped root)
        total = 0
        for category, cat_dir in categories:
            ids = list_task_ids(cat_dir)
            chosen = sample_ids(ids, args.num_per_category, args.seed)
            for task_id in chosen:
                dest_root = create_workspace_root(workspace_base)
                instruction = instr_map.get(category, {}).get(task_id)
                dest_dir = prepare_task(category, task_id, cat_dir, dest_root, instruction)

                # Optional run
                if args.run:
                    work_dir = dest_root.parent
                    task_text = None
                    if args.task_template:
                        task_text = args.task_template.format(category=category, task_id=task_id, instruction=instruction or "")
                    elif args.task_from_instruction and instruction:
                        # Include the actual image path in the task
                        timestamp = dest_root.parent.name  # Get the timestamp from the workspace directory
                        image_path = f"workspace/{timestamp}/ReasonEdit/{category}/{task_id}/{task_id}.png"
                        task_text = f"{instruction}\n\nImage path: {image_path}"
                    cmd = build_command(main_py, work_dir, resolved_config, task_text, repo_root)
                    code = maybe_run_task(True, cmd, cwd=work_dir)
                    (dest_dir / "run_exit_code.txt").write_text(str(code))

                # Write index per workspace
                index_path = dest_root / "index.tsv"
                with index_path.open("w") as f:
                    f.write("category\ttask_id\tdir\tinstruction\n")
                    f.write(f"{category}\t{task_id}\t{dest_dir}\t{(instruction or '').replace('\t',' ').replace('\n',' ')}\n")
                total += 1
        print(f"Prepared {total} task workspaces (per-task mode)")
    elif args.separate_by_category:
        total = 0
        for category, cat_dir in categories:
            dest_root = create_workspace_root(workspace_base)
            summary = []
            ids = list_task_ids(cat_dir)
            chosen = sample_ids(ids, args.num_per_category, args.seed)
            for task_id in chosen:
                instruction = instr_map.get(category, {}).get(task_id)
                dest_dir = prepare_task(category, task_id, cat_dir, dest_root, instruction)
                summary.append((category, task_id, dest_dir, instruction))

            if args.run:
                for category2, task_id2, dest_dir2, instruction2 in summary:
                    work_dir = dest_root.parent
                    task_text = None
                    if args.task_template:
                        task_text = args.task_template.format(category=category2, task_id=task_id2, instruction=instruction2 or "")
                    elif args.task_from_instruction and instruction2:
                        # Include the actual image path in the task
                        timestamp = dest_root.parent.name  # Get the timestamp from the workspace directory
                        image_path = f"workspace/{timestamp}/ReasonEdit/{category2}/{task_id2}/{task_id2}.png"
                        task_text = f"{instruction2}\n\nImage path: {image_path}"
                    cmd = build_command(main_py, work_dir, resolved_config, task_text, repo_root)
                    code = maybe_run_task(True, cmd, cwd=work_dir)
                    (dest_dir2 / "run_exit_code.txt").write_text(str(code))

            index_path = dest_root / "index.tsv"
            with index_path.open("w") as f:
                f.write("category\ttask_id\tdir\tinstruction\n")
                for category2, task_id2, dest_dir2, instruction2 in summary:
                    f.write(f"{category2}\t{task_id2}\t{dest_dir2}\t{(instruction2 or '').replace('\t',' ').replace('\n',' ')}\n")
            total += len(summary)
        print(f"Prepared {total} tasks under separate timestamped roots (one per category)")
    else:
        # Single timestamped workspace root holding both categories
        dest_root = create_workspace_root(workspace_base)
        summary = []
        for category, cat_dir in categories:
            ids = list_task_ids(cat_dir)
            chosen = sample_ids(ids, args.num_per_category, args.seed)
            for task_id in chosen:
                instruction = instr_map.get(category, {}).get(task_id)
                dest_dir = prepare_task(category, task_id, cat_dir, dest_root, instruction)
                summary.append((category, task_id, dest_dir, instruction))

        if args.run:
            for category, task_id, dest_dir, instruction in summary:
                work_dir = dest_root.parent
                task_text = None
                if args.task_template:
                    task_text = args.task_template.format(category=category, task_id=task_id, instruction=instruction or "")
                elif args.task_from_instruction and instruction:
                    # Include the actual image path in the task
                    timestamp = dest_root.parent.name  # Get the timestamp from the workspace directory
                    image_path = f"workspace/{timestamp}/ReasonEdit/{category}/{task_id}/{task_id}.png"
                    task_text = f"{instruction}\n\nImage path: {image_path}"
                cmd = build_command(main_py, work_dir, resolved_config, task_text)
                code = maybe_run_task(True, cmd, cwd=work_dir)
                (dest_dir / "run_exit_code.txt").write_text(str(code))

        index_path = dest_root / "index.tsv"
        with index_path.open("w") as f:
            f.write("category\ttask_id\tdir\tinstruction\n")
            for category, task_id, dest_dir, instruction in summary:
                f.write(f"{category}\t{task_id}\t{dest_dir}\t{(instruction or '').replace('\t',' ').replace('\n',' ')}\n")
        print(f"Prepared {len(summary)} tasks under {dest_root}")


if __name__ == "__main__":
    main()


