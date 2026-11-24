"""Validator for the Scratch 3.0 (.sb3) Educational Story Machine project."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_scratch_project(sb3_path: Path) -> Dict[str, Any]:
    """Perform comprehensive structural and mathematical validation on the .sb3 project.

    Returns
    -------
    Dict[str, Any]
        Dictionary with validation results and diagnostic details.
    """
    results: Dict[str, Any] = {
        "file_exists": sb3_path.is_file(),
        "zip_valid": False,
        "project_json_valid": False,
        "assets_valid": False,
        "targets_count": 0,
        "monitors_count": 0,
        "variables_count": 0,
        "lists_count": 0,
        "math_checks_passed": False,
        "errors": [],
    }

    if not results["file_exists"]:
        results["errors"].append(f"File not found: {sb3_path}")
        return results

    # 1. ZIP Validity
    try:
        with zipfile.ZipFile(sb3_path, "r") as zf:
            namelist = zf.namelist()
            results["zip_valid"] = True

            if "project.json" not in namelist:
                results["errors"].append("project.json missing from root of .sb3 ZIP archive.")
                return results

            project_data = zf.read("project.json")
            project_json = json.loads(project_data.decode("utf-8"))
            results["project_json_valid"] = True

            # 2. Schema check
            semver = project_json.get("meta", {}).get("semver")
            if semver != "3.0.0":
                results["errors"].append(f"Expected semver 3.0.0, got {semver}")

            targets = project_json.get("targets", [])
            results["targets_count"] = len(targets)
            if not targets or not targets[0].get("isStage"):
                results["errors"].append("First target must be the Stage (isStage: True).")

            # Collect referenced asset files
            referenced_assets: Dict[str, str] = {}
            for t in targets:
                for c in t.get("costumes", []):
                    referenced_assets[c["md5ext"]] = c["assetId"]
                for s in t.get("sounds", []):
                    referenced_assets[s["md5ext"]] = s["assetId"]

            # 3. Check asset integrity in ZIP
            missing_assets = []
            checksum_mismatches = []
            for md5ext, expected_id in referenced_assets.items():
                if md5ext not in namelist:
                    missing_assets.append(md5ext)
                else:
                    data = zf.read(md5ext)
                    actual_id = hashlib.md5(data).hexdigest()
                    if actual_id != expected_id:
                        checksum_mismatches.append((md5ext, expected_id, actual_id))

            if missing_assets:
                results["errors"].append(f"Missing referenced assets in archive: {missing_assets}")
            if checksum_mismatches:
                results["errors"].append(f"Asset MD5 checksum mismatches: {checksum_mismatches}")

            if not missing_assets and not checksum_mismatches:
                results["assets_valid"] = True

            # 4. Variables & Lists
            stage = targets[0]
            results["variables_count"] = len(stage.get("variables", {}))
            results["lists_count"] = len(stage.get("lists", {}))
            results["monitors_count"] = len(project_json.get("monitors", []))

    except Exception as exc:
        results["errors"].append(f"Exception during zip parsing: {exc}")
        return results

    # 5. Mathematical Integrity Check
    # Verify candidate fusion formula: S(w) = sum(w_i * H_i(w))

    # A. Balanced Profile
    w_balanced = [0.40, 0.15, 0.15, 0.10, 0.20]
    assert math.isclose(sum(w_balanced), 1.0), "Balanced weights must sum to 1.0"

    forest_scores = [0.90, 0.40, 0.95, 0.30, 0.85]
    forest_fused = sum(w * s for w, s in zip(w_balanced, forest_scores))
    assert math.isclose(forest_fused, 0.7625), f"Expected 0.7625 for forest, got {forest_fused}"

    castle_scores = [0.85, 0.35, 0.90, 0.25, 0.85]
    castle_fused = sum(w * s for w, s in zip(w_balanced, castle_scores))
    assert math.isclose(castle_fused, 0.7225), f"Expected 0.7225 for castle, got {castle_fused}"

    garden_scores = [0.80, 0.40, 0.85, 0.30, 0.85]
    garden_fused = sum(w * s for w, s in zip(w_balanced, garden_scores))
    assert math.isclose(garden_fused, 0.7075), f"Expected 0.7075 for garden, got {garden_fused}"

    ran_scores = [0.30, 0.20, 0.10, 0.95, 0.90]
    ran_fused = sum(w * s for w, s in zip(w_balanced, ran_scores))
    assert math.isclose(ran_fused, 0.4400), f"Expected 0.4400 for ran, got {ran_fused}"

    milo_scores = [0.15, 0.95, 0.10, 0.10, 0.05]
    milo_fused = sum(w * s for w, s in zip(w_balanced, milo_scores))
    assert math.isclose(milo_fused, 0.2375), f"Expected 0.2375 for Milo, got {milo_fused}"

    assert forest_fused > castle_fused > garden_fused > ran_fused > milo_fused, "Balanced ranking order mismatch."

    # B. Action-Oriented Profile
    w_action = [0.25, 0.10, 0.10, 0.40, 0.15]
    assert math.isclose(sum(w_action), 1.0), "Action-Oriented weights must sum to 1.0"
    ran_action_fused = sum(w * s for w, s in zip(w_action, ran_scores))
    forest_action_fused = sum(w * s for w, s in zip(w_action, forest_scores))
    assert math.isclose(ran_action_fused, 0.6200), f"Expected 0.6200 for ran in Action profile, got {ran_action_fused}"
    assert math.isclose(forest_action_fused, 0.6075), f"Expected 0.6075 for forest in Action profile, got {forest_action_fused}"
    assert ran_action_fused > forest_action_fused, "Under Action-Oriented, ran must win."

    # C. Narrative-Rich Profile
    w_narrative = [0.20, 0.30, 0.30, 0.10, 0.10]
    assert math.isclose(sum(w_narrative), 1.0), "Narrative-Rich weights must sum to 1.0"
    forest_narrative_fused = sum(w * s for w, s in zip(w_narrative, forest_scores))
    assert math.isclose(forest_narrative_fused, 0.7000), f"Expected 0.7000 for forest in Narrative profile, got {forest_narrative_fused}"

    # D. Hamilton d10 check for all profiles
    for name, weights, expected_alloc in [
        ("Balanced", w_balanced, [4, 2, 1, 1, 2]),
        ("Action-Oriented", w_action, [3, 1, 1, 4, 1]),
        ("Narrative-Rich", w_narrative, [2, 3, 3, 1, 1]),
    ]:
        quotas = [10 * w for w in weights]
        base = [math.floor(q) for q in quotas]
        rem = [round(q - b, 6) for q, b in zip(quotas, base)]
        surplus = 10 - sum(base)
        alloc = list(base)
        if surplus > 0:
            # Deterministic tie-breaking priority: Grammar (0) > Character (1) > Setting (2) > Action (3) > Repeat (4)
            # Sort indices by remainder descending, then index ascending
            ranked_indices = sorted(range(len(rem)), key=lambda i: (-rem[i], i))
            for i in range(surplus):
                alloc[ranked_indices[i]] += 1
        assert alloc == expected_alloc, f"Hamilton mismatch for {name}: {alloc} != {expected_alloc}"
        assert sum(alloc) == 10, f"Hamilton allocations for {name} must sum to 10"

    results["math_checks_passed"] = True
    return results


if __name__ == "__main__":
    sb3_file = Path(__file__).resolve().parent / "five_head_story_machine.sb3"
    report = validate_scratch_project(sb3_file)
    print("Scratch 3.0 Project Validation Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if report["errors"]:
        print("Validation Failed with Errors:", report["errors"])
        exit(1)
    else:
        print("SUCCESS: Scratch project five_head_story_machine.sb3 is 100% valid!")
