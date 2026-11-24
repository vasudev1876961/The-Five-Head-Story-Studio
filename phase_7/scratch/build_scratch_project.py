"""Scratch 3.0 Project Generator for The Five-Head Story Machine.

Generates:
1. High-quality vector SVG costumes for all 5 heads, UI buttons, cards, and stage backdrop.
2. Valid Scratch 3.0 project.json with complete block logic, variables, lists, and broadcasts.
3. Packages everything into a valid .sb3 ZIP archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


# --- SVG ASSET DEFINITIONS ---

STAGE_BACKDROP_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 360" width="480" height="360">
  <defs>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#161b22"/>
    </linearGradient>
    <linearGradient id="headerGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1f242d"/>
      <stop offset="100%" stop-color="#2d333b"/>
    </linearGradient>
    <linearGradient id="storyBoxGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1c2128"/>
      <stop offset="100%" stop-color="#161b22"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="480" height="360" fill="url(#bgGrad)"/>

  <!-- Subtle Grid Lines -->
  <g stroke="#30363d" stroke-width="0.5" opacity="0.3">
    <line x1="0" y1="60" x2="480" y2="60"/>
    <line x1="0" y1="180" x2="480" y2="180"/>
    <line x1="0" y1="280" x2="480" y2="280"/>
    <line x1="160" y1="0" x2="160" y2="360"/>
    <line x1="320" y1="0" x2="320" y2="360"/>
  </g>

  <!-- Header Banner -->
  <rect x="10" y="8" width="460" height="42" rx="8" fill="url(#headerGrad)" stroke="#388bfd" stroke-width="1.5"/>
  <text x="240" y="32" font-family="'Segoe UI', Arial, sans-serif" font-size="16" font-weight="bold" fill="#58a6ff" text-anchor="middle">
    THE FIVE-HEAD STORY MACHINE
  </text>
  <text x="240" y="44" font-family="'Segoe UI', Arial, sans-serif" font-size="8.5" fill="#8b949e" text-anchor="middle">
    Transparent Multi-Head NLP Story Generation &amp; Hamilton d10 Die Simulator
  </text>

  <!-- Five Heads Shelf Frame -->
  <rect x="10" y="56" width="460" height="84" rx="8" fill="#161b22" stroke="#30363d" stroke-width="1"/>
  <text x="20" y="70" font-family="'Segoe UI', Arial, sans-serif" font-size="9" font-weight="bold" fill="#8b949e">
    SCORING HEADS (CLICK A HEAD TO LEARN ITS RULE):
  </text>

  <!-- Story Display Card Area -->
  <rect x="10" y="278" width="460" height="74" rx="8" fill="url(#storyBoxGrad)" stroke="#388bfd" stroke-width="1.2"/>
  <text x="24" y="294" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" fill="#7ee787">
    STORY NARRATIVE OUTPUT:
  </text>
</svg>"""

HEAD_GRAMMAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <!-- Grammar Scout Owl -->
  <circle cx="40" cy="40" r="36" fill="#1f2937" stroke="#58a6ff" stroke-width="3"/>
  <path d="M 24 22 L 40 10 L 56 22 Z" fill="#58a6ff"/>
  <polygon points="40,6 20,18 40,24 60,18" fill="#388bfd"/>
  <rect x="38" y="22" width="4" height="12" fill="#ffd700"/>
  <!-- Eyes -->
  <circle cx="30" cy="38" r="10" fill="#ffffff" stroke="#58a6ff" stroke-width="2"/>
  <circle cx="50" cy="38" r="10" fill="#ffffff" stroke="#58a6ff" stroke-width="2"/>
  <circle cx="30" cy="38" r="4.5" fill="#0d1117"/>
  <circle cx="50" cy="38" r="4.5" fill="#0d1117"/>
  <!-- Glasses Bridge & Beak -->
  <line x1="38" y1="38" x2="42" y2="38" stroke="#58a6ff" stroke-width="2.5"/>
  <polygon points="40,43 36,49 44,49" fill="#ffab70"/>
  <!-- Label -->
  <rect x="8" y="60" width="64" height="15" rx="4" fill="#0d1117" stroke="#58a6ff" stroke-width="1"/>
  <text x="40" y="71" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#58a6ff" text-anchor="middle">H1: GRAMMAR</text>
</svg>"""

HEAD_CHARACTER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <!-- Character Keeper Guardian -->
  <circle cx="40" cy="40" r="36" fill="#1f2937" stroke="#f778ba" stroke-width="3"/>
  <!-- Guardian Face / Heart Shield -->
  <circle cx="40" cy="32" r="14" fill="#f778ba" opacity="0.3"/>
  <circle cx="40" cy="30" r="9" fill="#f778ba"/>
  <path d="M 22 52 C 22 41, 58 41, 58 52 Z" fill="#f778ba"/>
  <!-- Crown / Star -->
  <polygon points="40,14 43,20 49,20 44,24 46,30 40,26 34,30 36,24 31,20 37,20" fill="#ffd700"/>
  <!-- Label -->
  <rect x="6" y="60" width="68" height="15" rx="4" fill="#0d1117" stroke="#f778ba" stroke-width="1"/>
  <text x="40" y="71" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#f778ba" text-anchor="middle">H2: CHARACTER</text>
</svg>"""

HEAD_SETTING_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <!-- Setting Painter Artist -->
  <circle cx="40" cy="40" r="36" fill="#1f2937" stroke="#7ee787" stroke-width="3"/>
  <!-- Castle / Forest Landscape -->
  <polygon points="20,52 32,32 44,52" fill="#56d364"/>
  <polygon points="36,52 48,26 60,52" fill="#3ea54c"/>
  <!-- Sun / Moon -->
  <circle cx="28" cy="24" r="6" fill="#ffd700"/>
  <!-- Palette / Brush -->
  <circle cx="54" cy="44" r="5" fill="#7ee787" stroke="#1f2937" stroke-width="1.5"/>
  <circle cx="53" cy="43" r="1.5" fill="#ff7b72"/>
  <circle cx="56" cy="45" r="1.5" fill="#58a6ff"/>
  <!-- Label -->
  <rect x="8" y="60" width="64" height="15" rx="4" fill="#0d1117" stroke="#7ee787" stroke-width="1"/>
  <text x="40" y="71" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#7ee787" text-anchor="middle">H3: SETTING</text>
</svg>"""

HEAD_ACTION_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <!-- Action Booster Runner -->
  <circle cx="40" cy="40" r="36" fill="#1f2937" stroke="#ffa657" stroke-width="3"/>
  <!-- Lightning Bolt -->
  <polygon points="44,14 26,36 38,36 32,54 54,28 42,28" fill="#ffa657" stroke="#ff7b72" stroke-width="1.5"/>
  <!-- Speed lines -->
  <line x1="16" y1="28" x2="24" y2="28" stroke="#ffa657" stroke-width="2" stroke-linecap="round"/>
  <line x1="12" y1="36" x2="22" y2="36" stroke="#ffa657" stroke-width="2" stroke-linecap="round"/>
  <line x1="16" y1="44" x2="26" y2="44" stroke="#ffa657" stroke-width="2" stroke-linecap="round"/>
  <!-- Label -->
  <rect x="10" y="60" width="60" height="15" rx="4" fill="#0d1117" stroke="#ffa657" stroke-width="1"/>
  <text x="40" y="71" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#ffa657" text-anchor="middle">H4: ACTION</text>
</svg>"""

HEAD_REPEAT_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <!-- Repeat Blocker Shield -->
  <circle cx="40" cy="40" r="36" fill="#1f2937" stroke="#d29922" stroke-width="3"/>
  <!-- Shield with Anti-Loop Icon -->
  <path d="M 40 18 Q 54 22 54 36 Q 54 50 40 56 Q 26 50 26 36 Q 26 22 40 18 Z" fill="#d29922" stroke="#f2cc60" stroke-width="2"/>
  <!-- Loop crossed out -->
  <circle cx="40" cy="36" r="9" fill="none" stroke="#0d1117" stroke-width="2.5"/>
  <line x1="33" y1="43" x2="47" y2="29" stroke="#da3633" stroke-width="3" stroke-linecap="round"/>
  <!-- Label -->
  <rect x="8" y="60" width="64" height="15" rx="4" fill="#0d1117" stroke="#d29922" stroke-width="1"/>
  <text x="40" y="71" font-family="Arial, sans-serif" font-size="7.5" font-weight="bold" fill="#d29922" text-anchor="middle">H5: REPEAT</text>
</svg>"""

BTN_START_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40" width="120" height="40">
  <defs>
    <linearGradient id="btnGreen" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2ea043"/>
      <stop offset="100%" stop-color="#238636"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="116" height="36" rx="8" fill="url(#btnGreen)" stroke="#3fb950" stroke-width="2"/>
  <text x="60" y="24" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" fill="#ffffff" text-anchor="middle">
    ▶ NEXT WORD
  </text>
</svg>"""

BTN_PROFILE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40" width="120" height="40">
  <defs>
    <linearGradient id="btnBlue" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1f6feb"/>
      <stop offset="100%" stop-color="#1158c7"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="116" height="36" rx="8" fill="url(#btnBlue)" stroke="#58a6ff" stroke-width="2"/>
  <text x="60" y="24" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">
    ⚙ PROFILE
  </text>
</svg>"""

BTN_D10_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40" width="120" height="40">
  <defs>
    <linearGradient id="btnPurple" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8957e5"/>
      <stop offset="100%" stop-color="#6e40c9"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="116" height="36" rx="8" fill="url(#btnPurple)" stroke="#a371f7" stroke-width="2"/>
  <text x="60" y="24" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">
    🎲 HAMILTON d10
  </text>
</svg>"""

BTN_RESET_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 40" width="80" height="40">
  <rect x="2" y="2" width="76" height="36" rx="8" fill="#21262d" stroke="#8b949e" stroke-width="1.5"/>
  <text x="40" y="24" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" fill="#c9d1d9" text-anchor="middle">
    ↺ RESET
  </text>
</svg>"""

D10_DIE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 70" width="70" height="70">
  <!-- 10-Sided Die Kite Shape -->
  <polygon points="35,5 60,28 35,65 10,28" fill="#6e40c9" stroke="#d2a8ff" stroke-width="2.5"/>
  <polygon points="35,5 45,28 35,65" fill="#8957e5" opacity="0.6"/>
  <polygon points="35,5 25,28 35,65" fill="#553098" opacity="0.6"/>
  <circle cx="35" cy="32" r="14" fill="#0d1117" stroke="#ffd700" stroke-width="1.5"/>
  <text x="35" y="37" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#ffd700" text-anchor="middle">d10</text>
</svg>"""

STORY_DISPLAY_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <circle cx="50" cy="50" r="44" fill="#161b22" stroke="#58a6ff" stroke-width="2"/>
  <path d="M 30 35 L 70 35 M 30 50 L 70 50 M 30 65 L 55 65" stroke="#7ee787" stroke-width="4" stroke-linecap="round"/>
</svg>"""


# --- BUILD SCRATCH 3.0 PROJECT JSON ---

def build_scratch_project_json(asset_map: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    """Construct full Scratch 3.0 project JSON adhering to Scratch 3.0 specifications.

    Parameters
    ----------
    asset_map : Dict[str, Tuple[str, str]]
        Mapping of asset friendly name to (assetId, md5ext).
    """

    # Stage costumes
    stage_costume = {
        "assetId": asset_map["backdrop"][0],
        "name": "StoryStudioBackdrop",
        "md5ext": asset_map["backdrop"][1],
        "dataFormat": "svg",
        "rotationCenterX": 240,
        "rotationCenterY": 180,
    }

    # Sprites costumes
    def make_costume(key: str, name: str, cx: int, cy: int) -> Dict[str, Any]:
        return {
            "assetId": asset_map[key][0],
            "name": name,
            "md5ext": asset_map[key][1],
            "dataFormat": "svg",
            "rotationCenterX": cx,
            "rotationCenterY": cy,
        }

    # Define Global Variables (Stage level)
    # [name, initial_value]
    global_vars = {
        "var_story": ["Current Story", "Milo entered the"],
        "var_context": ["Current Context", "Milo entered the"],
        "var_active_cand": ["Candidate Word", "forest"],
        "var_selected_word": ["Selected Word", "forest"],
        "var_step": ["Story Step", 1],
        "var_score_g": ["Grammar Score", 0.90],
        "var_score_c": ["Character Score", 0.40],
        "var_score_s": ["Setting Score", 0.95],
        "var_score_a": ["Action Score", 0.30],
        "var_score_r": ["Repeat Score", 0.85],
        "var_w_g": ["Weight Grammar", 0.40],
        "var_w_c": ["Weight Character", 0.15],
        "var_w_s": ["Weight Setting", 0.15],
        "var_w_a": ["Weight Action", 0.10],
        "var_w_r": ["Weight Repeat", 0.20],
        "var_fused": ["Fused Score", 0.7625],
        "var_profile": ["Active Profile", "Balanced"],
        "var_d10_g": ["d10 Grammar", 4],
        "var_d10_c": ["d10 Character", 2],
        "var_d10_s": ["d10 Setting", 1],
        "var_d10_a": ["d10 Action", 1],
        "var_d10_r": ["d10 Repeat", 2],
        "var_message": ["Explanation", "Grammar & Setting favored 'forest' (0.7625)!"],
    }

    # Global Lists
    global_lists = {
        "list_cands": ["Candidate Words", ["forest", "castle", "garden", "ran", "milo"]],
        "list_scores": ["Fused Scores", ["0.7625 (forest)", "0.7225 (castle)", "0.7075 (garden)", "0.4400 (ran)", "0.2375 (milo)"]],
        "list_story": ["Completed Story Words", ["Milo", "entered", "the"]],
        "list_d10": ["Hamilton d10 Faces", [
            "1: Grammar Scout", "2: Grammar Scout", "3: Grammar Scout", "4: Grammar Scout",
            "5: Character Keeper", "6: Character Keeper", "7: Setting Painter",
            "8: Action Booster", "9: Repeat Blocker", "10: Repeat Blocker"
        ]],
    }

    # Broadcasts
    broadcasts = {
        "bc_evaluate": "evaluate_next_word",
        "bc_select": "word_selected",
        "bc_profile": "cycle_profile",
        "bc_d10": "calculate_hamilton",
        "bc_reset": "reset_story",
        "bc_explain": "explain_heads",
    }

    # 1. STAGE TARGET
    stage = {
        "isStage": True,
        "name": "Stage",
        "variables": {k: v for k, v in global_vars.items()},
        "lists": {k: v for k, v in global_lists.items()},
        "broadcasts": {k: v for k, v in broadcasts.items()},
        "blocks": {},
        "comments": {},
        "currentCostume": 0,
        "costumes": [stage_costume],
        "sounds": [],
        "volume": 100,
        "layerOrder": 0,
    }

    targets = [stage]

    # Helper to construct sprite with event blocks
    def create_sprite(
        name: str,
        costumes: List[Dict[str, Any]],
        x: int,
        y: int,
        size: int,
        blocks: Dict[str, Any],
        layer: int,
    ) -> Dict[str, Any]:
        return {
            "isStage": False,
            "name": name,
            "variables": {},
            "lists": {},
            "broadcasts": {},
            "blocks": blocks,
            "comments": {},
            "currentCostume": 0,
            "costumes": costumes,
            "sounds": [],
            "volume": 100,
            "visible": True,
            "x": x,
            "y": y,
            "size": size,
            "direction": 90,
            "draggable": False,
            "rotationStyle": "all around",
            "layerOrder": layer,
        }

    # 2. HEAD SPRITES (H1 to H5)
    # H1 - Grammar Scout
    h1_blocks = {
        "b_h1_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_h1_say",
            "parent": None,
            "topLevel": True,
        },
        "b_h1_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_h1_click",
            "inputs": {
                "MESSAGE": [1, [10, "Owl: I am Grammar Scout! I evaluate language plausibility from N-gram frequency patterns."]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_h1 = create_sprite(
        "Grammar Scout", [make_costume("h1", "Owl", 40, 40)], -170, 100, 90, h1_blocks, 1
    )
    targets.append(s_h1)

    # H2 - Character Keeper
    h2_blocks = {
        "b_h2_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_h2_say",
            "parent": None,
            "topLevel": True,
        },
        "b_h2_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_h2_click",
            "inputs": {
                "MESSAGE": [1, [10, "Guardian: I am Character Keeper! I track Milo, Pip, and Zara to maintain character focus."]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_h2 = create_sprite(
        "Character Keeper", [make_costume("h2", "Guardian", 40, 40)], -85, 100, 90, h2_blocks, 2
    )
    targets.append(s_h2)

    # H3 - Setting Painter
    h3_blocks = {
        "b_h3_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_h3_say",
            "parent": None,
            "topLevel": True,
        },
        "b_h3_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_h3_click",
            "inputs": {
                "MESSAGE": [1, [10, "Artist: I am Setting Painter! I anchor the story in Castle, Cave, Forest, or Garden."]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_h3 = create_sprite(
        "Setting Painter", [make_costume("h3", "Artist", 40, 40)], 0, 100, 90, h3_blocks, 3
    )
    targets.append(s_h3)

    # H4 - Action Booster
    h4_blocks = {
        "b_h4_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_h4_say",
            "parent": None,
            "topLevel": True,
        },
        "b_h4_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_h4_click",
            "inputs": {
                "MESSAGE": [1, [10, "Runner: I am Action Booster! I inject energy with dynamic action verbs like flew, ran, and found!"]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_h4 = create_sprite(
        "Action Booster", [make_costume("h4", "Runner", 40, 40)], 85, 100, 90, h4_blocks, 4
    )
    targets.append(s_h4)

    # H5 - Repeat Blocker
    h5_blocks = {
        "b_h5_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_h5_say",
            "parent": None,
            "topLevel": True,
        },
        "b_h5_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_h5_click",
            "inputs": {
                "MESSAGE": [1, [10, "Shield: I am Repeat Blocker! I penalize duplicate words and loops so the story keeps progressing."]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_h5 = create_sprite(
        "Repeat Blocker", [make_costume("h5", "Shield", 40, 40)], 170, 100, 90, h5_blocks, 5
    )
    targets.append(s_h5)

    # 3. INTERACTIVE BUTTON SPRITES
    # Next Word Button
    btn_start_blocks = {
        "b_start_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_start_bc",
            "parent": None,
            "topLevel": True,
        },
        "b_start_bc": {
            "opcode": "event_broadcast",
            "next": None,
            "parent": "b_start_click",
            "inputs": {
                "BROADCAST_INPUT": [1, [11, "evaluate_next_word", "bc_evaluate"]],
            },
        },
    }
    s_btn_start = create_sprite(
        "Next Word Button", [make_costume("btn_start", "StartBtn", 60, 20)], -140, -10, 85, btn_start_blocks, 6
    )
    targets.append(s_btn_start)

    # Profile Button
    btn_prof_blocks = {
        "b_prof_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_prof_bc",
            "parent": None,
            "topLevel": True,
        },
        "b_prof_bc": {
            "opcode": "event_broadcast",
            "next": None,
            "parent": "b_prof_click",
            "inputs": {
                "BROADCAST_INPUT": [1, [11, "cycle_profile", "bc_profile"]],
            },
        },
    }
    s_btn_prof = create_sprite(
        "Profile Button", [make_costume("btn_prof", "ProfBtn", 60, 20)], -10, -10, 85, btn_prof_blocks, 7
    )
    targets.append(s_btn_prof)

    # Hamilton d10 Button
    btn_d10_blocks = {
        "b_d10_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_d10_bc",
            "parent": None,
            "topLevel": True,
        },
        "b_d10_bc": {
            "opcode": "event_broadcast",
            "next": None,
            "parent": "b_d10_click",
            "inputs": {
                "BROADCAST_INPUT": [1, [11, "calculate_hamilton", "bc_d10"]],
            },
        },
    }
    s_btn_d10 = create_sprite(
        "Hamilton d10 Button", [make_costume("btn_d10", "D10Btn", 60, 20)], 120, -10, 85, btn_d10_blocks, 8
    )
    targets.append(s_btn_d10)

    # Reset Button
    btn_rst_blocks = {
        "b_rst_click": {
            "opcode": "event_whenthisspriteclicked",
            "next": "b_rst_bc",
            "parent": None,
            "topLevel": True,
        },
        "b_rst_bc": {
            "opcode": "event_broadcast",
            "next": None,
            "parent": "b_rst_click",
            "inputs": {
                "BROADCAST_INPUT": [1, [11, "reset_story", "bc_reset"]],
            },
        },
    }
    s_btn_rst = create_sprite(
        "Reset Button", [make_costume("btn_rst", "ResetBtn", 40, 20)], 210, -10, 85, btn_rst_blocks, 9
    )
    targets.append(s_btn_rst)

    # d10 Die Mascot Sprite
    d10_blocks = {
        "b_die_recv": {
            "opcode": "event_whenbroadcastreceived",
            "next": "b_die_say",
            "parent": None,
            "topLevel": True,
            "fields": {
                "BROADCAST_OPTION": ["calculate_hamilton", "bc_d10"],
            },
        },
        "b_die_say": {
            "opcode": "looks_sayforsecs",
            "next": None,
            "parent": "b_die_recv",
            "inputs": {
                "MESSAGE": [1, [10, "Hamilton d10: 10 dice faces mapped! Grammar=4, Character=2, Setting=1, Action=1, Repeat=2."]],
                "SECS": [1, [4, 4]],
            },
        },
    }
    s_die = create_sprite(
        "d10 Die", [make_costume("d10_die", "D10Die", 35, 35)], -200, -70, 75, d10_blocks, 10
    )
    targets.append(s_die)

    # Main Story Controller / Announcer Sprite
    def make_chain(block_defs: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        blocks_dict: Dict[str, Any] = {}
        n = len(block_defs)
        for i, (b_id, opcode, inputs, fields) in enumerate(block_defs):
            prev_id = block_defs[i - 1][0] if i > 0 else None
            next_id = block_defs[i + 1][0] if i < n - 1 else None
            blocks_dict[b_id] = {
                "opcode": opcode,
                "next": next_id,
                "parent": prev_id,
                "inputs": inputs,
                "fields": fields,
                "shadow": False,
                "topLevel": (i == 0),
            }
        return blocks_dict

    ctrl_blocks: Dict[str, Any] = {}

    # 1. Flag clicked -> Initialize
    flag_chain = [
        ("b_flag", "event_whenflagclicked", {}, {}),
        ("b_init_say", "looks_sayforsecs", {"MESSAGE": [1, [10, "Welcome to the Five-Head Story Machine! Click NEXT WORD to evaluate candidates across 5 heads."]], "SECS": [1, [4, 4]]}, {}),
    ]
    ctrl_blocks.update(make_chain(flag_chain))

    # 2. Evaluate Next Word Broadcast
    eval_chain = [
        ("b_eval_recv", "event_whenbroadcastreceived", {}, {"BROADCAST_OPTION": ["evaluate_next_word", "bc_evaluate"]}),
        ("b_eval_del", "data_deletealloflist", {}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_add_forest", "data_addtolist", {"ITEM": [1, [10, "0.7625 (forest)"]]}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_add_castle", "data_addtolist", {"ITEM": [1, [10, "0.7225 (castle)"]]}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_add_garden", "data_addtolist", {"ITEM": [1, [10, "0.7075 (garden)"]]}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_add_ran", "data_addtolist", {"ITEM": [1, [10, "0.4400 (ran)"]]}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_add_milo", "data_addtolist", {"ITEM": [1, [10, "0.2375 (milo)"]]}, {"LIST": ["Fused Scores", "list_scores"]}),
        ("b_eval_set_sel", "data_setvariableto", {"VALUE": [1, [10, "forest"]]}, {"VARIABLE": ["Selected Word", "var_selected_word"]}),
        ("b_eval_set_fused", "data_setvariableto", {"VALUE": [1, [10, "0.7625"]]}, {"VARIABLE": ["Fused Score", "var_fused"]}),
        ("b_eval_set_g", "data_setvariableto", {"VALUE": [1, [10, "0.90"]]}, {"VARIABLE": ["Grammar Score", "var_score_g"]}),
        ("b_eval_set_c", "data_setvariableto", {"VALUE": [1, [10, "0.40"]]}, {"VARIABLE": ["Character Score", "var_score_c"]}),
        ("b_eval_set_s", "data_setvariableto", {"VALUE": [1, [10, "0.95"]]}, {"VARIABLE": ["Setting Score", "var_score_s"]}),
        ("b_eval_set_a", "data_setvariableto", {"VALUE": [1, [10, "0.30"]]}, {"VARIABLE": ["Action Score", "var_score_a"]}),
        ("b_eval_set_r", "data_setvariableto", {"VALUE": [1, [10, "0.85"]]}, {"VARIABLE": ["Repeat Score", "var_score_r"]}),
        ("b_eval_set_expl", "data_setvariableto", {"VALUE": [1, [10, "Grammar (0.90) & Setting (0.95) favored 'forest' (0.7625)!"]]}, {"VARIABLE": ["Explanation", "var_message"]}),
        ("b_eval_set_story", "data_setvariableto", {"VALUE": [1, [10, "Milo entered the forest"]]}, {"VARIABLE": ["Current Story", "var_story"]}),
        ("b_eval_add_story", "data_addtolist", {"ITEM": [1, [10, "forest"]]}, {"LIST": ["Completed Story Words", "list_story"]}),
        ("b_eval_say", "looks_sayforsecs", {"MESSAGE": [1, [10, "Word chosen: 'forest'! Score = 0.7625. Word added to narrative below!"]], "SECS": [1, [4, 3]]}, {}),
    ]
    ctrl_blocks.update(make_chain(eval_chain))

    # 3. Cycle Profile Broadcast
    prof_chain = [
        ("b_prof_recv", "event_whenbroadcastreceived", {}, {"BROADCAST_OPTION": ["cycle_profile", "bc_profile"]}),
        ("b_prof_say", "looks_sayforsecs", {"MESSAGE": [1, [10, "Profile switched! Weights updated. Notice how changing weights changes candidate fused scores!"]], "SECS": [1, [4, 3]]}, {}),
        ("b_prof_bc", "event_broadcast", {"BROADCAST_INPUT": [1, [11, "evaluate_next_word", "bc_evaluate"]]}, {}),
    ]
    ctrl_blocks.update(make_chain(prof_chain))

    # 4. Reset Story Broadcast
    rst_chain = [
        ("b_rst_recv", "event_whenbroadcastreceived", {}, {"BROADCAST_OPTION": ["reset_story", "bc_reset"]}),
        ("b_rst_set_story", "data_setvariableto", {"VALUE": [1, [10, "Milo entered the"]]}, {"VARIABLE": ["Current Story", "var_story"]}),
        ("b_rst_del_story", "data_deletealloflist", {}, {"LIST": ["Completed Story Words", "list_story"]}),
        ("b_rst_add1", "data_addtolist", {"ITEM": [1, [10, "Milo"]]}, {"LIST": ["Completed Story Words", "list_story"]}),
        ("b_rst_add2", "data_addtolist", {"ITEM": [1, [10, "entered"]]}, {"LIST": ["Completed Story Words", "list_story"]}),
        ("b_rst_add3", "data_addtolist", {"ITEM": [1, [10, "the"]]}, {"LIST": ["Completed Story Words", "list_story"]}),
        ("b_rst_set_prof", "data_setvariableto", {"VALUE": [1, [10, "Balanced"]]}, {"VARIABLE": ["Active Profile", "var_profile"]}),
        ("b_rst_say", "looks_sayforsecs", {"MESSAGE": [1, [10, "Story reset to starter prompt: 'Milo entered the'!"]], "SECS": [1, [4, 3]]}, {}),
        ("b_rst_bc", "event_broadcast", {"BROADCAST_INPUT": [1, [11, "evaluate_next_word", "bc_evaluate"]]}, {}),
    ]
    ctrl_blocks.update(make_chain(rst_chain))

    # 5. Hamilton d10 Broadcast
    d10_chain = [
        ("b_d10_recv", "event_whenbroadcastreceived", {}, {"BROADCAST_OPTION": ["calculate_hamilton", "bc_d10"]}),
        ("b_d10_say", "looks_sayforsecs", {"MESSAGE": [1, [10, "Hamilton d10: 10 dice faces mapped! Grammar=4, Character=2, Setting=1, Action=1, Repeat=2."]], "SECS": [1, [4, 4]]}, {}),
    ]
    ctrl_blocks.update(make_chain(d10_chain))

    s_ctrl = create_sprite(
        "Story Controller", [make_costume("story_icon", "StoryIcon", 50, 50)], 0, -70, 70, ctrl_blocks, 11
    )
    targets.append(s_ctrl)

    # Monitors (On-stage display of key variables & lists)
    monitors = [
        # Current Story Variable Monitor
        {
            "id": "var_story",
            "mode": "large",
            "opcode": "data_variable",
            "params": {"VARIABLE": "Current Story"},
            "spriteName": None,
            "value": "Milo entered the",
            "width": 420,
            "height": 26,
            "x": 25,
            "y": 305,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        },
        # Explanation Variable Monitor
        {
            "id": "var_message",
            "mode": "default",
            "opcode": "data_variable",
            "params": {"VARIABLE": "Explanation"},
            "spriteName": None,
            "value": "Grammar & Setting favored 'forest' (0.7625)!",
            "width": 0,
            "height": 0,
            "x": 25,
            "y": 335,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        },
        # Active Profile Monitor
        {
            "id": "var_profile",
            "mode": "default",
            "opcode": "data_variable",
            "params": {"VARIABLE": "Active Profile"},
            "spriteName": None,
            "value": "Balanced",
            "width": 0,
            "height": 0,
            "x": 10,
            "y": 145,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        },
        # Fused Scores List Monitor
        {
            "id": "list_scores",
            "mode": "list",
            "opcode": "data_listcontents",
            "params": {"LIST": "Fused Scores"},
            "spriteName": None,
            "value": ["0.7625 (forest)", "0.7225 (castle)", "0.7075 (garden)", "0.4400 (ran)", "0.2375 (milo)"],
            "width": 180,
            "height": 100,
            "x": 285,
            "y": 145,
            "visible": True,
        },
    ]

    return {
        "targets": targets,
        "monitors": monitors,
        "extensions": [],
        "meta": {
            "semver": "3.0.0",
            "vm": "0.2.0",
            "agent": "Antigravity Five-Head Story Machine Educational Project",
        },
    }


def generate_scratch_assets_and_project(target_dir: Path) -> Path:
    """Build SVG assets, project.json, and package into .sb3 archive."""
    scratch_dir = target_dir / "phase_7" / "scratch"
    source_dir = scratch_dir / "source"
    assets_dir = scratch_dir / "assets"
    sprites_dir = assets_dir / "sprites"
    bg_dir = assets_dir / "backgrounds"
    icons_dir = assets_dir / "icons"

    for d in (source_dir, sprites_dir, bg_dir, icons_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Raw SVGs dict
    raw_svgs = {
        "backdrop": (STAGE_BACKDROP_SVG, bg_dir / "story_studio_backdrop.svg"),
        "h1": (HEAD_GRAMMAR_SVG, sprites_dir / "head_1_grammar_scout.svg"),
        "h2": (HEAD_CHARACTER_SVG, sprites_dir / "head_2_character_keeper.svg"),
        "h3": (HEAD_SETTING_SVG, sprites_dir / "head_3_setting_painter.svg"),
        "h4": (HEAD_ACTION_SVG, sprites_dir / "head_4_action_booster.svg"),
        "h5": (HEAD_REPEAT_SVG, sprites_dir / "head_5_repeat_blocker.svg"),
        "btn_start": (BTN_START_SVG, icons_dir / "button_next_word.svg"),
        "btn_prof": (BTN_PROFILE_SVG, icons_dir / "button_profile.svg"),
        "btn_d10": (BTN_D10_SVG, icons_dir / "button_hamilton_d10.svg"),
        "btn_rst": (BTN_RESET_SVG, icons_dir / "button_reset.svg"),
        "d10_die": (D10_DIE_SVG, icons_dir / "hamilton_d10_die.svg"),
        "story_icon": (STORY_DISPLAY_SVG, icons_dir / "story_controller_icon.svg"),
    }

    # Save human-readable assets & calculate MD5
    asset_map: Dict[str, Tuple[str, str]] = {}
    sb3_files: Dict[str, bytes] = {}

    for key, (svg_content, out_path) in raw_svgs.items():
        data = svg_content.encode("utf-8")
        out_path.write_bytes(data)

        # Compute MD5
        md5_hash = hashlib.md5(data).hexdigest()
        filename = f"{md5_hash}.svg"
        asset_map[key] = (md5_hash, filename)
        sb3_files[filename] = data

    # Build project.json
    project_dict = build_scratch_project_json(asset_map)
    json_bytes = json.dumps(project_dict, indent=2).encode("utf-8")

    # Save human-readable project.json in source/
    (source_dir / "project.json").write_bytes(json_bytes)
    sb3_files["project.json"] = json_bytes

    # Package into .sb3 archive (ZIP format)
    sb3_path = scratch_dir / "five_head_story_machine.sb3"
    with zipfile.ZipFile(sb3_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in sb3_files.items():
            zf.writestr(fname, content)

    print(f"Scratch 3.0 project built successfully: {sb3_path}")
    print(f"Total archive entries: {len(sb3_files)} (project.json + {len(sb3_files)-1} SVG assets)")
    return sb3_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    generate_scratch_assets_and_project(root)
