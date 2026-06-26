"""Public-safe configuration for the legacy RaMA heart Qwen/SAM code.

All LLM credentials and model paths default to empty / environment variables.
Values are resolved through the unified config loader (env vars or
``configs/rama_config.yaml``); NO API key is hardcoded here. To run the live
Qwen-VL scoring stage, set ``RAMA_LLM_API_KEY`` (or fill ``llm.api_key`` in
``configs/rama_config.yaml``).

The released pipeline normally runs from CACHED Qwen outputs and never calls the
API; these settings are only needed if you re-run the live scoring stage.
"""

from __future__ import annotations

import os

try:
    from rama_config_loader import get as _cfg_get
except Exception:  # pragma: no cover - allow import without package context
    def _cfg_get(dotted_key, env=None, default=None):  # type: ignore
        return os.environ.get(env) if env else default


# ---------------------------------------------------------------------------
# LLM (Qwen-VL) meta-controller credentials and settings
# ---------------------------------------------------------------------------
# API key: prefer RAMA_LLM_API_KEY env var; never commit a real key.
OPENAI_API_KEY = _cfg_get("llm.api_key", "RAMA_LLM_API_KEY", "")
OPENAI_BASE_URL = _cfg_get(
    "llm.base_url",
    "RAMA_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
OPENAI_MODEL = _cfg_get("llm.model", "RAMA_CACHE_MODEL", "qwen3-vl-plus")
LLM_PROVIDER = os.getenv("RAMA_LLM_PROVIDER", "openai")

# Tri-partition score thresholds (paper Sec. 2.1 / experiment settings).
DEFAULT_TAU_LOW = int(os.getenv("RAMA_TAU_LOW", "40"))
DEFAULT_TAU_HIGH = int(os.getenv("RAMA_TAU_HIGH", "90"))

# Enable optional LLM deep-thinking (long reasoning) mode. Disable for batch scoring
# with multiprocessing to avoid hangs.
ENABLE_THINKING = os.getenv("RAMA_ENABLE_THINKING", "false").lower() == "true"

# Merge the system prompt into the user message (needed by some models, e.g.
# Gemini).
MERGE_SYSTEM_PROMPT = os.getenv("RAMA_MERGE_SYSTEM_PROMPT", "false").lower() == "true"

# Diagnostic grid overlay size used by grid_visualizer.
GRID_SIZE = int(os.getenv("RAMA_GRID_SIZE", "8"))

# Output directory for ad-hoc agent runs.
SAVE_DIR = os.getenv("RAMA_SAVE_DIR", "./asga_outputs")

# ---------------------------------------------------------------------------
# SAM agent settings
# ---------------------------------------------------------------------------
SAM3_ENABLE_SEG = os.getenv("RAMA_SAM3_ENABLE_SEG", "1") != "0"
SAM3_ENABLE_INST = os.getenv("RAMA_SAM3_ENABLE_INST", "1") != "0"
SAM3_CKPT_PATH = _cfg_get("sam_agents.sam3_ckpt", "RAMA_SAM3_CKPT", "")
MEDSAM_CKPT = _cfg_get("sam_agents.medsam2_ckpt", "RAMA_MEDSAM2_CKPT", "")

# Initial-mask prompt logits used by the SAM3 wrapper.
CONF_POS_LOGIT = float(os.getenv("RAMA_CONF_POS_LOGIT", "0.3"))
CONF_NEG_LOGIT = float(os.getenv("RAMA_CONF_NEG_LOGIT", "-0.3"))
