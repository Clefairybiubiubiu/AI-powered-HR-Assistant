"""
Scaffolding for integrating the O*NET skill taxonomy.

This module intentionally ships without the actual O*NET CSV parsing logic.
It establishes the abstractions, caching helpers, and logger usage so that
real data can be connected later without refactoring the rest of the app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .utils.logger import get_logger


class ONETSkillTaxonomy:
    """
    Lightweight facade for the O*NET taxonomy.

    Once populated with real data, this class will be responsible for:
    - Loading O*NET CSV datasets (Skills, Work Activities, Technology Skills, etc.)
    - Normalizing skill names and building lookup tables
    - Expanding a skill into related concepts (synonyms, work activities, knowledge areas)
    - Serving as the single entry point for the matching pipeline
    """

    def __init__(
        self,
        data_dir: str | Path = "data/onet/",
        enable_cache: bool = True,
        auto_load: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.enable_cache = enable_cache
        self.cache_dir = Path("cache") / "onet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_loaded = False
        self._skill_index: Dict[str, Dict] = {}
        self.logger = get_logger(__name__)

        if auto_load:
            self.load_onet_data()

    # ------------------------------------------------------------------ #
    # Placeholder public API
    # ------------------------------------------------------------------ #
    def load_onet_data(self, data_dir: Optional[str | Path] = None) -> None:
        """
        Load O*NET CSV files and populate in-memory indexes.

        Args:
            data_dir: Optional override for the default data directory.

        Expected behavior (future implementation):
            - Parse Skills.txt, Technology Skills.txt, Work Activities.txt, etc.
            - Normalize all skill names (lowercase, strip punctuation).
            - Build dictionaries for quick lookup and fuzzy matching.
        """
        if data_dir:
            self.data_dir = Path(data_dir)

        if self._data_loaded:
            self.logger.debug("O*NET data already loaded; skipping reload.")
            return

        # Future implementation will populate self._skill_index
        self.logger.info("ONETSkillTaxonomy.load_onet_data: placeholder executed.")
        self._data_loaded = True

    def get_related_skills(self, skill: str) -> List[str]:
        """
        Return a list of related skills for the provided skill name.

        In the final implementation, this will query the O*NET data to find:
            - Synonyms
            - Closely related technology skills
            - Skills appearing in the same work activities
        """
        self.logger.debug("get_related_skills called for %s (stub).", skill)
        return []

    def get_skill_category(self, skill: str) -> Optional[str]:
        """
        Return the O*NET category for a skill (e.g., Cognitive, Technical, Social).

        Returns:
            Category string if available, otherwise None.
        """
        self.logger.debug("get_skill_category called for %s (stub).", skill)
        return None

    def expand_skill(self, skill: str) -> Dict[str, List[str]]:
        """
        Expand a skill into related O*NET concepts.

        Expected output structure:
        {
            "skill": "SQL",
            "synonyms": [...],
            "related_skills": [...],
            "work_activities": [...],
            "knowledge_areas": [...],
            "technology_skills": [...]
        }
        """
        self.logger.debug("expand_skill called for %s (stub).", skill)
        cached = self._load_cache(skill)
        if cached is not None:
            return cached

        result: Dict[str, List[str]] = {
            "skill": skill,
            "synonyms": [],
            "related_skills": [],
            "work_activities": [],
            "knowledge_areas": [],
            "technology_skills": [],
        }

        self._save_cache(skill, result)
        return result

    # ------------------------------------------------------------------ #
    # Cache helpers (currently no-ops except for directory creation)
    # ------------------------------------------------------------------ #
    def _load_cache(self, skill: str) -> Optional[Dict]:
        """
        Load cached expansion results if present.

        In the future, this will read from cache/onet/<normalized-skill>.json.
        """
        if not self.enable_cache:
            return None
        cache_path = self.cache_dir / f"{self._normalize_skill(skill)}.json"
        if cache_path.exists():
            try:
                import json

                return json.loads(cache_path.read_text())
            except Exception as exc:  # pragma: no cover - best-effort logging
                self.logger.warning("Failed to load O*NET cache for %s: %s", skill, exc)
        return None

    def _save_cache(self, skill: str, data: Dict) -> None:
        """
        Persist expansion results for later reuse.
        """
        if not self.enable_cache:
            return
        cache_path = self.cache_dir / f"{self._normalize_skill(skill)}.json"
        try:
            import json

            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as exc:  # pragma: no cover - best-effort logging
            self.logger.warning("Failed to save O*NET cache for %s: %s", skill, exc)

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        """
        Normalize skills for consistent lookups and cache keys.
        """
        return skill.strip().lower().replace(" ", "_")


