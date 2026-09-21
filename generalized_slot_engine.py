"""
generalized_slot_engine.py
==========================
Machine-readable dynamic slot-acceptor and assortment constructor.
Supports arbitrary domain -> subject -> topic taxonomies, generalized slot
acceptors, and syntactic surface realization.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VocabularyTerm:
    term_id: str
    term: str
    type: str  # e.g., 'relationship', 'entity', 'target', 'temporal', 'evidence_status'
    domain: str = "*"
    subject: str = "*"
    topic: str = "*"
    tags: Tuple[str, ...] = field(default_factory=tuple)
    inflections: Dict[str, str] = field(default_factory=dict)

    def render(self, form: Optional[str] = None) -> str:
        if not form or form == "default":
            return self.term
        return self.inflections.get(form, self.term)


@dataclass
class SlotAcceptor:
    """
    Generalized gatekeeper defining what vocabulary terms are admissible
    for a target data slot.
    """
    allowed_types: Set[str]
    domain_scope: Set[str] = field(default_factory=lambda: {"*"})
    subject_scope: Set[str] = field(default_factory=lambda: {"*"})
    topic_scope: Set[str] = field(default_factory=lambda: {"*"})
    required_tags: Set[str] = field(default_factory=set)
    excluded_tags: Set[str] = field(default_factory=set)
    predicate: Optional[Callable[[VocabularyTerm], bool]] = None

    def accepts(self, term: VocabularyTerm) -> bool:
        # 1. Type check
        if term.type not in self.allowed_types:
            return False

        # 2. Domain-Subject-Topic scope check (with wildcard '*' support)
        if "*" not in self.domain_scope and term.domain != "*" and term.domain not in self.domain_scope:
            return False

        if "*" not in self.subject_scope and term.subject != "*" and term.subject not in self.subject_scope:
            return False

        if "*" not in self.topic_scope and term.topic != "*" and term.topic not in self.topic_scope:
            return False

        # 3. Tag filters
        term_tags = set(term.tags)
        if self.required_tags and not self.required_tags.issubset(term_tags):
            return False

        if self.excluded_tags and bool(self.excluded_tags.intersection(term_tags)):
            return False

        # 4. Custom predicate check
        if self.predicate and not self.predicate(term):
            return False

        return True


@dataclass
class SlotDefinition:
    slot_id: str
    semantic_role: str
    acceptor: SlotAcceptor
    is_optional: bool = False
    default_term: Optional[str] = None


@dataclass
class QueryTemplate:
    template_id: str
    pattern: str  # e.g., "What {rel} {entity:plural} relate to {target} [{temporal}]?"
    required_slots: List[str]
    optional_slots: List[str] = field(default_factory=list)

    def extract_slot_requests(self) -> List[Tuple[str, Optional[str]]]:
        """Extract (slot_name, format_specifier) from pattern."""
        matches = re.findall(r"\{([a-zA-Z0-9_]+)(?::([a-zA-Z0-9_]+))?\}", self.pattern)
        return [(m[0], m[1] if m[1] else None) for m in matches]


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class GeneralizedTaxonomyEngine:
    def __init__(self):
        self.vocabulary: Dict[str, VocabularyTerm] = {}
        self.slots: Dict[str, SlotDefinition] = {}
        self.templates: List[QueryTemplate] = []
        self.cross_slot_constraints: List[Dict[str, Any]] = []

    def register_term(self, term: VocabularyTerm) -> None:
        self.vocabulary[term.term_id] = term

    def register_slot(self, slot: SlotDefinition) -> None:
        self.slots[slot.slot_id] = slot

    def register_template(self, template: QueryTemplate) -> None:
        self.templates.append(template)

    def add_cross_constraint(self, constraint: Dict[str, Any]) -> None:
        self.cross_slot_constraints.append(constraint)

    def get_candidate_terms(self, slot_id: str) -> List[VocabularyTerm]:
        if slot_id not in self.slots:
            raise KeyError(f"Slot '{slot_id}' not registered in engine.")
        acceptor = self.slots[slot_id].acceptor
        return [term for term in self.vocabulary.values() if acceptor.accepts(term)]

    def validate_combination(self, combination: Dict[str, VocabularyTerm]) -> bool:
        """Evaluate multi-slot constraint rules."""
        for rule in self.cross_slot_constraints:
            rule_type = rule.get("type")

            if rule_type == "requires":
                trigger = rule["if"]
                consequent = rule["then"]
                slot_a, val_a = trigger["slot"], trigger["value"]
                slot_b, vals_b = consequent["slot"], set(consequent["values"])

                if slot_a in combination and combination[slot_a].term_id == val_a:
                    if slot_b not in combination or combination[slot_b].term_id not in vals_b:
                        return False

            elif rule_type == "excludes":
                trigger = rule["if"]
                forbidden = rule["then"]
                slot_a, val_a = trigger["slot"], trigger["value"]
                slot_b, vals_b = forbidden["slot"], set(forbidden["values"])

                if slot_a in combination and combination[slot_a].term_id == val_a:
                    if slot_b in combination and combination[slot_b].term_id in vals_b:
                        return False

            elif rule_type == "domain_alignment":
                # Ensure all selected terms across non-wildcard slots share compatible domains
                active_domains = {
                    term.domain for term in combination.values()
                    if term.domain != "*"
                }
                if len(active_domains) > 1:
                    return False

        return True

    def render_query(
        self,
        template: QueryTemplate,
        combination: Dict[str, VocabularyTerm],
    ) -> str:
        """
        Renders template using surface-level inflection transformations.
        Handles optional bracketed blocks e.g. ' [at {temporal}]' gracefully.
        """
        rendered = template.pattern

        # Resolve optional blocks denoted by [...]
        optional_blocks = re.findall(r"\[(.*?)\]", rendered)
        for block in optional_blocks:
            slots_in_block = re.findall(r"\{([a-zA-Z0-9_]+)(?::[a-zA-Z0-9_]+)?\}", block)
            all_slots_present = all(
                s in combination and combination[s] is not None
                for s in slots_in_block
            )
            if all_slots_present:
                # Keep inner content without brackets
                rendered = rendered.replace(f"[{block}]", block)
            else:
                # Strip out optional block entirely
                rendered = rendered.replace(f"[{block}]", "")

        # Format remaining slots
        slot_refs = template.extract_slot_requests()
        format_args = {}
        for slot_name, inflection in slot_refs:
            if slot_name in combination and combination[slot_name] is not None:
                format_args[slot_name] = combination[slot_name].render(inflection)
            else:
                format_args[slot_name] = ""

        # Apply standard formatting and clean up residual double spacing
        try:
            output = rendered.format(**format_args)
            output = re.sub(r"\s+", " ", output).strip()
            # Clean punctuation spacing before '?' or '.'
            output = re.sub(r"\s+([?.!,])", r"\1", output)
            return output
        except KeyError as exc:
            raise ValueError(f"Unfilled slot in template: {exc}")

    def assortments(
        self,
        slot_names: Sequence[str],
        *,
        randomized: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[Dict[str, VocabularyTerm]]:
        """
        Lazily generates all valid combinations across requested slots.
        """
        rng = random.Random(seed)
        slot_candidates: List[List[VocabularyTerm]] = []

        for name in slot_names:
            candidates = self.get_candidate_terms(name)
            if not candidates and not self.slots[name].is_optional:
                raise ValueError(f"No candidate vocabulary accepted for required slot: '{name}'")
            if self.slots[name].is_optional:
                # Add None sentinel for optional slots
                candidates = list(candidates) + [None]  # type: ignore
            else:
                candidates = list(candidates)

            if randomized:
                rng.shuffle(candidates)
            slot_candidates.append(candidates)

        for combo_tuple in itertools.product(*slot_candidates):
            combo = dict(zip(slot_names, combo_tuple))
            # Remove None values for validation
            active_combo = {k: v for k, v in combo.items() if v is not None}
            if self.validate_combination(active_combo):
                yield active_combo

    def generate_records(
        self,
        slot_names: Sequence[str],
        *,
        seed: Optional[int] = None,
        randomize_templates: bool = True,
    ) -> Iterator[Tuple[Dict[str, Any], str]]:
        """
        Yields (metadata_dict, rendered_query) for every valid assortment.
        """
        rng = random.Random(seed)
        if not self.templates:
            raise ValueError("No templates registered.")

        for combo in self.assortments(slot_names, randomized=True, seed=seed):
            if randomize_templates:
                template = rng.choice(self.templates)
            else:
                template = self.templates[0]

            query_text = self.render_query(template, combo)
            meta = {k: (v.term if v else None) for k, v in combo.items()}
            meta["template_id"] = template.template_id
            yield meta, query_text

    def export_csv(
        self,
        output_path: Path | str,
        slot_names: Sequence[str],
        limit: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> int:
        fieldnames = list(slot_names) + ["template_id", "generated_query"]
        count = 0
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for meta, query in self.generate_records(slot_names, seed=seed):
                row = dict(meta)
                row["generated_query"] = query
                writer.writerow(row)
                count += 1
                if limit and count >= limit:
                    break
        return count

