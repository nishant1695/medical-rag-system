"""
Medical Safety Classifier

Classifies queries to route them appropriately:
- literature        : General research / literature review questions
- clinical_query    : Patient-specific scenarios, treatment planning, diagnosis
                      assistance (the PRIMARY use-case of this system — proceed
                      with clinical decision-support format + evidence disclaimer)
- emergency         : Acute life-threatening situations (hard-blocked — directs
                      to emergency services)

Design intent
-------------
This system is a clinical decision-support tool for healthcare professionals.
Clinical vignette questions ("55-year-old with X presents with Y…") are the
core use-case, not edge cases to be warned against. The classifier's job is to
ensure the *right answer format* is used, not to gate-keep legitimate queries.
"""
import re
from typing import Literal

SafetyClass = Literal["literature", "clinical_query", "emergency"]


class MedicalSafetyClassifier:
    """Classifies queries and shapes the system prompt accordingly."""

    # Signals a concrete patient scenario / clinical decision question
    CLINICAL_QUERY_PATTERNS = [
        r"\b\d+[\s-]?year[\s-]?old\s+(male|female|patient|man|woman|child|boy|girl)\b",
        r"\bpresents? with\b",
        r"\bhistory of\b",
        r"\b(he|she|they)\s+(has|had|is|was|presents?|underwent|received)\b",
        r"\bwhat is the (most appropriate|next|best|recommended)\b",
        r"\bhow (should|would|do) (I|we|you) (manage|treat|approach|handle)\b",
        r"\btreatment (plan|options?|approach) for\b",
        r"\bdiagnosis (of|for)\b",
        r"\bdifferential diagnosis\b",
        r"\bmanagement of\b",
        r"\bsentinel lymph node\b",
        r"\bbiopsy (shows?|revealed?|found)\b",
        r"\bmutation\b",
        r"\bstaging\b",
        r"\bfirst[\s-]line\b",
        r"\bsecond[\s-]line\b",
    ]

    EMERGENCY_PATTERNS = [
        r"\brespiratory\s+arrest\b",
        r"\bcardiac\s+arrest\b",
        r"\bcode\s+blue\b",
        r"\bunresponsive\b",
        r"\bcritically?\s+ill\b",
        r"\blife\s+threatening\b",
        r"\bsevere\s+bleeding\b",
        r"\btrauma\s+patient\b",
        r"\bpulse(less)?\b.{0,20}\bbreathing\b",
    ]

    # Explicit literature indicators — always classify as literature even if
    # clinical trigger words are present
    LITERATURE_INDICATORS = [
        r"\bin the literature\b",
        r"\bwhat does the (literature|evidence) (say|show|suggest|indicate)\b",
        r"\baccording to (studies|research|papers|guidelines)\b",
        r"\b(meta-analysis|systematic review|RCT|randomized controlled trial)\b",
        r"\bwhat are the (published )?outcomes\b",
        r"\b(evidence|guidelines|recommendations) (for|regarding|on)\b",
        r"\bin general\b",
        r"\btypically\b",
        r"\busually\b",
        r"\bstandard of care\b",
        r"\bsummarise the evidence\b",
        r"\bsummarize the evidence\b",
        r"\bwhat does research (show|say|suggest)\b",
    ]

    def classify(self, query: str) -> SafetyClass:
        query_lower = query.lower()

        # Literature indicators override everything
        if any(re.search(p, query_lower) for p in self.LITERATURE_INDICATORS):
            return "literature"

        # Emergency check (hard-block)
        if any(re.search(p, query_lower) for p in self.EMERGENCY_PATTERNS):
            return "emergency"

        # Clinical scenario / decision query
        if any(re.search(p, query_lower) for p in self.CLINICAL_QUERY_PATTERNS):
            return "clinical_query"

        return "literature"

    def should_block_query(self, safety_class: SafetyClass) -> bool:
        """Only genuine emergencies are hard-blocked."""
        return safety_class == "emergency"

    def get_warning_message(self, safety_class: SafetyClass) -> str:
        """
        Returns a user-facing message prepended to the response where needed.
        Clinical queries get no intrusive warning — just a footer disclaimer.
        """
        if safety_class == "emergency":
            return (
                "🚨 **This system cannot be used for emergency medical decisions.**\n\n"
                "If this is a medical emergency, call emergency services (911) immediately "
                "and follow your institution's emergency protocols.\n\n"
                "This system is for literature review and clinical decision support only — "
                "it cannot replace real-time clinical judgment in emergencies."
            )
        return ""  # No warning for clinical_query or literature

    def get_system_prompt_modification(self, safety_class: SafetyClass) -> str:
        """
        Additional instructions appended to the system prompt based on query type.
        """
        if safety_class == "clinical_query":
            return """

## Clinical Decision Support Mode
This is a clinical scenario query. Your answer must be:
1. **Structured** — use the sections: Clinical Summary, Evidence-Based Options,
   Key Considerations, Evidence Quality.
2. **Evidence-graded** — cite every recommendation with [cid] and state the
   evidence level (Level I–V).
3. **Clinically actionable** — answer as a knowledgeable colleague would:
   specific, direct, ranked by evidence strength.
4. **Appropriately caveated** — end every response with the standard disclaimer.

Do NOT hedge to the point of being useless. The user is a qualified clinician
who needs evidence synthesis, not a refusal to engage.
"""
        elif safety_class == "emergency":
            return """
CRITICAL: Direct the user to emergency services. Do not provide clinical guidance.
"""
        return ""


# Global instance
safety_classifier = MedicalSafetyClassifier()
