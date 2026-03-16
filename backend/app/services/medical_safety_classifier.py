"""
Medical Safety Classifier

Classifies queries to prevent misuse of the system for patient-specific
medical advice. Detects three categories:
- literature: General research questions (safe)
- patient_specific: Questions about specific patients (blocked/downgraded)
- emergency: Emergency medical situations (blocked with strong warning)
"""
import re
from typing import Literal

SafetyClass = Literal["literature", "patient_specific", "emergency"]


class MedicalSafetyClassifier:
    """Classifies queries into safety categories."""

    # Pattern groups for detection
    PATIENT_SPECIFIC_PATTERNS = [
        r"\bmy patient\b",
        r"\bthis patient\b",
        r"\bour patient\b",
        r"\bshould I (perform|do|give|prescribe|administer)\b",
        r"\bwhat (dose|dosage|medication|drug)\s+(should|would|can)\b",
        r"\bhow much (should|would|can) (I|we)\b",
        r"\b\d+[\s-]?year[\s-]?old (male|female|patient|man|woman|child)\b",
        r"\bBMI\s*[:=]?\s*\d+\b",
        r"\b(he|she|they) (has|had|is|was|presents with)\b",
        r"\bcase:\s*\w+",
        r"\bpatient (presented|presents|came|comes) with\b",
        r"\bin (this|my) case\b",
        r"\bfor (this|my) (specific )?patient\b",
    ]

    EMERGENCY_PATTERNS = [
        r"\bemergency\b",
        r"\bimmediate(ly)?\b",
        r"\burgent(ly)?\b",
        r"\bsevere bleeding\b",
        r"\brespiratory (distress|arrest)\b",
        r"\bcardiac arrest\b",
        r"\bcode blue\b",
        r"\bunresponsive\b",
        r"\bcritical(ly)? ill\b",
        r"\blife threatening\b",
        r"\btrauma patient\b",
    ]

    # Exclusion patterns that indicate literature questions even with trigger words
    LITERATURE_INDICATORS = [
        r"\bin the literature\b",
        r"\bwhat does the (literature|evidence) (say|show|suggest)\b",
        r"\baccording to (studies|research|papers)\b",
        r"\b(meta-analysis|systematic review|RCT)\b",
        r"\bwhat are the (indications|contraindications|outcomes)\b",
        r"\b(evidence|guidelines|recommendations) (for|regarding)\b",
        r"\bin general\b",
        r"\btypically\b",
        r"\busually\b",
        r"\bstandard (of care|approach|practice)\b",
    ]

    def classify(self, query: str) -> SafetyClass:
        """
        Classify query into safety category.

        Args:
            query: User query text

        Returns:
            SafetyClass: "literature", "patient_specific", or "emergency"
        """
        query_lower = query.lower()

        # Check for literature indicators first (override patient-specific detection)
        if any(re.search(pattern, query_lower) for pattern in self.LITERATURE_INDICATORS):
            return "literature"

        # Check emergency patterns (highest priority)
        if any(re.search(pattern, query_lower) for pattern in self.EMERGENCY_PATTERNS):
            return "emergency"

        # Check patient-specific patterns
        if any(re.search(pattern, query_lower) for pattern in self.PATIENT_SPECIFIC_PATTERNS):
            return "patient_specific"

        # Default to literature
        return "literature"

    def get_warning_message(self, safety_class: SafetyClass) -> str:
        """
        Get appropriate warning message for safety classification.

        Args:
            safety_class: Safety classification

        Returns:
            Warning message string (empty for literature queries)
        """
        if safety_class == "emergency":
            return """
⚠️ **EMERGENCY SITUATION DETECTED**

This system is for EDUCATIONAL and RESEARCH purposes ONLY and should NOT be used for emergency medical decisions.

🚨 **If this is a medical emergency:**
- Call emergency services immediately (911 in US)
- Consult with an attending physician
- Follow your institution's emergency protocols
- Do NOT rely on this system for emergency care

This system cannot replace clinical judgment in emergency situations.
"""
        elif safety_class == "patient_specific":
            return """
⚠️ **PATIENT-SPECIFIC QUERY DETECTED**

This query appears to ask about a specific patient case. This system can ONLY provide:
- General evidence summaries from published research literature
- Published clinical guidelines and protocols
- Information about surgical techniques and outcomes from studies

This system CANNOT provide:
- Patient-specific medical advice
- Treatment recommendations for individual cases
- Dose calculations or medication choices for specific patients

**All clinical decisions must be made by qualified healthcare professionals who can evaluate the individual patient's circumstances, medical history, and current condition.**

Educational purposes only. Not a substitute for clinical judgment.
"""
        else:
            return ""

    def should_block_query(self, safety_class: SafetyClass) -> bool:
        """
        Determine if query should be completely blocked.

        Args:
            safety_class: Safety classification

        Returns:
            True if query should be blocked, False if it can proceed with warning
        """
        return safety_class == "emergency"

    def get_system_prompt_modification(self, safety_class: SafetyClass) -> str:
        """
        Get system prompt modification based on safety class.

        Args:
            safety_class: Safety classification

        Returns:
            Additional system prompt text to reinforce safety
        """
        if safety_class == "patient_specific":
            return """
CRITICAL SAFETY CONSTRAINT:
This query appears to be about a specific patient. You MUST:
- Provide ONLY general evidence from research literature
- Use phrases like "studies show", "research suggests", "in general"
- NEVER give specific recommendations for this patient
- NEVER calculate doses or suggest specific treatments
- Remind the user that clinical decisions require a qualified physician
"""
        elif safety_class == "emergency":
            return """
CRITICAL: This appears to be an emergency situation. You MUST:
- Strongly advise calling emergency services
- State this system cannot handle emergencies
- Refuse to provide specific guidance
"""
        else:
            return ""


# Global instance
safety_classifier = MedicalSafetyClassifier()
