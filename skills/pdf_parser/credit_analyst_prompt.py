"""
Credit Analyst System Prompt and Integration

Provides expert credit analysis capabilities for intelligent section analysis,
metadata enrichment, and NER in credit agreements.
"""

CREDIT_ANALYST_SYSTEM_PROMPT = """You are an expert syndicated credit analyst specializing in complex, large-scale credit agreements ($100M-$10B+ facilities). Your expertise lies in understanding how credit documents are structured as interconnected systems, not isolated sections, and are able to identify which sections may contain specific information relative to the question presented.

    CORE PRINCIPLE: Credit agreements follow logical chains where components must connect and contain multiple layers that must be analyzed:
    - Primary provisions that appear to answer a question
    - Cross-referenced provisions that modify or add conditions
    - Overriding provisions (e.g., Change of Control can override merger provisions)
    - Alternative paths (e.g., multiple debt/lien baskets for the same action)
    - Structural chains: Definitions → Covenants → Defaults → Cure Rights → Enforcement

    YOUR TASK: Create a prioritized search plan by listing section IDs in the order they should be searched (you will use all sections in your search).

    PRIORITIZATION FRAMEWORK:

    1. IDENTIFY THE QUESTION TYPE
    - Definitional: "What is X?" → Start with definitions, then mechanics
    - Mechanical: "How does X work?" → Start with primary section, then related mechanics
    - Permissibility: "Can borrower do X?" → Restrictions first, then exceptions/carve-outs
    - Consequences: "What happens if X?" → Defaults, then cure rights, then enforcement
    - Comprehensive: "Analyze X" → Full chain from definitions through enforcement

    2. FOLLOW THE COMPLETE LOGICAL CHAIN (Don't Stop Early!)
    - Definitions: If technical/capitalized terms exist, ALWAYS include definition sections
    - Primary Content: The main section addressing the question
    - Cross-References: Any sections referenced within the primary section
    - Parallel Provisions: Alternative ways to accomplish the same thing (e.g., multiple debt baskets)
    - Structural Dependencies: 
        * For debt questions → Also include corresponding lien sections
        * For covenant questions → Also include related default provisions
        * For merger/acquisition questions → Also include Change of Control definitions and defaults
    - Exceptions/Baskets: ALL carve-outs, not just the first one (e.g., general basket AND ratio basket AND Available Amount)
    - Consequences: Related defaults, cure periods, grace periods, WHO can enforce
    - Modifying Provisions: Sections that might override or limit the primary answer

    3. APPLY EXPERT HEURISTICS
    - For financial covenant questions → definitions + covenant section + corresponding defaults + cure rights + springing language
    - For "Can borrower incur debt/liens?" → ALL debt baskets (general, ratio-based, Available Amount) + corresponding lien permissions for EACH + related defaults
    - For merger/acquisition questions → Fundamental Changes covenant + Change of Control definition + Change of Control default
    - For payment/prepayment questions → payment mechanics + prepayment provisions + fees + yield protection
    - For restriction questions → Start with prohibition, then CHECK ALL exception baskets systematically

    4. OPTIMIZE SEARCH EFFICIENCY
    - Place highest-probability sections first
    - Group related sections together in logical reading order
    - Include upstream dependencies before downstream references (e.g., definitions before provisions that use them)
    - Cast a wide net - include all sections in your prioritized list; the retrieval process will stop when the question is answered

    OUTPUT: Return a structured output with a prioritized list of section IDs with brief reasoning for your search strategy."""


class CreditAnalystEnhancer:
    """Enhances document analysis with credit analyst expertise."""
    
    def __init__(self):
        """Initialize credit analyst enhancer."""
        self.system_prompt = CREDIT_ANALYST_SYSTEM_PROMPT
    
    def get_system_prompt(self) -> str:
        """Get the credit analyst system prompt."""
        return self.system_prompt
    
    def analyze_section_importance(
        self,
        section_title: str,
        section_content: str,
        section_level: int
    ) -> dict:
        """
        Analyze a section's importance and role in the credit agreement structure.
        
        Args:
            section_title: Title of the section
            section_content: Content of the section
            section_level: Hierarchical level of the section
            
        Returns:
            Dictionary with importance rating and classification
        """
        classification = self._classify_section_type(section_title, section_content)
        importance = self._calculate_importance(classification, section_level)
        
        return {
            "classification": classification,
            "importance_score": importance,
            "typical_dependencies": self._get_typical_dependencies(classification),
            "likely_cross_references": self._find_likely_cross_references(section_title)
        }
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """Classify section type based on title and content patterns."""
        title_lower = title.lower()
        content_lower = content.lower()[:500]  # Check first 500 chars
        
        # Definitions
        if "definition" in title_lower or "definitions" in title_lower:
            return "DEFINITIONS"
        
        # Covenants
        if "covenant" in title_lower or "undertaking" in title_lower:
            return "COVENANT"
        
        # Conditions
        if "condition" in title_lower or "precedent" in title_lower:
            return "CONDITION"
        
        # Events of Default
        if "default" in title_lower or "event of default" in title_lower:
            return "DEFAULT"
        
        # Remedies/Enforcement
        if "remedy" in title_lower or "enforcement" in title_lower or "acceleration" in title_lower:
            return "REMEDY"
        
        # Financial/Representations
        if "representation" in title_lower or "warranty" in title_lower:
            return "REPRESENTATION"
        
        # Fees/Charges
        if "fee" in title_lower or "interest" in title_lower or "charge" in title_lower:
            return "FEES"
        
        # Structure/Mechanics
        if "loan" in title_lower or "advance" in title_lower or "disbursement" in title_lower:
            return "MECHANICS"
        
        # Change of Control
        if "change of control" in title_lower or "merger" in title_lower or "acquisition" in title_lower:
            return "CHANGE_OF_CONTROL"
        
        # Debt/Lien Permissions
        if "debt" in title_lower or "lien" in title_lower or "indebtedness" in title_lower:
            return "DEBT_LIEN"
        
        # Other
        return "GENERAL"
    
    def _calculate_importance(self, classification: str, level: int) -> float:
        """Calculate importance score (0-1) based on classification and level."""
        # Base importance by type
        importance_weights = {
            "DEFINITIONS": 0.95,
            "COVENANT": 0.90,
            "DEFAULT": 0.92,
            "CONDITION": 0.85,
            "CHANGE_OF_CONTROL": 0.88,
            "DEBT_LIEN": 0.87,
            "REMEDY": 0.86,
            "MECHANICS": 0.80,
            "FEES": 0.75,
            "REPRESENTATION": 0.70,
            "GENERAL": 0.50
        }
        
        base_score = importance_weights.get(classification, 0.50)
        
        # Adjust for hierarchy level (higher levels = more important)
        level_adjustment = max(0, 0.1 - (level * 0.01))
        
        return min(1.0, base_score + level_adjustment)
    
    def _get_typical_dependencies(self, classification: str) -> list:
        """Get typical sections that depend on or relate to this section type."""
        dependencies = {
            "DEFINITIONS": ["COVENANT", "CONDITION", "DEFAULT", "MECHANICS"],
            "COVENANT": ["DEFAULT", "REMEDY", "DEFINITIONS"],
            "DEFAULT": ["REMEDY", "COVENANT"],
            "CONDITION": ["DEFINITIONS", "MECHANICS"],
            "CHANGE_OF_CONTROL": ["COVENANT", "DEFAULT", "DEFINITIONS"],
            "DEBT_LIEN": ["COVENANT", "DEFAULT", "CONDITION"],
            "REMEDY": ["DEFAULT", "COVENANT"],
            "MECHANICS": ["DEFINITIONS", "CONDITION", "FEES"],
            "FEES": ["MECHANICS", "DEFINITIONS"],
            "REPRESENTATION": ["CONDITION", "DEFAULT"],
            "GENERAL": []
        }
        return dependencies.get(classification, [])
    
    def _find_likely_cross_references(self, title: str) -> list:
        """Identify likely cross-references based on section title."""
        keywords = []
        title_lower = title.lower()
        
        # Common cross-reference patterns
        patterns = {
            "debt": ["lien", "covenant", "default", "condition"],
            "lien": ["debt", "covenant", "condition"],
            "merger": ["change of control", "fundamental change", "covenant", "default"],
            "covenant": ["definitions", "default", "condition"],
            "default": ["covenant", "remedy", "acceleration"],
            "payment": ["interest", "fee", "prepayment"],
            "financial": ["definitions", "covenant", "default"],
            "change of control": ["merger", "covenant", "default", "fundamental change"],
        }
        
        for keyword, references in patterns.items():
            if keyword in title_lower:
                keywords.extend(references)
        
        return list(set(keywords))  # Deduplicate
    
    def create_search_strategy(self, question: str, available_sections: list) -> dict:
        """
        Create a prioritized search strategy for a given question using all sections.
        
        Args:
            question: The analytical question
            available_sections: List of available section data (with titles, IDs, classifications)
            
        Returns:
            Prioritized search strategy with section ordering and reasoning
        """
        # Classify the question type
        question_type = self._classify_question_type(question)
        
        # Score each section based on question type and section classification
        scored_sections = []
        for section in available_sections:
            score = self._score_section_for_question(
                section,
                question_type,
                question
            )
            scored_sections.append({
                "section_id": section.get("id"),
                "section_title": section.get("title"),
                "score": score,
                "relevance_reason": self._explain_relevance(section, question_type)
            })
        
        # Sort by score (descending)
        scored_sections.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "question_type": question_type,
            "prioritized_sections": scored_sections,
            "search_strategy": self._build_search_explanation(question_type),
            "total_sections_to_search": len(scored_sections)
        }
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of analytical question."""
        q_lower = question.lower()
        
        if "what is" in q_lower or "define" in q_lower:
            return "DEFINITIONAL"
        elif "how" in q_lower and ("work" in q_lower or "does" in q_lower):
            return "MECHANICAL"
        elif "can" in q_lower or "permitted" in q_lower or "allow" in q_lower:
            return "PERMISSIBILITY"
        elif "what happen" in q_lower or "consequence" in q_lower or "result" in q_lower:
            return "CONSEQUENCES"
        elif "analyze" in q_lower or "comprehensive" in q_lower:
            return "COMPREHENSIVE"
        else:
            return "GENERAL"
    
    def _score_section_for_question(self, section: dict, question_type: str, question: str) -> float:
        """Score a section's relevance to the question."""
        base_score = 0.0
        title_lower = section.get("title", "").lower()
        classification = self._classify_section_type(title_lower, section.get("content", ""))
        
        # Question type → Section type mapping
        if question_type == "DEFINITIONAL":
            if classification == "DEFINITIONS":
                base_score = 0.95
            elif classification == "GENERAL":
                base_score = 0.3
            else:
                base_score = 0.5
        
        elif question_type == "MECHANICAL":
            if classification in ["MECHANICS", "DEFINITIONS", "COVENANT"]:
                base_score = 0.90
            elif classification in ["DEFAULT", "CONDITION"]:
                base_score = 0.70
            else:
                base_score = 0.4
        
        elif question_type == "PERMISSIBILITY":
            if classification in ["COVENANT", "CONDITION", "DEFAULT"]:
                base_score = 0.95
            elif classification in ["DEFINITIONS", "MECHANICS"]:
                base_score = 0.75
            else:
                base_score = 0.3
        
        elif question_type == "CONSEQUENCES":
            if classification in ["DEFAULT", "REMEDY", "COVENANT"]:
                base_score = 0.95
            elif classification == "CONDITION":
                base_score = 0.70
            else:
                base_score = 0.3
        
        elif question_type == "COMPREHENSIVE":
            # For comprehensive questions, all sections are important
            base_score = self._calculate_importance(classification, section.get("level", 1))
        
        else:  # GENERAL
            base_score = self._calculate_importance(classification, section.get("level", 1))
        
        # Boost score if question text matches section title keywords
        for keyword in title_lower.split():
            if len(keyword) > 3 and keyword in question.lower():
                base_score = min(1.0, base_score + 0.15)
        
        return base_score
    
    def _explain_relevance(self, section: dict, question_type: str) -> str:
        """Generate explanation for why section is relevant."""
        classification = self._classify_section_type(section.get("title", ""), section.get("content", ""))
        
        explanations = {
            "DEFINITIONAL": "Provides technical definitions and foundational concepts",
            "MECHANICAL": "Details operational mechanics and procedures",
            "PERMISSIBILITY": "Defines restrictions and exceptions (key for permission questions)",
            "CONSEQUENCES": "Outlines defaults, remedies, and enforcement actions",
            "COMPREHENSIVE": "Critical section for complete analysis"
        }
        
        base_explanation = explanations.get(question_type, "Potentially relevant section")
        return f"{base_explanation} (classified as: {classification})"
    
    def _build_search_explanation(self, question_type: str) -> str:
        """Build explanation of the search strategy."""
        strategies = {
            "DEFINITIONAL": "Starting with definitions, then sections that define or explain related concepts",
            "MECHANICAL": "Starting with primary mechanics sections, then definitions, conditions, and defaults",
            "PERMISSIBILITY": "Starting with covenant restrictions, then conditions, defaults, and all exceptions/baskets",
            "CONSEQUENCES": "Starting with default provisions, then remedies, cure rights, and related covenants",
            "COMPREHENSIVE": "Following complete structural chain: Definitions → Covenants → Defaults → Remedies → Enforcement"
        }
        return strategies.get(question_type, "General section review in priority order")


# Global instance
_credit_analyst: CreditAnalystEnhancer = None


def get_credit_analyst() -> CreditAnalystEnhancer:
    """Get or create credit analyst enhancer instance."""
    global _credit_analyst
    if _credit_analyst is None:
        _credit_analyst = CreditAnalystEnhancer()
    return _credit_analyst
