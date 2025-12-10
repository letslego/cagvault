"""
LLM-based Section Evaluator for Credit Agreement Analysis

Uses an expert credit analyst system prompt to evaluate whether document sections
fully answer user questions, with intelligent cross-reference identification and
priority section queuing for complete answers.
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


CREDIT_ANALYST_EVALUATOR_SYSTEM_PROMPT = """You are an expert credit analyst evaluating whether a document section fully answers a user's question.

    CRITICAL: You must determine if this section provides a COMPLETE answer to ALL aspects of the question, not just partial information.
    1. Credit agreements contain LAYERED provisions - a section that appears to answer a question may be modified, overridden, or incomplete without cross-references
    2. You must be HIGHLY SKEPTICAL before triggering ANSWER
    3. Check if the current section contains cross-references to OTHER sections that are essential to a complete answer
    4. For certain question types, multiple related provisions MUST be analyzed together

    DYNAMIC PLANNING:
    If you discover this section contains explicit cross-references to other sections that are ESSENTIAL to answer the question, you can request they be checked next by listing their section IDs in the 'priority_sections' field.

    Only use priority_sections when:
    - This section explicitly mentions "See Section X" or "as defined in Section Y"
    - The cross-referenced sections are MANDATORY to answer the question (per the cross-reference checks below)
    - You're confident the section IDs are valid

    DO NOT use priority_sections for:
    - Speculative references (sections you think might be helpful)
    - Sections already mentioned in previous notes
    - General exploration

    The system will automatically insert these sections next in the search queue if they haven't been checked yet.

    MANDATORY CROSS-REFERENCE CHECKS:
    Before triggering ANSWER, verify you have checked:

    - For "Who" PARTY/ROLE/ENTITY questions:
        - Must list them by name and role.
    
    - For COVENANT BREACH questions:
        - The covenant itself
        - The corresponding DEFAULT provision to understand WHO is affected and if it's "springing"
        - Any CURE RIGHTS mentioned in either section
        - Must have ALL THREE to answer completely

    - For "CAN BORROWER INCUR DEBT?" questions:
        - ALL relevant debt baskets (not just one), including:
            * General basket
            * Ratio-based basket
            * Available Amount basket
        - For SECURED debt, corresponding LIEN permissions for EACH basket
        - Must verify basket-to-lien matching is correct
        - Must note any EBITDA dependencies or conditions

    - For ACQUISITION/MERGER questions:
        - The merger covenant
        - The CHANGE OF CONTROL definition
        - The Change of Control DEFAULT provision
        - Must have ALL THREE to avoid missing overriding provisions

    - For questions involving capitalized/defined terms:
        - The relevant definitions MUST be included
        - Calculations or measurements MUST reference the definition

    Your evaluation process:
    1. Review notes from previously analyzed sections (if any) to understand what's been found so far
    2. Identify all components of the user's question (there may be multiple parts)
    3. Check THIS section for explicit cross-references to other sections
    4. Determine if this section + previous findings address EACH component completely
    5. **CRITICAL: Before deciding ANSWER, ask yourself:**
        - "Does this section reference other sections I haven't seen yet?"
        - "For this question type, what related provisions are MANDATORY to check?"
        - "Could there be overriding provisions elsewhere?"
        - "Have I verified ALL alternative paths/baskets?"
        - "If this is about debt, have I matched it to the correct lien provision?"
    6. Extract any relevant information from THIS section in notes field
    7. Decide: ANSWER (complete) or PASS (incomplete/irrelevant)
    8. When ANSWER, synthesize a complete response using information from this section AND previous notes
    9. ALWAYS cite specific sections when providing your answer

    ANSWER if:
    - Every part of the question is addressed completely (either this section or combined with previous notes)
    - All MANDATORY cross-references for this question type have been analyzed
    - No unresolved references to other sections exist
    - All alternative paths have been checked (for permissibility questions)
    - Structural dependencies are verified (e.g., debt + lien provisions match)
    - A credit analyst would consider this answer bulletproof, not just plausible
    - Your answer explicitly cites ALL sections used and notes any dependencies/conditions

    You may also ANSWER with a reasoned interpretation if:
    - You have sufficient information to make a well-supported inference
    - You clearly indicate the answer is based on interpretation rather than explicit provisions
    - You explain your reasoning and cite the sections that support your interpretation
    - You note what explicit information is missing and why your interpretation is reasonable

    PASS if:
    - Only partial information is present and you cannot reasonably infer an answer
    - The section references other sections for complete details that would materially change your interpretation
    - Critical thresholds, dates, conditions, or exceptions are missing that prevent any reasonable inference
    - This section references other sections not yet retrieved that are likely essential
    - For breach questions: haven't seen the default provision yet
    - For debt questions: haven't checked ALL baskets or verified lien matching
    - For merger questions: haven't checked Change of Control provisions

    NOTES FIELD: Capture NEW information from THIS section only. This builds context for analyzing subsequent sections. Include:
    - Key facts/provisions found
    - Cross-references to other sections mentioned
    - Gaps or unanswered components
    - Question type indicators (e.g., "Need to check defaults" or "Need other debt baskets")

   Remember: It's better to PASS and retrieve more sections than to give an incomplete answer that misses critical provisions."""


ANSWER_QUALITY_EVALUATION_PROMPT = """You are an expert credit analyst evaluating the QUALITY and COMPLETENESS of an answer about a credit agreement.

CRITICAL EVALUATION CRITERIA:
Your task is to assess whether the proposed answer is BULLETPROOF and COMPLETE, or if it misses critical provisions.

1. LAYERED PROVISIONS CHECK:
   - Does the answer account for how provisions modify, override, or limit each other?
   - Are cross-references explored and their impact integrated?
   - Could there be overriding or superior provisions elsewhere?

2. MANDATORY CROSS-REFERENCE VERIFICATION:
   For the question type, verify ALL mandatory checks were performed:
   
   - PARTY/ROLE questions: All parties listed by name and role? ✓
   - COVENANT BREACH: Covenant + DEFAULT + CURE RIGHTS all analyzed? ✓
   - DEBT INCURRENCE: ALL baskets (general, ratio, Available Amount) checked? ✓
   - SECURED DEBT: Lien permissions verified for EACH basket? ✓
   - MERGER/ACQUISITION: Merger covenant + Change of Control definition + Change of Control DEFAULT all reviewed? ✓
   - DEFINED TERMS: Definitions provided and calculations reference them? ✓

3. COMPLETENESS ASSESSMENT:
   - Does every part of the question get a complete answer?
   - Are all alternative paths/baskets covered?
   - Are conditions, thresholds, and exceptions noted?
   - Are dependencies and limitations identified?

4. CITATION RIGOR:
   - Does the answer cite specific sections?
   - Are citations accurate and precise?
   - Are there unsubstantiated claims?

5. INTELLECTUAL HONESTY:
   - Does the answer acknowledge limitations or missing information?
   - Is reasoning transparent?
   - Are inferences clearly labeled as such?

EVALUATE THIS ANSWER:
Question: {question}

Proposed Answer:
{proposed_answer}

Sections Cited:
{sections_cited}

Previous Context/Notes:
{previous_context}

EVALUATION OUTPUT (respond in JSON):
{{
    "quality_score": 0.0-1.0,
    "is_complete": true/false,
    "is_bulletproof": true/false,
    "mandatory_checks_passed": {{"check_name": true/false}},
    "missing_mandatory_checks": ["Check 1", "Check 2"] or [],
    "gaps_identified": ["Gap 1", "Gap 2"] or [],
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "recommendation": "ACCEPT" or "REVISE" or "RETRIEVE_MORE_SECTIONS",
    "required_additions": "What would make this answer more complete?",
    "explanation": "Detailed explanation of assessment"
}}

SCORING GUIDANCE:
- 0.9-1.0: Bulletproof answer, all mandatory checks passed, all sections cited
- 0.7-0.9: Good answer, most mandatory checks passed, minor gaps noted
- 0.5-0.7: Partial answer, some mandatory checks missing, significant gaps
- 0.0-0.5: Incomplete answer, major gaps, cannot be relied upon"""


class LLMSectionEvaluator:
    """Evaluates whether sections provide complete answers to questions using LLM."""
    
    def __init__(self):
        """Initialize the LLM section evaluator."""
        self.system_prompt = CREDIT_ANALYST_EVALUATOR_SYSTEM_PROMPT
        self.quality_prompt = ANSWER_QUALITY_EVALUATION_PROMPT
        logger.info("LLMSectionEvaluator initialized")
    
    def get_system_prompt(self) -> str:
        """Get the credit analyst evaluator system prompt."""
        return self.system_prompt
    
    def get_quality_evaluation_prompt(self) -> str:
        """Get the answer quality evaluation prompt."""
        return self.quality_prompt
    
    def create_evaluation_prompt(
        self,
        question: str,
        section_title: str,
        section_content: str,
        previous_notes: Optional[str] = None,
        already_checked_sections: Optional[List[str]] = None
    ) -> str:
        """
        Create a structured prompt for LLM to evaluate a section against a question.
        
        Args:
            question: The user's question
            section_title: Title of the current section
            section_content: Content of the current section
            previous_notes: Notes from previously analyzed sections
            already_checked_sections: List of section titles already checked
            
        Returns:
            Formatted prompt for LLM evaluation
        """
        already_checked = ""
        if already_checked_sections:
            already_checked = f"\nSections already analyzed:\n" + "\n".join(f"  - {s}" for s in already_checked_sections)
        
        previous_context = ""
        if previous_notes:
            previous_context = f"\nPrevious findings from other sections:\n{previous_notes}"
        
        prompt = f"""QUESTION: {question}

CURRENT SECTION: {section_title}

SECTION CONTENT:
{section_content}
{previous_context}{already_checked}

EVALUATION TASK:
1. Does this section contain information relevant to the question?
2. Does it provide COMPLETE answers to ALL aspects of the question?
3. Are there explicit cross-references to other sections needed for a complete answer?
4. What mandatory cross-reference checks apply to this question type?
5. Should we retrieve additional sections before answering?

OUTPUT FORMAT (respond in JSON):
{{
    "decision": "ANSWER" or "PASS",
    "confidence": 0.0-1.0,
    "relevant": true/false,
    "notes": "Key information found, cross-references identified, gaps noted",
    "priority_sections": ["Section ID 1", "Section ID 2"] or [],
    "reasoning": "Explain your decision",
    "answer": "Complete answer if decision is ANSWER, otherwise null",
    "citations": ["Section name 1", "Section name 2"] if ANSWER
}}"""
        
        return prompt
    
    def extract_section_references(self, section_content: str) -> List[str]:
        """
        Extract explicit cross-references from section content.
        
        Looks for patterns like "Section X", "See X", "as defined in Section X".
        
        Args:
            section_content: The section content to analyze
            
        Returns:
            List of section references found
        """
        import re
        
        references = []
        
        # Pattern: "Section X" or "section X"
        section_pattern = r'[Ss]ection\s+(\d+(?:\.\d+)*(?:\s*\([a-zA-Z]\))?)'
        matches = re.findall(section_pattern, section_content)
        references.extend(matches)
        
        # Pattern: "See X" where X is a section reference
        see_pattern = r'[Ss]ee\s+(?:[Ss]ection\s+)?(\d+(?:\.\d+)*(?:\s*\([a-zA-Z]\))?)'
        matches = re.findall(see_pattern, section_content)
        references.extend(matches)
        
        # Pattern: "as defined in Section X"
        defined_pattern = r'[Aa]s\s+defined\s+in\s+(?:[Ss]ection\s+)?(\d+(?:\.\d+)*(?:\s*\([a-zA-Z]\))?)'
        matches = re.findall(defined_pattern, section_content)
        references.extend(matches)
        
        # Remove duplicates and return
        return list(set(references))
    
    def should_pass_on_missing_references(
        self,
        question: str,
        section_type: str,
        found_references: List[str],
        previous_notes: str
    ) -> tuple[bool, str]:
        """
        Determine if section should be PASS'd due to missing mandatory cross-references.
        
        Args:
            question: The user's question
            section_type: Classification of the current section (COVENANT, DEFAULT, etc.)
            found_references: References found in this section
            previous_notes: Notes from previously analyzed sections
            
        Returns:
            (should_pass, reason_message)
        """
        q_lower = question.lower()
        
        # Breach question - need covenant, default, and cure rights
        if "breach" in q_lower or "default" in q_lower or "violate" in q_lower:
            if section_type == "COVENANT" and "DEFAULT" not in previous_notes:
                return True, "Need to check corresponding DEFAULT provision for complete analysis"
            if section_type == "DEFAULT" and "COVENANT" not in previous_notes:
                return True, "Need to check the covenant itself to understand the breach mechanism"
            if "cure" in q_lower and "CURE" not in previous_notes:
                return True, "Need to check CURE RIGHTS provisions"
        
        # Debt question - need all baskets
        if "incur" in q_lower and ("debt" in q_lower or "indebtedness" in q_lower):
            if "General basket" not in previous_notes:
                return True, "Need to check ALL debt baskets (general, ratio-based, Available Amount)"
            if "lien" in q_lower and "lien permission" not in previous_notes:
                return True, "Need to verify lien permissions for each debt basket"
        
        # Merger/Acquisition question - need all three components
        if any(term in q_lower for term in ["merger", "acquisition", "change of control"]):
            if not all(term in previous_notes for term in ["Merger", "Change of Control", "DEFAULT"]):
                return True, "Need to check: merger covenant, Change of Control definition, and Change of Control default"
        
        # Defined terms - need definitions
        if "what is" in q_lower or "define" in q_lower or "means" in q_lower:
            if section_type != "DEFINITIONS" and "DEFINITIONS" not in previous_notes:
                return True, "Need to check DEFINITIONS section for capitalized terms"
        
        return False, ""
    
    def get_mandatory_cross_references(self, question: str, section_type: str) -> List[str]:
        """
        Get mandatory section types that must be analyzed for this question.
        
        Args:
            question: The user's question
            section_type: Classification of current section
            
        Returns:
            List of mandatory section types to check
        """
        q_lower = question.lower()
        mandatory = []
        
        # Covenant breach questions
        if "breach" in q_lower or "default" in q_lower:
            if section_type == "COVENANT":
                mandatory.extend(["DEFAULT", "REMEDY", "DEFINITIONS"])
            elif section_type == "DEFAULT":
                mandatory.extend(["COVENANT", "REMEDY"])
        
        # Debt questions
        if "incur" in q_lower and ("debt" in q_lower or "indebtedness" in q_lower):
            mandatory.extend(["DEFINITIONS", "CONDITION"])
            if "lien" in q_lower or "secured" in q_lower:
                mandatory.append("DEBT_LIEN")
        
        # Merger/Acquisition
        if any(term in q_lower for term in ["merger", "acquisition", "change of control"]):
            mandatory.extend(["CHANGE_OF_CONTROL", "DEFAULT", "COVENANT"])
        
        # Definition questions
        if "what is" in q_lower or "define" in q_lower:
            mandatory.append("DEFINITIONS")
        
        # Remove duplicates
        return list(set(mandatory))
    
    def create_quality_evaluation_prompt(
        self,
        question: str,
        proposed_answer: str,
        sections_cited: List[str],
        previous_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for LLM to evaluate answer quality and completeness.
        
        Args:
            question: The user's question
            proposed_answer: The proposed answer to evaluate
            sections_cited: List of sections cited in the answer
            previous_context: Previous findings and context
            
        Returns:
            Formatted prompt for quality evaluation
        """
        sections_str = "\n".join(f"  - {s}" for s in sections_cited) if sections_cited else "  None"
        context_str = previous_context or "No previous context"
        
        return self.quality_prompt.format(
            question=question,
            proposed_answer=proposed_answer,
            sections_cited=sections_str,
            previous_context=context_str
        )


def get_llm_evaluator() -> LLMSectionEvaluator:
    """Get or create LLM section evaluator instance."""
    global _evaluator
    if '_evaluator' not in globals():
        _evaluator = LLMSectionEvaluator()
    return _evaluator


# Claude Skill functions

def evaluate_section_completeness(
    question: str,
    section_title: str,
    section_content: str,
    previous_notes: Optional[str] = None,
    already_checked_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Claude Skill: Evaluate whether a section provides a complete answer to a question.
    
    Uses expert credit analyst framework to determine if additional sections are needed.
    
    Args:
        question: The user's question
        section_title: Title of the section being evaluated
        section_content: Full content of the section
        previous_notes: Notes from previously analyzed sections
        already_checked_sections: Titles of sections already analyzed
        
    Returns:
        Evaluation result with decision, reasoning, and priority sections for retrieval
    """
    evaluator = get_llm_evaluator()
    
    # Extract references from content
    references = evaluator.extract_section_references(section_content)
    
    # Build context
    context = {
        "question": question,
        "section_title": section_title,
        "section_length": len(section_content),
        "explicit_references_found": references,
        "previous_notes": previous_notes or "None",
        "already_checked": already_checked_sections or []
    }
    
    # Note: In actual implementation, this would call the LLM with the system prompt
    # For now, return the evaluation prompt and context for use with Claude
    
    return {
        "evaluation_context": context,
        "system_prompt": evaluator.get_system_prompt(),
        "evaluation_prompt": evaluator.create_evaluation_prompt(
            question,
            section_title,
            section_content,
            previous_notes,
            already_checked_sections
        ),
        "message": "Provide this system prompt and evaluation_prompt to Claude for analysis"
    }


def extract_cross_references(section_content: str) -> List[str]:
    """
    Claude Skill: Extract explicit cross-references from a section.
    
    Args:
        section_content: The section content to analyze
        
    Returns:
        List of section references found
    """
    evaluator = get_llm_evaluator()
    return evaluator.extract_section_references(section_content)


def get_mandatory_checks(question: str, section_type: str) -> Dict[str, Any]:
    """
    Claude Skill: Get mandatory cross-reference checks for a question type.
    
    Args:
        question: The user's question
        section_type: Classification of the current section
        
    Returns:
        Mandatory sections and checks required for complete answer
    """
    evaluator = get_llm_evaluator()
    
    mandatory = evaluator.get_mandatory_cross_references(question, section_type)
    
    return {
        "question": question,
        "section_type": section_type,
        "mandatory_sections": mandatory,
        "guidance": "These section types must be analyzed together for a complete answer"
    }


def evaluate_answer_quality(
    question: str,
    proposed_answer: str,
    sections_cited: List[str],
    previous_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Claude Skill: Evaluate the quality and completeness of a proposed answer.
    
    Uses expert credit analyst framework to assess whether an answer is bulletproof
    and addresses all mandatory cross-reference requirements.
    
    Args:
        question: The user's question
        proposed_answer: The proposed answer to evaluate
        sections_cited: List of sections cited in the answer
        previous_context: Previous findings and context from other sections
        
    Returns:
        Quality evaluation with scores, gaps identified, and improvement recommendations
    """
    evaluator = get_llm_evaluator()
    
    # Create quality evaluation prompt
    quality_prompt = evaluator.create_quality_evaluation_prompt(
        question=question,
        proposed_answer=proposed_answer,
        sections_cited=sections_cited,
        previous_context=previous_context
    )
    
    return {
        "question": question,
        "system_prompt": evaluator.get_system_prompt(),
        "quality_evaluation_prompt": quality_prompt,
        "instruction": "Send system_prompt + quality_evaluation_prompt to Claude for quality assessment",
        "sections_evaluated": sections_cited,
        "context_used": previous_context is not None
    }

