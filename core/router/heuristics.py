"""
Tier-1 Micro-Rule & Deterministic Heuristic Router
Latency Budget: < 1.0 ms
Om's Research Pillar: Hybrid Cascade Intent Routing
"""

import re
from typing import Optional, Tuple, Dict, Any

# Pre-compiled regex patterns for lightning-fast matching (<0.5ms)

# Code Patterns: syntax, markdown blocks, function definitions, common libraries
CODE_REGEX = re.compile(
    r"(```[a-zA-Z]*|def\s+[a-zA-Z_]\w*\(|function\s+\w*\(|class\s+\w+[:\{]|"
    r"import\s+[a-zA-Z_]|from\s+[a-zA-Z_].*import|#include\s*<|public\s+static\s+void|"
    r"console\.log\(|System\.out\.println|return\s+[0-9a-zA-Z_\{\[\(]|"
    r"lambda\s*.*:|npm\s+install|git\s+(commit|push|pull|checkout)|"
    r"sql\s+(select|insert|update|delete)|SELECT\s+.*\s+FROM|"
    r"\b(dockerfile|kubernetes|yaml|json|regex|regexes|asyncio|fastapi|react|vue|typescript)\b|"
    r"\b(segfault|nullpointerexception|traceback|syntaxerror|runtimeerror|indentationerror)\b|"
    r"write\s+(a\s+)?(script|function|program|class|algorithm|unit\s+test|code)|"
    r"debug\s+(this|my)?\s*(code|script|error)|fix\s+the\s+bug)",
    re.IGNORECASE
)

# Math Patterns: LaTeX, calculus symbols, equations, discrete math
MATH_REGEX = re.compile(
    r"(\\frac\{|\\sqrt\{|\\int|\\sum|\\partial|\\lim_|"
    r"\b(calculate|compute|solve|integrate|differentiate|evaluate)\b.*\b(integral|derivative|limit|equation|matrix|polynomial|eigenvalue|eigenvector|determinant)\b|"
    r"\b(pythagorean|fibonacci|euler's|riemann|fourier\s+transform|laplace\s+transform)\b|"
    r"\b(modulo|modulus|combinatorics|permutation|combination|binomial\s+theorem)\b|"
    r"\b(differential\s+equation|ode|pde|linear\s+algebra|vector\s+space|simplex|topology)\b|"
    r"\b(gcd|lcm|prime\s+factorization|isomorphic|homomorphism)\b|"
    r"(\d+\s*[\+\-\*\/]\s*\d+\s*=|\b(prove|proof\s+that)\b.*(\btheorem\b|\bequals\b|=))|"
    r"(\bsin\(|\bcos\(|\btan\(|\blog\(|\bln\(|\bexp\())",
    re.IGNORECASE
)

# Medical Patterns: clinical symptoms, pharmacology, anatomy, pathology
MEDICAL_REGEX = re.compile(
    r"\b(symptom|symptoms|syndrome|diagnos(is|es)|differential\s+diagnos(is|es)|prognos(is|es))\b|"
    r"\b(patient|patients|clinical|pathology|etiology|pathophysiology|pharmacology|pharmacokinetics)\b|"
    r"\b(diabetes|hypertension|myocardial\s+infarction|arrhythmia|ischemia|stroke|seizure|pancreatitis)\b|"
    r"\b(antibiotic|analgesic|anticoagulant|insulin|metformin|lisinopril|atorvastatin|ibuprofen)\b|"
    r"\b(dosage|mg\/kg|intravenous|subcutaneous|oral\s+dose|contraindication|contraindications|adverse\s+effects)\b|"
    r"\b(oncology|carcinoma|metastasis|chemotherapy|biopsy|histology|radiology|mri|ct\s+scan)\b|"
    r"\b(cardiology|neurology|pediatrics|endocrinology|nephrology|hematology|immunology)\b|"
    r"\b(erythrocyte|leukocyte|platelets|hemoglobin|hba1c|creatinine|troponin|bilirubin|lipase|amylase)\b|"
    r"\b(hypertensive|hypoglycemia|hyperglycemia|ketoacidosis|sepsis|tachycardia|bradycardia|epigastric)\b|"
    r"\b(acute\s+(pain|condition|abdomen)|chronic\s+disease|prescribe|prescription|medication|therapeutics)\b",
    re.IGNORECASE
)

# Science Patterns: physics laws, chemistry reactions, astronomy, thermodynamics
SCIENCE_REGEX = re.compile(
    r"\b(thermodynamics|entropy|enthalpy|gibbs\s+free\s+energy|kinetics|stoichiometry)\b|"
    r"\b(covalent\s+bond|ionic\s+bond|molarity|avogadro|periodic\s+table|orbital|valence)\b|"
    r"\b(quantum\s+mechanics|schrodinger|wavefunction|heisenberg|photoelectric|planck)\b|"
    r"\b(general\s+relativity|special\s+relativity|gravitational|spacetime|black\s+hole)\b|"
    r"\b(electromagnetism|maxwell's\s+equations|coulomb|faraday|lorentz\s+force)\b|"
    r"\b(astronomy|supernova|redshift|cosmology|hubble|dark\s+matter|dark\s+energy)\b|"
    r"\b(photosynthesis|cellular\s+respiration|krebs\s+cycle|mitochondria|dna\s+replication|rna)\b|"
    r"\b(newton's\s+laws|conservation\s+of\s+momentum|angular\s+momentum|torque|friction)\b",
    re.IGNORECASE
)

# Writing Patterns: creative composition, essays, emails, prose, style
WRITING_REGEX = re.compile(
    r"\b(write\s+(an?\s+)?(essay|poem|haiku|story|novel|article|blog\s+post|speech|letter|email))\b|"
    r"\b(paraphrase|rewrite|rephrase|proofread|critique|summarize\s+this\s+text|tone\s+of\s+voice)\b|"
    r"\b(metaphor|alliteration|rhyme|dialogue|narrative|protagonist|antagonist|plot\s+twist)\b|"
    r"\b(persuasive\s+essay|cover\s+letter|press\s+release|formal\s+tone|poetic)\b",
    re.IGNORECASE
)

def evaluate_heuristics(query: str) -> Optional[Tuple[str, float, str]]:
    """
    Evaluates tier-1 deterministic micro-rules.
    Returns: (domain, confidence, rule_description) or None if inconclusive.
    Average execution time: 0.1 - 0.4 ms.
    """
    cleaned = query.strip()
    if not cleaned:
        return ("general", 0.50, "Empty prompt fallback")

    # Priority 1: High-confidence code indicators (e.g. backticks or explicit code commands)
    if "```" in cleaned or "def " in cleaned or "class " in cleaned or "#include" in cleaned:
        return ("code", 0.98, "Code block or syntax declaration token detected")

    code_match = CODE_REGEX.search(cleaned)
    math_match = MATH_REGEX.search(cleaned)
    med_match = MEDICAL_REGEX.search(cleaned)
    sci_match = SCIENCE_REGEX.search(cleaned)
    write_match = WRITING_REGEX.search(cleaned)

    # Count signals
    signals = []
    if code_match:
        signals.append(("code", 0.93, f"Code regex match: '{code_match.group(0)}'"))
    if math_match:
        signals.append(("mathematics", 0.94, f"Math regex match: '{math_match.group(0)}'"))
    if med_match:
        signals.append(("medical", 0.96, f"Clinical/Medical regex match: '{med_match.group(0)}'"))
    if sci_match:
        signals.append(("science", 0.91, f"Science/Physics regex match: '{sci_match.group(0)}'"))
    if write_match:
        signals.append(("writing", 0.90, f"Writing/Composition regex match: '{write_match.group(0)}'"))

    # If exactly one strong rule matches with high confidence
    if len(signals) == 1:
        return signals[0]
    elif len(signals) > 1:
        # Sort by confidence
        signals.sort(key=lambda x: x[1], reverse=True)
        # If the top match clearly dominates
        return signals[0]

    return None
