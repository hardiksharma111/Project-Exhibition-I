"""
Itiksha's Research Pillar: Evaluation Framework & Standard Benchmark Datasets
Curates representative domain queries mapped to GSM8k, PubMedQA, HumanEval, and ARC.
"""

from typing import Dict, List, Any

BENCHMARK_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    "mathematics": [
        {
            "id": "math-gsm8k-01",
            "benchmark_source": "GSM8k",
            "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell in April and May combined?",
            "ground_truth": "72",
            "difficulty": "Easy",
            "key_steps": ["Calculate May sales: 48 / 2 = 24", "Add April and May: 48 + 24 = 72"]
        },
        {
            "id": "math-math-02",
            "benchmark_source": "MATH Subset",
            "prompt": "Find all real solutions to the equation: 2x^2 - 7x + 3 = 0. Show the quadratic formula steps.",
            "ground_truth": "x = 3, x = 0.5",
            "difficulty": "Medium",
            "key_steps": ["Discriminant: b^2 - 4ac = 49 - 24 = 25", "x = (7 +/- 5) / 4"]
        },
        {
            "id": "math-calc-03",
            "benchmark_source": "Calculus Benchmark",
            "prompt": "Evaluate the indefinite integral: int( (3x^2 + 2x) * e^(x^3 + x^2) ) dx.",
            "ground_truth": "e^(x^3 + x^2) + C",
            "difficulty": "Hard",
            "key_steps": ["Substitute u = x^3 + x^2, du = (3x^2 + 2x)dx", "int(e^u du) = e^u + C"]
        }
    ],
    "medical": [
        {
            "id": "med-pubmed-01",
            "benchmark_source": "PubMedQA",
            "prompt": "Does intensive glycemic control reduce microvascular complications in patients with newly diagnosed type 2 diabetes mellitus?",
            "ground_truth": "Yes. Clinical trials (UKPDS) prove reduction in retinopathy, nephropathy, and neuropathy.",
            "difficulty": "Medium",
            "key_steps": ["Identify UKPDS landmark findings", "Verify microvascular vs macrovascular outcomes"]
        },
        {
            "id": "med-medqa-02",
            "benchmark_source": "MedQA (USMLE)",
            "prompt": "A 58-year-old male presents with sudden-onset crushing substernal chest pain radiating to the left jaw. ECG reveals ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
            "ground_truth": "Right Coronary Artery (RCA) - Inferior Wall STEMI",
            "difficulty": "Hard",
            "key_steps": ["Recognize inferior wall leads II, III, aVF", "Map inferior myocardial perfusion to RCA"]
        }
    ],
    "code": [
        {
            "id": "code-humaneval-01",
            "benchmark_source": "HumanEval/1",
            "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> List[str]` that splits balanced parenthesis groups.",
            "ground_truth": "Returns balanced groups using open/close counter.",
            "difficulty": "Medium",
            "key_steps": ["Iterate characters", "Track depth balance", "Group balanced tokens"]
        },
        {
            "id": "code-humaneval-02",
            "benchmark_source": "HumanEval/12",
            "prompt": "Write a Python function `longest(strings: List[str]) -> Optional[str]` that returns the longest string from a list, or None if empty.",
            "ground_truth": "max(strings, key=len, default=None)",
            "difficulty": "Easy",
            "key_steps": ["Check empty list condition", "Find maximum length entry"]
        }
    ],
    "science": [
        {
            "id": "sci-arc-01",
            "benchmark_source": "ARC Challenge",
            "prompt": "Explain why light refracts and bends towards the normal when passing from air into a dense glass prism.",
            "ground_truth": "Phase velocity of light decreases in optical medium with higher refractive index (Snell's Law).",
            "difficulty": "Medium",
            "key_steps": ["Refractive index definition", "Wavefront deceleration in denser optical medium"]
        }
    ],
    "writing": [
        {
            "id": "write-synth-01",
            "benchmark_source": "Stylistic & Synthesis Benchmark",
            "prompt": "Compose an executive brief summarizing how localized SLMs enhance data sovereignty for enterprise compliance.",
            "ground_truth": "Structured summary covering GDPR/HIPAA compliance, zero third-party egress, and cost efficiency.",
            "difficulty": "Medium",
            "key_steps": ["Data sovereignty", "Regulatory isolation", "Total Cost of Ownership"]
        }
    ],
    "general": [
        {
            "id": "gen-mmlu-01",
            "benchmark_source": "MMLU / General Reasoning",
            "prompt": "Compare centralized cloud AI data pipelines against distributed edge computing in terms of latency, bandwidth, and privacy.",
            "ground_truth": "Comparative analysis of cloud scalability vs edge zero-latency and data privacy.",
            "difficulty": "Easy",
            "key_steps": ["Bandwidth efficiency", "Air-gap security", "Latency constraints"]
        }
    ]
}
