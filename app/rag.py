import os
import re
import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Keywords that strongly indicate CCPA-compliant/safe behavior
SAFE_INDICATORS = [
    "clear privacy policy",
    "allows customers to opt out",
    "allow customers to opt out",
    "opt out of data selling at any time",
    "deleted all personal data",
    "within 45 days",
    "verified request",
    "consumer's verified request",
    "do not sell my personal information",
    "equal service and pricing",
    "regardless of whether they exercise",
    "privacy rights",
    "as required",
    "compliant",
    "comply with",
    "in compliance",
]

# Keywords that strongly indicate CCPA violation
HARMFUL_INDICATORS = [
    "without informing",
    "without giving them a chance to opt out",
    "without opt-out",
    "without consent",
    "without getting",
    "doesn't mention",
    "does not mention",
    "ignoring their request",
    "ignoring the request",
    "keeping all records",
    "charge customers who opted out",
    "higher price",
    "selling personal data of",
    "selling our customers' personal information",
    "without telling",
    "without notifying",
    "refuse to delete",
    "refusing to delete",
    "not honoring",
    "discriminat",
]

# Section mapping based on topic keywords
SECTION_MAPPING = {
    "1798.100": [
        "disclose", "disclosure", "privacy policy", "doesn't mention",
        "does not mention", "what data", "collect", "browsing history",
        "geolocation", "biometric", "undisclosed", "not mention",
        "inform", "transparency"
    ],
    "1798.105": [
        "delete", "deletion", "removing", "erase", "right to delete",
        "ignoring their request", "keeping all records", "refuse to delete",
        "honor deletion"
    ],
    "1798.120": [
        "opt out", "opt-out", "selling", "sell", "data broker",
        "third-party", "third party", "minor", "14-year",
        "children", "parental consent", "do not sell",
        "personal information to"
    ],
    "1798.125": [
        "discriminat", "higher price", "charge more", "penaliz",
        "different price", "different level", "deny goods",
        "pricing", "financial incentive", "opted out"
    ]
}


def detect_relevant_sections(text: str) -> list:
    """Detect which CCPA sections are potentially violated based on keyword matching."""
    text_lower = text.lower()
    sections = []
    for section, keywords in SECTION_MAPPING.items():
        for kw in keywords:
            if kw in text_lower:
                section_str = f"Section {section}"
                if section_str not in sections:
                    sections.append(section_str)
                break
    return sections


def is_likely_safe(text: str) -> bool:
    """Check if the text strongly indicates safe/compliant behavior."""
    text_lower = text.lower()
    # Check if the prompt is completely unrelated to CCPA/privacy
    privacy_keywords = [
        "data", "privacy", "personal information", "consumer",
        "opt out", "opt-out", "sell", "delete", "collect",
        "information", "rights", "ccpa"
    ]
    is_privacy_related = any(kw in text_lower for kw in privacy_keywords)
    if not is_privacy_related:
        return True  # Completely unrelated = safe
    
    # Check for safe indicators
    safe_score = sum(1 for ind in SAFE_INDICATORS if ind in text_lower)
    harmful_score = sum(1 for ind in HARMFUL_INDICATORS if ind in text_lower)
    
    if safe_score > 0 and harmful_score == 0:
        return True
    return False


def is_likely_harmful(text: str) -> bool:
    """Check if the text strongly indicates a CCPA violation."""
    text_lower = text.lower()
    harmful_score = sum(1 for ind in HARMFUL_INDICATORS if ind in text_lower)
    safe_score = sum(1 for ind in SAFE_INDICATORS if ind in text_lower)
    
    if harmful_score > 0 and safe_score == 0:
        return True
    return False


class CCPAAnalyzer:
    def __init__(self):
        print("Initializing RAG logic...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Vector Store
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.load_local(
            "faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Load Model
        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        hf_token = os.environ.get("HF_TOKEN")
        
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            max_new_tokens=100
        )
        print("Model loaded successfully.")

    def analyze(self, user_prompt: str) -> dict:
        # --- STEP 1: Pre-classification using keyword heuristics ---
        if is_likely_safe(user_prompt):
            print(f"[PRE-CLASSIFY] Safe (keyword match)")
            return {"harmful": False, "articles": []}
        
        if is_likely_harmful(user_prompt):
            sections = detect_relevant_sections(user_prompt)
            if sections:
                print(f"[PRE-CLASSIFY] Harmful (keyword match) -> {sections}")
                return {"harmful": True, "articles": sections}
        
        # --- STEP 2: RAG-based LLM analysis for ambiguous cases ---
        docs = self.db.similarity_search(user_prompt, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        sys_prompt = f"""You are a legal compliance checker for the California Consumer Privacy Act (CCPA).

Analyze the following business practice and determine if it VIOLATES the CCPA.

KEY RULES:
- A practice is HARMFUL (violates CCPA) ONLY if it describes a business doing something WRONG or ILLEGAL
- A practice is SAFE (does not violate CCPA) if the business is following proper procedures  
- A practice is SAFE if it is completely unrelated to data privacy
- Being compliant, following the law, or describing proper behavior is NOT a violation

CCPA Sections:
- Section 1798.100: Businesses must disclose what personal data they collect
- Section 1798.105: Businesses must honor consumer data deletion requests  
- Section 1798.120: Consumers can opt out of sale of their data; minors need affirmative consent
- Section 1798.125: Businesses cannot discriminate against consumers who exercise privacy rights

EXAMPLES:
Practice: "We sell customer data without telling them" -> {{"harmful": true, "articles": ["Section 1798.120"]}}
Practice: "We deleted user data within 45 days after their request" -> {{"harmful": false, "articles": []}}
Practice: "Can we schedule a meeting for Monday?" -> {{"harmful": false, "articles": []}}

RELEVANT CCPA TEXT:
{context}

BUSINESS PRACTICE TO ANALYZE:
{user_prompt}

Respond with ONLY valid JSON. No other text:"""

        messages = [{"role": "user", "content": sys_prompt}]
        
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipe(prompt, max_new_tokens=80, do_sample=False)
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        print(f"[LLM RAW] {generated_text}")
        
        # Parse JSON from LLM output
        match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                harmful = bool(result.get("harmful", False))
                articles = result.get("articles", [])
                
                # Ensure articles are properly formatted
                formatted_articles = []
                for art in articles:
                    art_str = str(art)
                    # Standardize format
                    section_match = re.search(r'1798\.\d+', art_str)
                    if section_match:
                        formatted_articles.append(f"Section {section_match.group()}")
                
                if harmful:
                    if not formatted_articles:
                        # Use keyword-based section detection as fallback
                        formatted_articles = detect_relevant_sections(user_prompt)
                        if not formatted_articles:
                            formatted_articles = ["Section 1798.100"]
                    return {"harmful": True, "articles": formatted_articles}
                else:
                    return {"harmful": False, "articles": []}
                    
            except json.JSONDecodeError:
                pass
        
        # If LLM failed to produce JSON, use keyword classification
        if is_likely_harmful(user_prompt):
            sections = detect_relevant_sections(user_prompt)
            return {"harmful": True, "articles": sections if sections else ["Section 1798.100"]}
        
        return {"harmful": False, "articles": []}


# Singleton-like instance pattern
_analyzer = None
def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = CCPAAnalyzer()
    return _analyzer
