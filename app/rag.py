import os
import re
import json
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
        # Using Llama-3-8B-Instruct or Mistral-7B-Instruct
        model_id = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
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
            max_new_tokens=150
        )
        print("Model loaded successfully.")

    def analyze(self, user_prompt: str) -> dict:
        # Retrieve context
        docs = self.db.similarity_search(user_prompt, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        sys_prompt = """You are an expert legal assistant specialized in the California Consumer Privacy Act (CCPA).
You will be provided with a description of a business practice and relevant excerpts from the CCPA statute.
Determine if the business practice violates the CCPA.
Focus specifically on these key sections:
- Section 1798.100 (Must disclose what data you collect)
- Section 1798.105 (Must honor deletion requests)
- Section 1798.120 (Must allow opt-out of data selling; minors need consent)
- Section 1798.125 (Cannot discriminate/charge more for opting out)

Respond ONLY with valid JSON in this exact format, with no markdown formatting or backticks:
{"harmful": true, "articles": ["Section 1798.100"]}
If there is no violation, "harmful" MUST be false and "articles" MUST be [].

Excerpts:
""" + context + """

Business Practice:
""" + user_prompt + """

Respond ONLY with valid JSON.
"""
        
        # Adjust for format
        messages = [
            {"role": "user", "content": sys_prompt}
        ]
        
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipe(prompt, max_new_tokens=50, do_sample=False)
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        # Clean up in case there's markdown
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                # Enforce structure
                output = {
                    "harmful": bool(result.get("harmful", False)),
                    "articles": result.get("articles", [])
                }
                # If harmful is false, ensure articles is empty
                if not output["harmful"]:
                    output["articles"] = []
                # If harmful is true, ensure articles is not empty, fallback to a catchall
                elif output["harmful"] and len(output["articles"]) == 0:
                    output["articles"] = ["Section 1798.100"]  # Fallback
            except Exception:
                # parsing failed, fallback safely
                output = {"harmful": False, "articles": []}
        else:
            output = {"harmful": False, "articles": []}
            
        return output

# Singleton-like instance pattern
_analyzer = None
def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = CCPAAnalyzer()
    return _analyzer
