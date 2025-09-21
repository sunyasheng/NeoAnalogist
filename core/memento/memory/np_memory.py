"""
Non-parametric Memory System for Memento Integration
===================================================

This module implements the memory retrieval system from Memento,
adapted for integration with NeoAnalogist.

Key Features:
- BERT-based text embedding
- Cosine similarity search
- Case retrieval and storage
- Integration with NeoAnalogist's memory system
"""

import argparse
import json
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class NonParametricMemory:
    """Non-parametric memory system for case retrieval"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        max_length: int = 256,
        batch_size: int = 64
    ):
        self.logger = logging.getLogger("NonParametricMemory")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Setup device
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Memory storage
        self.cases: List[Dict[str, Any]] = []
        self.case_embeddings: Optional[torch.Tensor] = None
        self.is_loaded = False
    
    def _load_model(self):
        """Load the BERT model and tokenizer"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_cases(self, cases_path: str) -> bool:
        """Load cases from JSONL file"""
        try:
            self.logger.info(f"Loading cases from: {cases_path}")
            self.cases = self._load_jsonl(cases_path)
            self.logger.info(f"Loaded {len(self.cases)} cases")
            
            # Generate embeddings for all cases
            if self.cases:
                self._generate_case_embeddings()
                self.is_loaded = True
                return True
            else:
                self.logger.warning("No cases loaded")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load cases: {e}")
            return False
    
    def _load_jsonl(self, path: str) -> List[dict]:
        """Load JSONL file"""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception as e:
                    self.logger.warning(f"Failed to parse line {ln}, skipped: {e}")
        return items
    
    def _generate_case_embeddings(self):
        """Generate embeddings for all loaded cases"""
        if not self.cases:
            return
        
        self.logger.info("Generating case embeddings...")
        
        # Extract questions from cases
        questions = [case.get("question", "") for case in self.cases]
        
        # Generate embeddings
        self.case_embeddings = self._embed_texts(questions)
        
        self.logger.info(f"Generated embeddings for {len(questions)} cases")
    
    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of texts using the BERT model"""
        if not texts:
            return torch.empty(0, 0)
        
        vecs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, return_dict=True)
            
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                e = out.pooler_output
            else:
                e = out.last_hidden_state[:, 0, :]
            
            e = F.normalize(e, p=2, dim=1)
            vecs.append(e.cpu())
        
        return torch.cat(vecs, dim=0)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve similar cases for a given query"""
        if not self.is_loaded or not self.cases:
            self.logger.warning("No cases loaded for retrieval")
            return []
        
        try:
            # Embed the query
            query_embedding = self._embed_texts([query])[0].unsqueeze(0)
            
            # Calculate similarities
            similarities = (query_embedding @ self.case_embeddings.T).squeeze(0)
            
            # Get top-k results
            k = min(top_k, len(self.cases))
            topk_scores, topk_indices = torch.topk(similarities, k)
            
            # Format results
            results = []
            for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_indices.tolist()), 1):
                if score >= min_score:
                    case = self.cases[idx]
                    results.append({
                        "rank": rank,
                        "score": round(float(score), 6),
                        "question": case.get("question", ""),
                        "plan": case.get("plan", ""),
                        "reward": case.get("reward", 0),
                        "line_index": idx
                    })
            
            self.logger.info(f"Retrieved {len(results)} cases for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in retrieval: {e}")
            return []
    
    def add_case(self, question: str, plan: str, reward: int = 0):
        """Add a new case to memory"""
        case = {
            "question": question,
            "plan": plan,
            "reward": reward
        }
        self.cases.append(case)
        
        # Regenerate embeddings if we have the model loaded
        if self.model is not None:
            self._generate_case_embeddings()
        
        self.logger.info(f"Added new case: {question[:50]}...")
    
    def save_cases(self, output_path: str):
        """Save cases to JSONL file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for case in self.cases:
                    f.write(json.dumps(case) + "\n")
            self.logger.info(f"Saved {len(self.cases)} cases to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cases: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "total_cases": len(self.cases),
            "is_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": str(self.device),
            "has_embeddings": self.case_embeddings is not None
        }


# Legacy functions for compatibility with original Memento code
def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file (legacy function)"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception as e:
                print(f"[WARN] Failed to parse line {ln}, skipped: {e}", file=sys.stderr)
    return items


def extract_pairs(items: List[dict], key_field: str, value_field: str) -> List[Tuple[str, object, int]]:
    """Extract key-value pairs from items (legacy function)"""
    pairs = []
    for i, obj in enumerate(items):
        if key_field in obj and value_field in obj:
            pairs.append((str(obj[key_field]), obj[value_field], i))
        elif len(obj) == 2:
            ks = list(obj.keys())
            pairs.append((str(obj[ks[0]]), obj[ks[1]], i))
        else:
            pass
    return pairs


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> torch.Tensor:
    """Embed texts using BERT model (legacy function)"""
    vecs = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            e = out.last_hidden_state[:, 0, :]
        e = F.normalize(e, p=2, dim=1)  
        vecs.append(e.cpu())
    return torch.cat(vecs, dim=0)


def retrieve(
    task: str,
    pairs: List[Tuple[str, object, int]],
    tokenizer, 
    model,
    device_str: str = "auto",
    top_k: int = 5,
    max_length: int = 256,
) -> List[dict]:
    """Retrieve similar cases (legacy function)"""
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = [p[0] for p in pairs] 
    key_vecs = embed_texts(keys, tokenizer, model, device, max_length=max_length)
    query_vec = embed_texts([task], tokenizer, model, device, max_length=max_length)[0].unsqueeze(0)

    sims = (query_vec @ key_vecs.T).squeeze(0) 
    k = min(top_k, len(pairs))
    topk_scores, topk_idx = torch.topk(sims, k)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_idx.tolist()), 1):
        key, value, line_index = pairs[idx] 
        results.append(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "question": key,
                "plan": value,
                "line_index": line_index,  
            }
        )
    return results


# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory = NonParametricMemory()
    
    # Load cases
    cases_path = "core/memento/memory/cases.jsonl"
    if Path(cases_path).exists():
        memory.load_cases(cases_path)
        
        # Test retrieval
        query = "What is the capital of a county?"
        results = memory.retrieve(query, top_k=3)
        
        print(f"Query: {query}")
        print("Retrieved cases:")
        for result in results:
            print(f"  Rank {result['rank']}: {result['question']} (score: {result['score']})")
    else:
        print(f"Cases file not found: {cases_path}")
