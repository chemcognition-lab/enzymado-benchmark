# -----------------------------------------------------------------------------
# Made By Jose Manuel Barraza Chavez @ ChemicalCognitionLab @ University of Toronto (2025)
# Finding better representations for enzymes!
# Protein Language Model Embedders Factory
# For more crappy code go to: @ChemistryPlus on GitHub
# -----------------------------------------------------------------------------

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional
# -----------------------------------------------------------------------------
# SAFE IMPORTS
# -----------------------------------------------------------------------------
try:
    from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
except ImportError:
    pass 

try:
    from esm.models.esmc import ESMC
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    pass


'''
IF YOU ARE LOST 
WORRY NO MORE! CHECK THE DOCSTRING OF THE FUNCTION AT THE BOTTOM!
'''

class BaseEmbedder(ABC):
    """Abstract base class for protein embedders."""
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def get_embedding(self, sequence: str) -> np.ndarray:
        pass

# -----------------------------------------------------------------------------
# 1. ESMC Family (EvolutionaryScale - Open)
# -----------------------------------------------------------------------------
class ESMCEmbedder(BaseEmbedder):
    def _load_model(self):
        print(f"Loading ESMC model: {self.model_name}...")
        self.client = ESMC.from_pretrained(self.model_name).to(self.device)

    def get_embedding(self, sequence: str) -> np.ndarray:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.client.encode(protein)
        output = self.client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        return output.embeddings.detach().cpu().numpy().mean(axis=1).squeeze()

# -----------------------------------------------------------------------------
# 2. ESM3 Family (Strictly Open Version)
# -----------------------------------------------------------------------------
class ESM3Embedder(BaseEmbedder):
    def _load_model(self):
        # HARDCODED: Only one open model exists.
        print(f"Loading ESM3 model: {self.model_name} (Fixed Open Version)...")
        self.model = ESM3.from_pretrained(self.model_name).to(self.device)

    def get_embedding(self, sequence: str) -> np.ndarray:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        # Force return_embeddings=True (No generation)
        output = self.model.forward_and_sample(
            protein_tensor, 
            ESM3.Config(return_embeddings=True)
        )
        return output.embeddings.detach().cpu().numpy().mean(axis=1).squeeze()

# -----------------------------------------------------------------------------
# 3. ProsT5 Family (RostLab)
# -----------------------------------------------------------------------------
class ProsT5Embedder(BaseEmbedder):
    def _load_model(self):
        print(f"Loading ProsT5 model: {self.model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, sequence: str) -> np.ndarray:
        # T5 Pre-processing
        spaced_seq = " ".join(list(sequence)).replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        ids = self.tokenizer.batch_encode_plus(
            [spaced_seq], add_special_tokens=True, padding="longest", return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            embedding_repr = self.model(ids.input_ids, attention_mask=ids.attention_mask)
        
        return embedding_repr.last_hidden_state.detach().cpu().numpy()[0, :len(sequence)].mean(axis=0)

# -----------------------------------------------------------------------------
# 4. ProFluent-E1 Family
# -----------------------------------------------------------------------------
class ProFluentE1Embedder(BaseEmbedder):
    def _load_model(self):
        print(f"Loading ProFluent-E1 model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def get_embedding(self, sequence: str) -> np.ndarray:
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # E1 typically returns last_hidden_state directly or in tuple
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs[0]
            
        return embeddings.detach().cpu().numpy().mean(axis=1).squeeze()

# -----------------------------------------------------------------------------
# FACTORY FUNCTION
# -----------------------------------------------------------------------------
def get_embedder(family: str, model_name: str = None) -> BaseEmbedder:
    """
    Function to instantiate a protein language model embedder.

    Args:
        family (str): The model family key. Case-insensitive.
        model_name (str, optional): The specific HuggingFace/Repo identifier. 
                                    If None, the defaults (marked with *) are used.
                                    NOTE: For 'esm3', this argument is IGNORED 
                                    to ensure usage of the only open-source model.

    ---------------------------------------------------------------------------------------
    REFERENCE CHEATSHEET: VALID ARGUMENTS & DEFAULTS
    ---------------------------------------------------------------------------------------
    
    1. FAMILY: "esmc" (EvolutionaryScale - Open Weights)
       -------------------------------------------------
       | Model Name Argument | Params | Description                                       |
       |---------------------|--------|---------------------------------------------------|
       | "esmc_300m" (*)     | 300M   | Fastest inference. Recommended baseline.          |
       | "esmc_600m"         | 600M   | Better performance, slightly slower.              |

    2. FAMILY: "esm3" (EvolutionaryScale - Generative)
       -------------------------------------------------
       | Model Name Argument | Params | Description                                       |
       |---------------------|--------|---------------------------------------------------|
       | [IGNORED]           | 1.4B   | Always forces "esm3_sm_open_v1" (Only Open option)|

    3. FAMILY: "prost5" (RostLab - Encoder-Decoder)
       -------------------------------------------------
       | Model Name Argument                     | VRAM  | Description                      |
       |-----------------------------------------|-------|----------------------------------|
       | "Rostlab/prot_t5_xl_half_uniref50-enc"(*)| ~6GB | STANDARD. FP16 precision (Best). |
       | "Rostlab/prot_t5_xl_uniref50"           | ~12GB | Full precision.                  |

    4. FAMILY: "profluent" (ProFluent-E1)
       -------------------------------------------------
       | Model Name Argument                     | Type  | Description                      |
       |-----------------------------------------|-------|----------------------------------|
       | "Profluent-Bio/E1-150m" (*)             | Small | Lightest E1 model.               |
       | "Profluent-Bio/E1-300m"                 | Mid   | Balanced performance.            |
       | "Profluent-Bio/E1-600m"                 | Large | Best accuracy in E1 family.      |
    ---------------------------------------------------------------------------------------
    """
    family = family.lower()
    
    if family == "esmc":
        name = model_name if model_name else "esmc_300m"
        return ESMCEmbedder(name)
    
    elif family == "esm3":
        # LOGIC CHANGE: We enforce the only open model.
        # Even if the user requests something else, we ignore it to prevent crashes.
        target_model = "esm3_sm_open_v1"
        if model_name and model_name != target_model:
            print(f"Warning: Custom models for ESM3 are not supported in this lab script. Forcing '{target_model}'.")
        return ESM3Embedder(target_model)
        
    elif family == "prost5" or family == "prott5":
        name = model_name if model_name else "Rostlab/prot_t5_xl_half_uniref50-enc"
        return ProsT5Embedder(name)
        
    elif family == "profluent" or family == "e1":
        name = model_name if model_name else "Profluent-Bio/E1-150m" 
        return ProFluentE1Embedder(name)
        
    else:
        raise ValueError(f"Family '{family}' not recognized. Check docstring for options.")