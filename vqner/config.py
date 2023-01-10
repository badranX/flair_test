from dataclasses import dataclass

@dataclass
class Config:
    model_d: int = 300
    model_size: int = 400
    k: int = 20 #num of labels will be set by corpus
    k_neck: int = 60
    num_heads: int = 10
    input_hidden_size: int = 100
    input_d: int = None
