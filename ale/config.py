import torch


DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_CHECKPOINT_PATH = "/hdd/hdd1/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
