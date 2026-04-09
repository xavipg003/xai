from omegaconf import OmegaConf
from src.faster_rcnn.inference import inference

config = OmegaConf.load("config/config_faster.yaml")
inference(config)
