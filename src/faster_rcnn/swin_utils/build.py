import timm
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torch.nn import Module
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn

class SwinBackbone(Module):
        def __init__(self, swin_model, name):
            super().__init__()
            self.patch_model = swin_model
            if name.split('_')[1] == 'base':
                self.reduce_conv = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
            else:
                # Assuming 'tiny' backbone
                self.reduce_conv = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1) #Embedding para reducir dimensiones
            self.out_channels = 256  
            
        def forward(self, x):         
            features = self.patch_model.forward_features(x).permute(0, 3, 1, 2).contiguous()
            reduced = self.reduce_conv(features) 
            return {"0": reduced} 
        
class SwinBackboneFPN(Module):
    def __init__(self, swin_model, name):
        super().__init__()
        self.swin = swin_model
        
        if name.split('_')[1] == 'base':
            in_channels_list = [128, 256, 512, 1024]
        else:
            # Assuming 'tiny' backbone
            in_channels_list = [96, 192, 384, 768]
        self.out_channels = 256 
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=None
        )
        
    def forward(self, x):
        _, intermediate_features = self.swin.forward_intermediates(
            x, indices=[0, 1, 2, 3], norm=True
        )
        features_dict = {
            f'0': intermediate_features[0],
            f'1': intermediate_features[1],
            f'2': intermediate_features[2],
            f'3': intermediate_features[3]
        }

        fpn_outputs = self.fpn(features_dict)

        return fpn_outputs

def make_swin(config):
    backbone = timm.create_model(
        config['model']['backbone_name'],
        pretrained=True,
        img_size=config['size']
    )

    if config['model']['lora']:
        lora_config = LoraConfig(
            #task_type=TaskType.FEATURE_EXTRACTION,  
            inference_mode=False,
            r=config['model']['lora_r'],         
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["qkv", "proj"]
        )


        backbone = get_peft_model(backbone, lora_config)

    if config['model']['fpn']:
        backbone = SwinBackboneFPN(backbone, config['model']['backbone_name'])
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),)*4,
            aspect_ratios=((0.5, 1.0, 2.0),)*4
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], 
            output_size=7,
            sampling_ratio=2
        )

    else:
        backbone = SwinBackbone(backbone, config['model']['backbone_name'])
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], 
            output_size=7,
            sampling_ratio=2
        )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=config['model']['nclasses'], 
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=config['size'],
        max_size=config['size'],
    )

    return model
