import segmentation_models_pytorch as smp
import torch
from pytorch_seg_models import Unet as Unet2

# model = smp.Unet("efficientnet-b7", encoder_weights=None, classes=4, activation=None,
#                  aux_params={'classes': 2, 'dropout': 0.5})
# model = Unet2("se_resnext50_32x4d", encoder_weights=None, classes=4, activation='softmax')
model = smp.FPN("efficientnet-b0", encoder_weights=None, classes=4, activation=None,
                aux_params={'classes': 4, 'dropout': 0.5})
model.eval()
model.encoder.set_swish(memory_efficient=False)
ckpt = torch.load("../ckpt/effnetb0_fpn_seg_v3.pth")
model.load_state_dict(ckpt["state_dict"])
sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
traced.save("../ckpt/traced_effnetb0_fpn_seg_v3.pth")
print("saved")
