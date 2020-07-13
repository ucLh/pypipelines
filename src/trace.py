import segmentation_models_pytorch as smp
import torch

model = smp.Unet("efficientnet-b5", encoder_weights=None, classes=4, activation=None)
# model = smp.Unet("se_resnext50_32x4d", encoder_weights=None, classes=4, activation=None)
model.eval()
model.encoder.set_swish(memory_efficient=False)
ckpt = torch.load("../ckpt/efficientnetb5-v2.pth")
model.load_state_dict(ckpt["state_dict"])
sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
traced.save("../ckpt/traced_efficientnetb5-v2.pth")
print("saved")
