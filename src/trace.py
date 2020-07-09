import segmentation_models_pytorch as smp
import torch

model = smp.Unet("efficientnet-b5", encoder_weights=None, classes=4, activation=None)
model.eval()
model.encoder.set_swish(memory_efficient=False)
ckpt = torch.load("./model.pth")
model.load_state_dict(ckpt["state_dict"])
sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
traced.save("traced_model.pth")
print("saved")
