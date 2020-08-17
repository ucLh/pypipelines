import segmentation_models_pytorch as smp
import torch


def load_ckpt(path_to_ckpt):
    # model = smp.FPN("efficientnet-b0", encoder_weights=None, classes=4, activation=None,
    #                 aux_params={'classes': 4, 'dropout': 0.75})
    # # model.eval()
    # model.encoder.set_swish(memory_efficient=False)
    ckpt = torch.load(path_to_ckpt)
    return dict(ckpt["state_dict"])


params1 = load_ckpt("../ckpt/effnetb0_final_stage/model.pth")
params2 = load_ckpt("../ckpt/effnetb0_final_stage/epoch_150.pth")
params3 = load_ckpt("../ckpt/effnetb0_final_stage/epoch_135.pth")
params4 = load_ckpt("../ckpt/effnetb0_final_stage/epoch_145.pth")
params5 = load_ckpt("../ckpt/effnetb0_final_stage/epoch_155.pth")


for name1 in params1:
    if name1 in params2:
        params2[name1].data.copy_(0.2 * params1[name1].data + 0.2 * params2[name1].data)

for name1 in params3:
    if name1 in params2:
        params2[name1].data.copy_(0.2 * params1[name1].data + params2[name1].data)

for name1 in params4:
    if name1 in params2:
        params2[name1].data.copy_(0.2 * params1[name1].data + params2[name1].data)

for name1 in params5:
    if name1 in params2:
        params2[name1].data.copy_(0.2 * params1[name1].data + params2[name1].data)

model = smp.FPN("efficientnet-b0", encoder_weights=None, classes=4, activation=None,
                    aux_params={'classes': 4, 'dropout': 0.75})
model.eval()
model.encoder.set_swish(memory_efficient=False)
model.load_state_dict(params2)
torch.save(params2, "../ckpt/effnetb0_final_stage/effnetb0_averaged.pth")

sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
traced.save(f"../ckpt/effnetb0_final_stage/traced_effnetb0_averaged.pth")
print("saved")
