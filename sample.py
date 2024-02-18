import torch
import matplotlib.pyplot as plt
from model import VAE
from hyperparams import get_default_hyperparams

import clip

clip_model, _ = clip.load("ViT-B/32", device="cuda")

id2label = {
    0: "airplane",
    1: "aeroplane",
    2: "plane",
    3: "automobile",
    4: "automobile",
    5: "car",
    6: "ship",
    7: "boat",
    8: "ship"
}

H = get_default_hyperparams()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(H).to(device)
vae.load_state_dict(torch.load("weights-epoch30.pt", map_location=device))

classes = ["airplane", "aeroplane", "plane", "automobile", "vehicle", "car", "ship", "boat", "ferry", ]
num_samples = 5
_, ax = plt.subplots(num_samples, 9, figsize=(num_samples*5,15))
for i in range(9):
    img = torch.rand(num_samples,32,32,3).to(device)
    label = torch.LongTensor([i]).to(device)
    label = torch.as_tensor(clip_model.encode_text(clip.tokenize(id2label[int(label)]).cuda()), dtype=torch.float32).to(device)
    recs = vae.reconstruct(img, label, k=0)
    for j in range(num_samples):
        if j == 0: ax[j,i].set_title(classes[i], fontdict={"fontsize":20})
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        ax[j,i].imshow(recs[j])
plt.savefig("result.png")
plt.show()
