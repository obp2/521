import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from skimage.segmentation import slic
from skimage.transform import resize
from skimage.color import gray2rgb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

imagenet_path = './imagenet_samples'

# List of image file paths
image_paths = os.listdir(imagenet_path)

def lime_explanation(input_image, model, target_class, n_segments=50, n_samples=100):
    segments = slic(input_image, n_segments=n_segments, compactness=10)
    superpixels = np.unique(segments)
    masks = []
    preds = []
    for _ in range(n_samples):
        mask = np.random.choice([0,1], size=superpixels.shape)
        perturbed = input_image.copy()
        for i, val in enumerate(superpixels):
            if mask[i] == 0:
                perturbed[segments == val] = np.mean(input_image, axis=(0,1))
        pert_tens = preprocess(Image.fromarray(perturbed.astype('uint8')))
        pert_tens = pert_tens.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pert_tens)
            prob = torch.softmax(out,dim=1)[0, target_class].item()
        masks.append(mask)
        preds.append(prob)
    masks = np.array(masks)
    preds = np.array(preds)
    reg = LinearRegression().fit(masks, preds)
    importances = reg.coef_
    explanation_mask = np.zeros_like(segments, dtype=float)
    for i, val in enumerate(superpixels):
        explanation_mask[segments == val] = importances[i]
    explanation_mask = (explanation_mask - explanation_mask.min()) / (explanation_mask.max() - explanation_mask.min() + 1e-8)
    return explanation_mask, importances

def smoothgrad_explanation(input_tensor, model, target_class, n_samples=50, noise_level=0.2):
    input_tensor = input_tensor.unsqueeze(0).to(device)
    grads = []
    for _ in range(n_samples):
        noisy = input_tensor + noise_level * torch.randn_like(input_tensor)
        noisy.requires_grad_()
        out = model(noisy)
        prob = out[0, target_class]
        model.zero_grad()
        prob.backward(retain_graph=True)
        grad = noisy.grad.detach().cpu().numpy()
        grads.append(np.abs(grad))
        noisy.grad = None
    avg_grad = np.mean(np.stack(grads), axis=0)[0]
    saliency = np.max(avg_grad, axis=0)
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
    return saliency

for img_file in image_paths:
    img_full_path = os.path.join(imagenet_path, img_file)
    input_image = Image.open(img_full_path).convert('RGB')
    input_np = np.array(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # use GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # get label
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]
    print(f"Predicted label: {predicted_synset} ({predicted_label})")

    img_file_base = img_file.replace('.JPEG', '')
    # LIME
    lime_mask, lime_importances = lime_explanation(input_np, model, predicted_idx)
    plt.imshow(input_np)
    plt.imshow(lime_mask, cmap='jet', alpha=0.5)
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.savefig(f'lime_{img_file_base}.png')
    plt.close()
    print(f"LIME explanation saved as lime_{img_file_base}.png")

    # SmoothGrad
    smoothgrad_map = smoothgrad_explanation(input_tensor, model, predicted_idx)
    plt.imshow(input_np)
    plt.imshow(smoothgrad_map, cmap='jet', alpha=0.5)
    plt.title('SmoothGrad Explanation')
    plt.axis('off')
    plt.savefig(f'smoothgrad_{img_file_base}.png')
    plt.close()
    print(f"SmoothGrad explanation saved as smoothgrad_{img_file_base}.png")

    # Comparison
    # (HxWx3)lime_mask , convert to grayscale first:
    if lime_mask.ndim == 3 and lime_mask.shape[2] == 3:
        lime_mask_gray = np.mean(lime_mask, axis=2)
    else:
        lime_mask_gray = lime_mask
    lime_mask_resized = resize(
        lime_mask_gray,
        (224, 224),  # (model input size)
        preserve_range=True,
        anti_aliasing=True
    )

    lime_flat = lime_mask_resized.flatten()
    smoothgrad_flat = smoothgrad_map.flatten()
    spearman = spearmanr(lime_flat, smoothgrad_flat)
    kendall = kendalltau(lime_flat, smoothgrad_flat)
    print(f"Spearman correlation: {spearman.correlation}")
    print(f"Kendall correlation: {kendall.correlation}")
