# Deep Learning in Computer Vision (DLCV) Homework Reports

## DLCV HW1 Report
### Problem 1: Image Classification
- **Model A (from scratch) Accuracy:** 0.7004
- **Model B (with pre-trained weight) Accuracy:** 0.8844

### Problem 2: Self-Supervised Pre-training for Image Classification
- **SSL method used:** BYOL (Bootstrap Your Own Latent)
- **Comparison of validation accuracy in different settings:**
  - **Setting A:** 40.15%
  - **Setting B:** 46.55%
  - **Setting C:** 43.10%
  - **Setting D:** 33.99%
  - **Setting E:** 5.67%

### Problem 3: Semantic Segmentation
- **Validation mIoU for Model A (VGG16-FCN-32s):** 0.5527
- **Validation mIoU for Model B (Deeplab v3):** 0.7529

[View HW1 Report](URL-to-HW1-Report)

## DLCV HW2 Report
### Problem 1: Diffusion Models
- **Model architecture:** CFG with U-Net
- **Generated images:** 10 for each digit (0-9)
- **Reverse process visualization:** 6 images for digit "0" from t=1000 to t=0
- **Observations from implementation:**
  - **Effectiveness of denoising accelerates as noise decreases.**
  - **Use of Classifier Free Guidance to enhance conditioning capability.**

### Problem 2: DDIM
- **Face generation observation:** Images maintain high realism even as eta increases, indicating robustness of the DDIM model.

### Problem 3: DANN
- **Performance with DANN:**
  - **MNIST→SVHN:** 
    - **Trained on source:** 25.74%
    - **DANN:** 45.61%
    - **Trained on target:** 91.85%
  - **MNIST→USPS:**
    - **Trained on source:** 76.08%
    - **DANN:** 81.05%
    - **Trained on target:** 98.79%
- **t-SNE visualization:** Better domain adaptation performance for USPS compared to SVHN.
- **Conclusion from DANN:** Effective for domain adaptation, but requires fine-tuning between domain discrimination and classification accuracy.

[View HW2 Report](URL-to-HW2-Report)

## DLCV HW3 Report
### Problem 1: Zero-shot Image Classification with CLIP
- **Methods analysis:** Utilizes a contrastive learning framework.
- **Prompt-text accuracy:** 0.7232

### Problem 2: PEFT on Vision and Language Model for Image Captioning
- **Best setting performance:**
  - **CIDEr:** 0.825545172
  - **CLIPScore:** 0.723558813
  - **Parameters:** 29,175,296
- **Different attempts of PEFT:**
  - **Adapter CIDEr/CLIPScore:** 0.825545172 / 0.723558813
  - **Prefix CIDEr/CLIPScore:** 0.452588029 / 0.679535032
  - **LoRA CIDEr/CLIPScore:** 0.0000198 / 0.477362815

### Problem 3: Visualization of Attention in Image Captioning
- **Highest CLIPScore Pair:**
  - **Prediction:** "a young man sitting on a couch holding a wii controller."
  - **Score:** 1.0101318359375
- **Lowest CLIPScore Pair:**
  - **Prediction:** "a photo of a vintage car and a computer."
  - **Score:** 0.439453125

[View HW3 Report](URL-to-HW3-Report)

## DLCV HW4 Report
### Problem 1: 3D Novel View Synthesis
- **Model comprehension:** Utilizes NeRF to represent scenes as continuous functions mapping 3D points and viewing directions to colors and densities.
- **Key observations and experimentation:**
  - **PSNR:** 43.5840266 (set 1), 40.7728727 (set 2), 36.3061229 (set 3)
  - **SSIM:** 0.994262374 (set 1), 0.99131393 (set 2), 0.98403444 (set 3)
  - **LPIPS (VGG):** 0.100895444 (set 1), 0.115679097 (set 2), 0.113964383 (set 3)
- **Implementation details:** Adjusted positional encoding and MLP layers to optimize image fidelity and detail reproduction.

[View HW4 Report](URL-to-HW4-Report)
