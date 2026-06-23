# if you see this message, It means we are uploading our source code and checkpoints. Please wait....

# InstructMedIR
## All-in-One Medical Image Restoration via Human-Written Instruction

### Authors: Ako Bartani, Fatemeh Daneshfar, Pietro Lio
### Published in: MICCAI2026

![Alt text](imgs/multilingual.png)

#### Medical image restoration (MedIR) is essential for enhancing image quality in clinical workflows; however, most existing approaches are tailored to individual tasks or modalities, limiting their scalability and practical applicability. Furthermore, current multi-task MedIR frameworks commonly rely on task-specific heads, routing strategies, or handcrafted conditioning mechanisms, which hinder generalization across heterogeneous modalities and reduce model interpretability. With these motivations, we propose InstructMedIR, an all-in-one MedIR that leverages human-written instructions and discrete modality mask as guiding signals to perform multiple restoration tasks across heterogeneous modalities. InstructMedIR is equipped a contrastive-based task encoder that leverages human-written instructions to specify the user's desired restoration task. In parallel, InstructMedIR uses a modality encoder to predict a modality mask of input medical image. Subsequently, both generated signals are used to guide a universal encoder-decoder-based restoration network augmented by Feature-wise Linear Modulation layers. Experimental results demonstrate outperforms across multiple MedIR tasks (e.g., denoising, super-resolution, and deblurring) and modalities (e.g., MRI, CT, PET, and X-Ray) compared to existing multi-task restoration baselines.	

![Alt text](imgs/MedIR.png)

## Requirements
```
-Python 3.8+
-PyTorch >= 1.10 (CUDA-enabled for GPU training)
-torchvision
-numpy, scipy, pillow
-opencv-python
-cikit-image
-tqdm, matplotlib
```



