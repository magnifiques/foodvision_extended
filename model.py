import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision import datasets, transforms

# Custom transformation to handle palette images
def convert_to_rgba(image):
    # Check if the image mode is 'P' (palette mode)
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image

def create_model(num_classes: int = 20,
                     seed: int = 42):

  # 1. Download the default weights
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT

  # 2. Setup transforms
  vit_transforms = weights.transforms()

  # Custom transforms
  custom_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally
        transforms.Lambda(convert_to_rgba),      # Apply RGBA conversion
        transforms.RandomRotation(10),           # Randomly rotate images by up to 10 degrees
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust brightness, contrast, etc.
        # transforms.ToTensor(),                   # Convert images to tensor
])

  # 3. Add the custom transforms to convert Pallete into RGBA
  combined_transforms = transforms.Compose([
        custom_transforms,  # Custom Transforms
        vit_transforms,  # Include ViT's transforms
    ])

  # 3. create a model and transforms
  model = torchvision.models.vit_b_16(weights=weights)

  # 4. Freeze the base layers in the model (this will stop all layers from training)
  for parameters in model.parameters():
    parameters.requires_grad = False

  # 5. Set seeds for reproducibility
  torch.manual_seed(seed)

  # 6. Modify the number of output layers
  model.heads = torch.nn.Sequential(
      torch.nn.Dropout(p=0.5),
      torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
  )

  return model, combined_transforms