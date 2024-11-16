import argparse
from pathlib import Path
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import lightning as L
import hydra
from omegaconf import DictConfig , OmegaConf
from lightning.pytorch.loggers import Logger
import logging
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.catdog_classifier import CatDogClassifier
from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
import lightning as L

log = logging.getLogger(__name__)

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ['cat', 'dog']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

# Refactor model instantiation and checkpoint loading
def instantiate_model(cfg: DictConfig) -> L.LightningModule:
    log.info(f"Instantiating model <{cfg.model._target_}>")  
    
    # Dynamically instantiate the model using Hydra configuration
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Load model weights from checkpoint if provided
    if cfg.get("ckpt_path"):
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        model = model.__class__.load_from_checkpoint(cfg.ckpt_path)

    # Set the model to evaluation mode
    model.eval()
    
    return model

def remove_id(filename):
    # Use regex to match and remove the digits and underscore before the extension
    cleaned_filename = re.sub(r'_\d+', '', str(filename))
    return cleaned_filename

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path, image_files):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f}) \n Actual: {image_files.parent.name.lower()}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@task_wrapper
@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir / "infer_log.log")
    if cfg.get("infer"):
     print(OmegaConf.to_yaml(cfg))
     model = instantiate_model(cfg)
     input_folder = Path(cfg.paths.data_dir + str("/test",))
     output_folder = Path(cfg.paths.output_dir)
     output_folder.mkdir(exist_ok=True, parents=True)
     image_files = list(input_folder.glob('*/*'))
     print(image_files)
     with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file, image_file)
                
                progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

if __name__ == "__main__":
    main()