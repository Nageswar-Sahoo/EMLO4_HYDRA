from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
from typing import Dict
import logging
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")


class CatDogClassifier:
    def __init__(self, model_path="model.pt"):
        print("initializing model 1")
        self.device = torch.device("cpu")
        print("model_path: ", model_path)
        # Load the traced model
        try:
            logger.info(f"Model path exists: {os.path.exists(model_path)}")

            # Load the traced model
            self.model = torch.jit.load(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Model successfully loaded and initialized.")
        except FileNotFoundError as fnf_error:
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Error details: {fnf_error}")
        except RuntimeError as rt_error:
            logger.error(f"Runtime error occurred while loading the model.")
            logger.error(f"Error details: {rt_error}")
        except Exception as e:
            logger.error("An unexpected error occurred while loading the model.")
            logger.error(f"Error details: {e}", exc_info=True)
        # Define the same transforms used during training/testing

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Class labels
        self.labels = [
            "beagle", "boxer", "bulldog", "dachshund", "german_Shepherd",
            "Golden_Retriever", "Labrador_Retriever", "Poodle", 
            "Rottweiler", "Yorkshire_Terrier"
        ]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, float]:
        if image is None:
            return None
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Create prediction dictionary
        return {
            self.labels[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }

# Create classifier instance
classifier = CatDogClassifier()

@app.get("/", response_class=HTMLResponse)
async def render_form(request: Request):
    """Renders the upload form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Handles file upload and renders predictions."""
    try:
        # Read image bytes
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Perform prediction
        predictions = classifier.predict(image)
        if predictions is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Render predictions on an HTML page
        return templates.TemplateResponse(
            "result.html", {"request": request, "predictions": predictions}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
