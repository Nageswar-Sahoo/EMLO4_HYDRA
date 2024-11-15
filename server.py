import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64
from datetime import datetime

# Define labels for the specific dog breeds
DOG_BREEDS = ['beagle', 'boxer', 'bulldog', 'dachshund', 'german_shepherd',
              'golden_retriever', 'labrador_retriever', 'poodle', 'rottweiler', 'yorkshire_terrier']

class DogBreedClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        self.labels = DOG_BREEDS
        self.num_classes = len(self.labels)

        # Load a pre-trained model and adjust for 10-class output
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True, num_classes=10)
        self.model = self.model.to(device)
        self.model.eval()

        # Get model-specific transforms for preprocessing
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        
        # Decode base64 string to bytes and convert to PIL image
        img_bytes = base64.b64decode(image_bytes)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Apply transformations and move to device
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, x):
        """Perform prediction and return class probabilities"""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def encode_response(self, output):
        """Convert model output to API response"""
        responses = []
        for i in range(len(output)):  # Loop over the batch
            # Each output[i] should be a 1D tensor of shape [num_classes]
            probs, indices = torch.topk(output[i], k=1)  # Get top 1 prediction for each image
            idx = indices.item()  # Get the index of the top class
            prob = probs.item()   # Get the probability of the top class
            responses.append({
                "predictions": [
                    {
                        "label": self.labels[idx],
                        "probability": prob
                    }
                ]
            })
        return responses

if __name__ == "__main__":
    api = DogBreedClassifierAPI()
    # Configure server to use GPU acceleration if available
    server = ls.LitServer(
        api,
        accelerator="gpu",
    )
    server.run(port=8000)
