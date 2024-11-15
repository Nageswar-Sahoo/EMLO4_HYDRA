import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64
from datetime import datetime
precision = torch.bfloat16

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
        self.model = self.model.to(device).to(precision)
        self.model.eval()

        # Get model-specific transforms for preprocessing
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        return image_bytes
    
    def batch(self, inputs):
        """Process and batch multiple inputs"""
        batched_tensors = []
        for image_bytes in inputs:
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image_bytes)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            # Transform image to tensor
            tensor = self.transforms(image)
            batched_tensors.append(tensor)
            
        # Stack all tensors into a batch
        return torch.stack(batched_tensors).to(self.device).to(precision)

    @torch.no_grad()
    def predict(self, x):
        """Run inference on the input batch"""
        outputs = self.model(x)  # Model output is of shape [batch_size, num_classes]
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        return probabilities
    
    def unbatch(self, output):
        """Split batch output into individual predictions"""
        return [output[i] for i in range(len(output))]

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
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  # Adjust based on your GPU memory and requirements
        batch_timeout=0.01,  # Timeout in seconds to wait for forming batches
        workers_per_device=4
    )
    server.run(port=8000)
