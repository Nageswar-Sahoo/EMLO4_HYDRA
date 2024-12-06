import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image


class CatDogClassifier:
    """
    A classifier for predicting dog breeds using a pre-trained PyTorch model.
    """

    def __init__(self, model_path="model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._define_transforms()
        self.labels = [
            "beagle", "boxer", "bulldog", "dachshund", "german_Shepherd",
            "Golden_Retriever", "Labrador_Retriever", "Poodle", 
            "Rottweiler", "Yorkshire_Terrier"
        ]

    def _load_model(self, model_path):
        """
        Load the PyTorch JIT model and set it to evaluation mode.
        """
        model = torch.jit.load(model_path)
        model.to(self.device)
        model.eval()
        return model

    def _define_transforms(self):
        """
        Define the image transformations for preprocessing.
        """
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict(self, image):
        """
        Predict the probabilities for each dog breed given an input image.
        """
        if image is None:
            return None

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return {
            label: float(probabilities[idx])
            for idx, label in enumerate(self.labels)
        }


def create_interface(classifier):
    """
    Create a Gradio interface for the classifier.
    """
    return gr.Interface(
        fn=classifier.predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=10),
        title="Dog Breed Classifier",
        description="Upload an image to classify the breed of the dog.",
        examples=[
            ["examples/1.jpg"],
            ["examples/2.jpg"],
        ],
    )


if __name__ == "__main__":
    classifier = CatDogClassifier()
    demo = create_interface(classifier)
    demo.launch()
