import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import litserve as ls

class SmolLM:
    def __init__(self, device):
        checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model = torch.compile(self.model)
        self.model.eval()
        
    def apply_chat_template(self, messages):
        """Convert messages to model input format"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
    
    def __call__(self, prompt):
        """Run model inference"""
        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return inputs, outputs
    
    def decode_tokens(self, outputs):
        """Decode output tokens to text"""
        inputs, generate_ids = outputs
        # Only decode the new tokens (exclude input prompt)
        new_tokens = generate_ids[:, inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

class SmolLMAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model"""
        self.model = SmolLM(device)

    def decode_request(self, request):
        """Process the incoming request"""
        if not request.messages:
            raise ValueError("No messages provided")
        return self.model.apply_chat_template(request.messages)

    def predict(self, prompt, context):
        """Generate response"""
        yield self.model(prompt)

    def encode_response(self, outputs):
        """Format the response"""
        for output in outputs:
            yield {"role": "assistant", "content": self.model.decode_tokens(output)}

if __name__ == "__main__":
    api = SmolLMAPI()
    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        accelerator="gpu",
        workers_per_device=1
    )
    server.run(port=8000)