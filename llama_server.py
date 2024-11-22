import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, LlamaForCausalLM
import litserve as ls
precision = torch.bfloat16

class LlamaModel:
    def __init__(self, device):
        checkpoint = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the desired LLaMA checkpoint
        self.device = device
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)  # LLaMA models often require `use_fast=False`
        if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token if available
        self.model = model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).to(precision)
        self.model.eval()
        
    def apply_chat_template(self, messages):
        """Convert messages to model input format"""
        #print(messages)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )    
    
    def __call__(self, prompt):
        """Run model inference"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device).to(precision)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return inputs, outputs
    
    def decode_tokens(self, outputs):
        """Decode output tokens to text"""
        inputs, generate_ids = outputs
        new_tokens = generate_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

class LlamaAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model"""
        self.model = LlamaModel(device)

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
        #print(outputs)

        for output in outputs:
            yield {"role": "assistant", "content": self.model.decode_tokens(output)}

if __name__ == "__main__":
    api = LlamaAPI()
    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        accelerator="gpu",
        workers_per_device=2
    )
    server.run(port=8000)
