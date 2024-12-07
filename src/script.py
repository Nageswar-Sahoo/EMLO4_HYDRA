import os
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig
import torch
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

def download_from_s3(bucket_name: str, s3_key: str, download_path: str) -> None:
    """
    Download a file from an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 object key (path in the bucket).
        download_path (str): Local file path where the downloaded file will be saved.

    Returns:
        None
    """
    s3_client = boto3.client("s3")
    log.info(f"Attempting to download from S3. Bucket: {bucket_name}, Key: {s3_key}")
    log.info(f"Downloading {s3_key} from S3 bucket {bucket_name} to {download_path}...")

    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        log.info(f"Object metadata: {response}")
        log.info(f"Downloading {s3_key} from S3 bucket {bucket_name} to {download_path}...")
        s3_client.download_file(bucket_name, s3_key , download_path)
        log.info(f"File downloaded successfully to {download_path}")
    except s3_client.exceptions.NoSuchKey:
        log.error(f"The object key '{s3_key}' does not exist in bucket '{bucket_name}'.")
        raise
    except (NoCredentialsError, ClientError) as e:
        log.error(f"Failed to download file from S3: {e}")
        raise
 
@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Download checkpoint from S3 if specified
    local_ckpt_path = Path("best-checkpoint.ckpt")
    if cfg.get("s3"):
        s3_bucket = cfg.s3.bucket_name
        s3_key = os.path.join(cfg.s3.key_prefix, "best-checkpoint.ckpt")
        download_from_s3(s3_bucket, s3_key, str(local_ckpt_path))
        cfg.ckpt_path = str(local_ckpt_path)  # Update checkpoint path to local file

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint if specified
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint: {local_ckpt_path}")
        checkpoint = torch.load(local_ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    # Set model to eval mode
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224)  # Assuming standard image input size
    
    # Trace the model
    log.info("Tracing model...")
    traced_model = model.to_torchscript(method="trace", example_inputs=example_input)
    
    # Create output directory if it doesn't exist
    output_dir = Path("traced_models")
    output_dir.mkdir(exist_ok=True)
    
    # Save the traced model
    output_path = output_dir / "model.pt"
    torch.jit.save(traced_model, output_path)
    log.info(f"Traced model saved to: {output_path}")

if __name__ == "__main__":
    main()
