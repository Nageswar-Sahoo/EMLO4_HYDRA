import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix
import pandas as pd


class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        base_model: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        log_dir: str = "logs",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained model
        self.model = timm.create_model(
            base_model, pretrained=pretrained, num_classes=num_classes
        )

        # Multi-class accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # Initialize confusion matrix metrics for training and validation
        self.train_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Paths to save confusion matrix CSV files
        self.train_output_csv_path = log_dir + "train_confusion_matrix_details.csv"
        self.val_output_csv_path = log_dir   + "val_confusion_matrix_details.csv"
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        # Update confusion matrix for training
        self.train_conf_matrix.update(preds, y)
        return loss
    def on_train_epoch_end(self):
        # Compute and save confusion matrix for training at the end of each epoch
        cm_train = self.train_conf_matrix.compute().cpu().numpy()
        self.save_confusion_matrix_to_csv(cm_train, self.train_output_csv_path)

        # Reset confusion matrix for the next epoch
        self.train_conf_matrix.reset()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        # Update confusion matrix for validation
        self.val_conf_matrix.update(preds, y)
   
    def on_validation_epoch_end(self):
        # Compute and save confusion matrix for validation at the end of each epoch
        cm_val = self.val_conf_matrix.compute().cpu().numpy()
        self.save_confusion_matrix_to_csv(cm_val, self.val_output_csv_path)

        # Reset confusion matrix for the next epoch
        self.val_conf_matrix.reset()
    
    def save_confusion_matrix_to_csv(self, cm, output_csv_path):
        # Convert confusion matrix to DataFrame
        cm_df = pd.DataFrame(cm, index=range(cm.shape[0]), columns=range(cm.shape[1]))

        # Save confusion matrix to a CSV file (overwrite if file exists)
        cm_df.to_csv(output_csv_path, header=True, index=False)    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }