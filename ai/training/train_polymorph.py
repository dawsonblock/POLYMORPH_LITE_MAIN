"""
Training pipeline for Polymorph Discovery v1.0 model.

Trains a deep learning model to detect and classify polymorphs from Raman spectra.
Includes data loading from PostgreSQL, training loop with checkpointing,
validation metrics, and model export.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from retrofitkit.db.session import SessionLocal
from retrofitkit.db.models.workflow import WorkflowExecution
from retrofitkit.db.models.sample import Sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Architecture
    input_size: int = 900  # Typical Raman spectrum size
    hidden_size: int = 256
    num_classes: int = 10  # Number of polymorph types
    dropout: float = 0.3
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Paths
    data_dir: Path = Path("data/training")
    checkpoint_dir: Path = Path("ai/checkpoints")
    model_output_dir: Path = Path("ai/models")
    
    # Misc
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RamanDataset(Dataset):
    """
    PyTorch Dataset for Raman spectra from database.
    """
    
    def __init__(
        self, 
        spectra: List[np.ndarray], 
        labels: List[int],
        metadata: Optional[List[Dict]] = None
    ):
        self.spectra = spectra
        self.labels = labels
        self.metadata = metadata or [{}] * len(spectra)
        
    def __len__(self) -> int:
        return len(self.spectra)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.FloatTensor(self.spectra[idx]).unsqueeze(0)  # Add channel dim
        label = torch.LongTensor([self.labels[idx]])[0]
        return spectrum, label


class PolymorphCNN(nn.Module):
    """
    CNN-LSTM model for polymorph classification from Raman spectra.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flat_size = 128 * (config.input_size // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PolymorphTrainer:
    """Main training class for Polymorph Discovery model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = PolymorphCNN(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def load_data_from_db(self, db: Session) -> Tuple[List, List, List]:
        """
        Load training data from PostgreSQL database.
        
        Returns:
            Tuple of (spectra, labels, metadata)
        """
        logger.info("Loading data from database...")
        
        # Query workflow executions with Raman data
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.results.isnot(None)
        ).all()
        
        spectra = []
        labels = []
        metadata = []
        
        for execution in executions:
            if execution.results and 'raman_spectrum' in execution.results:
                spectrum_data = execution.results['raman_spectrum']
                intensities = spectrum_data.get('intensities', [])
                
                if len(intensities) == self.config.input_size:
                    # Normalize spectrum
                    intensities = np.array(intensities)
                    intensities = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
                    
                    spectra.append(intensities)
                    
                    # Label from metadata or default to 0
                    label = execution.results.get('polymorph_type', 0)
                    labels.append(int(label))
                    
                    metadata.append({
                        'execution_id': execution.id,
                        'timestamp': execution.start_ts,
                        'operator': execution.operator_email
                    })
        
        logger.info(f"Loaded {len(spectra)} spectra from database")
        return spectra, labels, metadata
        
    def create_dataloaders(
        self, 
        spectra: List, 
        labels: List, 
        metadata: List
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split data and create train/val/test dataloaders."""
        
        # Shuffle data
        indices = np.random.permutation(len(spectra))
        spectra = [spectra[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]
        
        # Calculate split sizes
        n_total = len(spectra)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)
        
        # Split data
        train_data = RamanDataset(
            spectra[:n_train],
            labels[:n_train],
            metadata[:n_train]
        )
        val_data = RamanDataset(
            spectra[n_train:n_train+n_val],
            labels[n_train:n_train+n_val],
            metadata[n_train:n_train+n_val]
        )
        test_data = RamanDataset(
            spectra[n_train+n_val:],
            labels[n_train+n_val:],
            metadata[n_train+n_val:]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_data, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        test_loader = DataLoader(
            test_data, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        logger.info(f"Created dataloaders: train={len(train_data)}, "
                   f"val={len(val_data)}, test={len(test_data)}")
        
        return train_loader, val_loader, test_loader
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for spectra, labels in train_loader:
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectra)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(spectra)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def export_model(self, version: str = "1.0.0"):
        """Export trained model for deployment."""
        model_path = self.config.model_output_dir / f"polymorph_detector_v{version}.pt"
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Update version registry
        self.update_version_registry(version, model_path)
        
        logger.info(f"Exported model: {model_path}")
        
    def update_version_registry(self, version: str, model_path: Path):
        """Update model_version.json with new version."""
        registry_path = Path("ai/model_version.json")
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        # Add new version
        registry["current_version"] = version
        registry["models"][version] = {
            "model_file": model_path.name,
            "created_at": datetime.now().isoformat(),
            "training_config": {
                "architecture": "CNN-LSTM",
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "num_classes": self.config.num_classes,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate
            },
            "performance": {
                "train_accuracy": self.history['train_acc'][-1] if self.history['train_acc'] else 0,
                "val_accuracy": self.history['val_acc'][-1] if self.history['val_acc'] else 0,
                "final_train_loss": self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                "final_val_loss": self.history['val_loss'][-1] if self.history['val_loss'] else 0
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
        logger.info(f"Updated version registry: {version}")
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        test_loader: DataLoader
    ):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save checkpoint if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                
        # Final evaluation on test set
        test_loss, test_acc = self.validate(test_loader)
        logger.info(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return self.history


def main():
    """Main training script."""
    # Configuration
    config = TrainingConfig()
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Initialize trainer
    trainer = PolymorphTrainer(config)
    
    # Load data from database
    db = SessionLocal()
    try:
        spectra, labels, metadata = trainer.load_data_from_db(db)
        
        if len(spectra) < 100:
            logger.warning(f"Only {len(spectra)} samples available - consider collecting more data")
            
        # Create dataloaders
        train_loader, val_loader, test_loader = trainer.create_dataloaders(
            spectra, labels, metadata
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader, test_loader)
        
        # Export model
        trainer.export_model(version="1.0.0")
        
        logger.info("Training complete!")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
