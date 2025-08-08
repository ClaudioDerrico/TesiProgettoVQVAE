import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt


class CalciumTrainer:
    """
    Simplified trainer for calcium imaging VQ-VAE models.
    Focuses only on reconstruction quality without behavior prediction.
    """
    
    def __init__(self, model, train_loader, test_loader, device='cuda', 
                 save_dir='./results', experiment_name=None):
        """
        Args:
            model: VQ-VAE model
            train_loader: training dataloader  
            test_loader: test dataloader
            device: device for training
            save_dir: directory to save results
            experiment_name: name for the experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Setup directories
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"calcium_vqvae_{int(time.time())}"
        self.experiment_name = experiment_name
        
        # Initialize tracking
        self.metrics = defaultdict(list)
        self.best_test_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Setup logging
        self.writer = SummaryWriter(f'{save_dir}/logs/{experiment_name}')
        
        print(f"Trainer initialized for experiment: {experiment_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Device: {device}")
    
    def train(self, num_epochs=100, learning_rate=1e-4, patience=20, 
              gradient_clip=1.0, save_best=True, eval_interval=5):
        """
        Main training loop.
        
        Args:
            num_epochs: maximum number of epochs
            learning_rate: initial learning rate
            patience: early stopping patience
            gradient_clip: gradient clipping value
            save_best: whether to save best model
            eval_interval: interval for detailed evaluation
        """
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Setup loss function
        reconstruction_loss = nn.MSELoss()
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}")
        print(f"Patience: {patience}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(self.optimizer, reconstruction_loss, gradient_clip)
            
            # Test phase
            test_metrics = self._test_epoch(reconstruction_loss)
            
            # Learning rate scheduling
            scheduler.step(test_metrics['total_loss'])
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, test_metrics)
            
            # Early stopping check
            if test_metrics['total_loss'] < self.best_test_loss:
                self.best_test_loss = test_metrics['total_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if save_best:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Detailed evaluation
            if (epoch + 1) % eval_interval == 0:
                self._detailed_evaluation(epoch)
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                total_time = time.time() - start_time
                self._print_progress(epoch, train_metrics, test_metrics, epoch_time, total_time)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best test loss: {self.best_test_loss:.6f} at epoch {self.best_epoch + 1}")
                break
        
        # Final evaluation
        print("\nTraining completed. Running final evaluation...")
        final_results = self._final_evaluation()
        
        # Save final results
        self._save_results(final_results)
        
        return final_results
    
    def _train_epoch(self, optimizer, recon_loss_fn, gradient_clip):
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'total_loss': 0.0,
            'perplexity': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                neural_data = batch[0].to(self.device)  # Take only neural data
            else:
                neural_data = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass - assuming your model returns (vq_loss, recon, perplexity, ...)
            outputs = self.model(neural_data)
            vq_loss, neural_recon, perplexity = outputs[0], outputs[1], outputs[2]
            
            # Reconstruction loss
            recon_loss = recon_loss_fn(neural_recon, neural_data)
            
            # Total loss (standard VQ-VAE formulation)
            total_loss = recon_loss + 0.25 * vq_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Update metrics
            epoch_metrics['recon_loss'] += recon_loss.item()
            epoch_metrics['vq_loss'] += vq_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['perplexity'] += perplexity.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _test_epoch(self, recon_loss_fn):
        """Test for one epoch."""
        self.model.eval()
        
        epoch_metrics = {
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'total_loss': 0.0,
            'perplexity': 0.0
        }
        
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    neural_data = batch[0].to(self.device)
                else:
                    neural_data = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(neural_data)
                vq_loss, neural_recon, perplexity = outputs[0], outputs[1], outputs[2]
                
                # Losses
                recon_loss = recon_loss_fn(neural_recon, neural_data)
                total_loss = recon_loss + 0.25 * vq_loss
                
                # Update metrics
                epoch_metrics['recon_loss'] += recon_loss.item()
                epoch_metrics['vq_loss'] += vq_loss.item()
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['perplexity'] += perplexity.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _detailed_evaluation(self, epoch):
        """Run detailed evaluation with reconstruction quality metrics."""
        print(f"\nDetailed evaluation at epoch {epoch + 1}...")
        
        self.model.eval()
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, (list, tuple)):
                    neural_data = batch[0].to(self.device)
                else:
                    neural_data = batch.to(self.device)
                
                outputs = self.model(neural_data)
                neural_recon = outputs[1]
                
                all_original.append(neural_data.cpu().numpy())
                all_reconstructed.append(neural_recon.cpu().numpy())
        
        original = np.concatenate(all_original, axis=0)
        reconstructed = np.concatenate(all_reconstructed, axis=0)
        
        # Calculate reconstruction quality metrics
        mse = mean_squared_error(original.flatten(), reconstructed.flatten())
        mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
        
        # Per-neuron correlations
        correlations = []
        for neuron_idx in range(original.shape[1]):
            orig_neuron = original[:, neuron_idx, :].flatten()
            recon_neuron = reconstructed[:, neuron_idx, :].flatten()
            corr, _ = pearsonr(orig_neuron, recon_neuron)
            correlations.append(corr if not np.isnan(corr) else 0)
        
        mean_correlation = np.mean(correlations)
        good_neurons = np.sum(np.array(correlations) > 0.5)
        good_neurons_pct = (good_neurons / len(correlations)) * 100
        
        # Log to tensorboard
        self.writer.add_scalar('Detailed/MSE', mse, epoch)
        self.writer.add_scalar('Detailed/MAE', mae, epoch)
        self.writer.add_scalar('Detailed/Mean_Correlation', mean_correlation, epoch)
        self.writer.add_scalar('Detailed/Good_Neurons_Pct', good_neurons_pct, epoch)
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean correlation: {mean_correlation:.3f}")
        print(f"  Good neurons (>0.5 corr): {good_neurons}/{len(correlations)} ({good_neurons_pct:.1f}%)")
    
    def _log_metrics(self, epoch, train_metrics, test_metrics):
        """Log metrics to tensorboard and internal tracking."""
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in test_metrics.items():
            self.writer.add_scalar(f'Test/{key}', value, epoch)
        
        # Log learning rate
        for param_group in self.optimizer.param_groups:
            self.writer.add_scalar('Training/LearningRate', param_group['lr'], epoch)
            break
        
        # Store in internal tracking
        for key, value in train_metrics.items():
            self.metrics[f'train_{key}'].append(value)
        
        for key, value in test_metrics.items():
            self.metrics[f'test_{key}'].append(value)
    
    def _print_progress(self, epoch, train_metrics, test_metrics, epoch_time, total_time):
        """Print training progress."""
        print(f"\nEpoch [{epoch+1:3d}] ({epoch_time:.1f}s, total: {total_time/60:.1f}min)")
        print(f"  Train | Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.1f}")
        print(f"  Test  | Recon: {test_metrics['recon_loss']:.4f}, "
              f"VQ: {test_metrics['vq_loss']:.4f}, "
              f"Total: {test_metrics['total_loss']:.4f}")
        print(f"  Best Test Loss: {self.best_test_loss:.4f} (epoch {self.best_epoch+1}), "
              f"Patience: {self.patience_counter}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_loss,
            'metrics': dict(self.metrics)
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
            torch.save(checkpoint, save_path)
        
        # Always save latest
        latest_path = os.path.join(self.save_dir, f'{self.experiment_name}_latest.pth')
        torch.save(checkpoint, latest_path)
    
    def _final_evaluation(self):
        """Run comprehensive final evaluation."""
        results = {
            'training_metrics': dict(self.metrics),
            'best_epoch': self.best_epoch,
            'best_test_loss': self.best_test_loss
        }
        
        # Load best model for evaluation
        best_path = os.path.join(self.save_dir, f'{self.experiment_name}_best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation on test set
        test_metrics = self._evaluate_reconstruction_quality()
        results['final_test_metrics'] = test_metrics
        
        # Codebook usage analysis if available
        if hasattr(self.model, 'get_codebook_usage'):
            codebook_stats = self.model.get_codebook_usage()
            results['codebook_usage'] = codebook_stats
            print(f"  Codebook usage: {codebook_stats.get('usage_percentage', 'N/A'):.1f}%")
        
        print(f"\nFinal Results:")
        print(f"  Best epoch: {self.best_epoch + 1}")
        print(f"  Best test loss: {self.best_test_loss:.6f}")
        print(f"  Final MSE: {test_metrics['mse']:.6f}")
        print(f"  Final mean correlation: {test_metrics['mean_correlation']:.3f}")
        
        return results
    
    def _evaluate_reconstruction_quality(self):
        """Comprehensive reconstruction quality evaluation."""
        self.model.eval()
        
        all_original = []
        all_reconstructed = []
        total_vq_loss = 0
        total_perplexity = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, (list, tuple)):
                    neural_data = batch[0].to(self.device)
                else:
                    neural_data = batch.to(self.device)
                
                outputs = self.model(neural_data)
                vq_loss, neural_recon, perplexity = outputs[0], outputs[1], outputs[2]
                
                all_original.append(neural_data.cpu().numpy())
                all_reconstructed.append(neural_recon.cpu().numpy())
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1
        
        original = np.concatenate(all_original, axis=0)
        reconstructed = np.concatenate(all_reconstructed, axis=0)
        
        # Calculate metrics
        mse = mean_squared_error(original.flatten(), reconstructed.flatten())
        mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
        
        # Per-neuron correlations
        correlations = []
        for neuron_idx in range(original.shape[1]):
            orig_neuron = original[:, neuron_idx, :].flatten()
            recon_neuron = reconstructed[:, neuron_idx, :].flatten()
            corr, _ = pearsonr(orig_neuron, recon_neuron)
            correlations.append(corr if not np.isnan(corr) else 0)
        
        mean_correlation = np.mean(correlations)
        good_neurons = np.sum(np.array(correlations) > 0.5)
        good_neurons_pct = (good_neurons / len(correlations)) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'mean_correlation': mean_correlation,
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'good_neurons': good_neurons,
            'good_neurons_pct': good_neurons_pct,
            'vq_loss': total_vq_loss / num_batches,
            'perplexity': total_perplexity / num_batches,
            'per_neuron_correlations': correlations,
            'reconstruction_examples': {
                'originals': original[:2],  # First 2 samples for visualization
                'reconstructions': reconstructed[:2]
            }
        }
        
        return metrics
    
    def _save_results(self, results):
        """Save final results and create visualizations."""
        import json
        import pickle
        
        # Save results as pickle (full data)
        pickle_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save basic results as JSON
        json_results = {
            'best_epoch': int(results['best_epoch']),
            'best_test_loss': float(results['best_test_loss']),
            'final_mse': float(results['final_test_metrics']['mse']),
            'final_mean_correlation': float(results['final_test_metrics']['mean_correlation']),
            'final_good_neurons_pct': float(results['final_test_metrics']['good_neurons_pct'])
        }
        
        json_path = os.path.join(self.save_dir, f'{self.experiment_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self._plot_training_curves(results['training_metrics'])
        self._plot_reconstruction_examples(results['final_test_metrics']['reconstruction_examples'])
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
    
    def _plot_training_curves(self, metrics):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(metrics['train_total_loss']) + 1)
        
        # Reconstruction Loss
        axes[0, 0].plot(epochs, metrics['train_recon_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(epochs, metrics['test_recon_loss'], label='Test', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reconstruction Loss')
        axes[0, 0].set_title('Reconstruction Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # VQ Loss
        axes[0, 1].plot(epochs, metrics['train_vq_loss'], label='Train', alpha=0.7)
        axes[0, 1].plot(epochs, metrics['test_vq_loss'], label='Test', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VQ Loss')
        axes[0, 1].set_title('Vector Quantization Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[1, 0].plot(epochs, metrics['train_total_loss'], label='Train', alpha=0.7)
        axes[1, 0].plot(epochs, metrics['test_total_loss'], label='Test', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perplexity
        axes[1, 1].plot(epochs, metrics['train_perplexity'], label='Train', alpha=0.7)
        axes[1, 1].plot(epochs, metrics['test_perplexity'], label='Test', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Codebook Perplexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, f'{self.experiment_name}_training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Training curves: {plot_path}")
    
    def _plot_reconstruction_examples(self, examples):
        """Plot reconstruction examples."""
        originals = examples['originals']
        reconstructions = examples['reconstructions']
        
        n_examples = min(2, originals.shape[0])
        n_neurons_to_show = min(20, originals.shape[1])
        
        fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_examples):
            # Select most active neurons
            neuron_vars = np.var(originals[i], axis=1)
            top_neurons = np.argsort(neuron_vars)[-n_neurons_to_show:]
            
            # Original
            im1 = axes[i, 0].imshow(originals[i, top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 0].set_ylabel(f'Example {i+1}\nNeurons')
            axes[i, 0].set_xlabel('Time')
            if i == 0:
                axes[i, 0].set_title('Original')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Reconstruction
            im2 = axes[i, 1].imshow(reconstructions[i, top_neurons, :], 
                                   aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 1].set_xlabel('Time')
            if i == 0:
                axes[i, 1].set_title('Reconstruction')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, f'{self.experiment_name}_reconstructions.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Reconstruction examples: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_test_loss = checkpoint.get('best_test_loss', float('inf'))
        self.metrics = defaultdict(list, checkpoint.get('metrics', {}))
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best test loss: {self.best_test_loss:.6f}")
        
        return checkpoint
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()


# Utility function for easy training
def train_vqvae(model, train_loader, test_loader, config=None):
    """
    Simple function to train a VQ-VAE model.
    
    Args:
        model: VQ-VAE model
        train_loader: training dataloader
        test_loader: test dataloader  
        config: training configuration dict
    
    Returns:
        tuple: (trainer, results)
    """
    default_config = {
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'patience': 20,
        'gradient_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './results',
        'experiment_name': None
    }
    
    if config:
        default_config.update(config)
    
    trainer = CalciumTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=default_config['device'],
        save_dir=default_config['save_dir'],
        experiment_name=default_config['experiment_name']
    )
    
    results = trainer.train(
        num_epochs=default_config['num_epochs'],
        learning_rate=default_config['learning_rate'],
        patience=default_config['patience'],
        gradient_clip=default_config['gradient_clip']
    )
    
    return trainer, results