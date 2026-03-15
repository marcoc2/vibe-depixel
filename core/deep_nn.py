"""
Deep Neural Network for Pixel Art Depixelization

Based on the approach by Diego Inacio:
https://github.com/diegoinacio/creative-coding-notebooks/blob/master/ML-and-AI/pixel-art-depixelization-deepNN.ipynb

The main idea: Given 2D coordinates as input, predict the relative pixel color as output.
Uses one-hot encoding on the color palette to avoid color interpolation and blurry results.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from PIL import Image

# TensorFlow import with lazy loading
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")


class DeepNNDepixelizer:
    """
    Deep Neural Network for depixelizing pixel art.
    
    The network learns to map 2D coordinates to colors from a discrete palette,
    using one-hot encoding to preserve sharp color boundaries.
    """
    
    # Default network architecture layers
    DEFAULT_ARCHITECTURE = [
        (8, 'relu'),
        (32, 'relu'),
        (32, 'relu'),
        (128, 'relu'),
        (128, 'relu'),
        (512, 'relu'),
    ]
    
    def __init__(
        self,
        architecture: list = None,
        dropout_rate: float = 0.125,
        learning_rate: float = 0.001,
        verbose: bool = True
    ):
        """
        Initialize the Deep NN Depixelizer.
        
        Args:
            architecture: List of (units, activation) tuples for hidden layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Adam optimizer learning rate
            verbose: Print training progress
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DeepNNDepixelizer")
        
        self.architecture = architecture or self.DEFAULT_ARCHITECTURE
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Model and data attributes
        self.model: Optional[tf.keras.Model] = None
        self.color_palette: Optional[np.ndarray] = None
        self.palette_size: int = 0
        self.input_shape: Tuple[int, int, int] = (0, 0, 3)
        
        # Training history
        self.history: Dict[str, list] = {}
    
    def _get_color_palette(self, image: np.ndarray) -> np.ndarray:
        """
        Extract unique colors from image and sort by frequency.
        
        Args:
            image: RGB image array normalized to [0, 1]
            
        Returns:
            Sorted color palette array
        """
        n1, n2, c = image.shape
        
        # Reshape to get all pixels
        pixels = image.reshape(-1, c)
        
        # Get unique colors
        color_palette = np.unique(pixels, axis=0)
        
        # Sort by luminance (sum of RGB values)
        luminance = color_palette.sum(axis=1)
        sorted_indices = np.argsort(luminance)
        color_palette = color_palette[sorted_indices]
        
        return color_palette
    
    def _one_hot_encode(
        self, 
        image: np.ndarray, 
        color_palette: np.ndarray
    ) -> np.ndarray:
        """
        Convert image pixels to one-hot encoded vectors based on color palette.
        
        Args:
            image: RGB image array normalized to [0, 1]
            color_palette: Array of unique colors
            
        Returns:
            One-hot encoded array of shape (num_pixels, palette_size)
        """
        n1, n2, c = image.shape
        pixels = image.reshape(-1, c)
        palette_size = len(color_palette)
        
        # Create one-hot encoded array
        one_hot = np.zeros((len(pixels), palette_size), dtype=np.float32)
        
        # Vectorized one-hot encoding
        for i, color in enumerate(color_palette):
            # Find all pixels matching this color
            matches = np.all(pixels == color, axis=1)
            one_hot[matches, i] = 1.0
        
        return one_hot
    
    def _create_coordinates(
        self, 
        height: int, 
        width: int,
        standardize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create standardized coordinate grids.
        
        Args:
            height: Image height
            width: Image width
            standardize: Whether to standardize coordinates (mean=0, std=1)
            
        Returns:
            Tuple of (S, T) coordinate arrays
        """
        T, S = np.mgrid[0:height, 0:width]
        S = S.astype(np.float32)
        T = T.astype(np.float32)
        
        if standardize:
            S = (S - S.mean()) / (S.std() + 1e-8)
            T = (T - T.mean()) / (T.std() + 1e-8)
        
        return S, T
    
    def _build_model(self, input_dim: int = 2, output_dim: int = 2) -> tf.keras.Model:
        """
        Build the neural network model.
        
        Architecture inspired by autoencoder decoder structure.
        Output uses softmax activation for one-hot encoded colors.
        
        Args:
            input_dim: Input dimension (2 for S, T coordinates)
            output_dim: Output dimension (size of color palette)
            
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers with dropout for regularization
        for i, (units, activation) in enumerate(self.architecture):
            model.add(tf.keras.layers.Dense(units, activation=activation))
            
            # Add dropout after certain layers
            if i in [1, 4]:  # After 2nd and 5th layers
                model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Output layer with softmax for one-hot encoding
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        image: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        validation_split: float = 0.1,
        upscale_factor: int = 16,
        early_stopping: bool = True,
        patience: int = 50
    ) -> Dict:
        """
        Train the neural network on the input image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            upscale_factor: Target upscale factor for test prediction
            early_stopping: Enable early stopping
            patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            image = np.asarray(image) / 255.0
        
        # Store input shape
        self.input_shape = image.shape
        height, width, channels = image.shape
        
        if self.verbose:
            print(f"Input image shape: {height}x{width}x{channels}")
        
        # Extract color palette
        self.color_palette = self._get_color_palette(image)
        self.palette_size = len(self.color_palette)
        
        if self.verbose:
            print(f"Color palette size: {self.palette_size} unique colors")
        
        # Create training coordinates
        S, T = self._create_coordinates(height, width)
        
        # Prepare training data
        # X: coordinates (S, T) for each pixel
        X_train = np.stack([S.ravel(), T.ravel()], axis=1).astype(np.float32)
        
        # Y: one-hot encoded colors
        Y_train_oh = self._one_hot_encode(image, self.color_palette)
        
        if self.verbose:
            print(f"Training samples: {len(X_train)}")
            print(f"Building model with {self.palette_size} output classes...")
        
        # Build model
        self.model = self._build_model(
            input_dim=2,
            output_dim=self.palette_size
        )
        
        if self.verbose:
            self.model.summary()
        
        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            ))
        
        # Train model
        if self.verbose:
            print(f"\nTraining for {epochs} epochs...")
        
        history = self.model.fit(
            X_train,
            Y_train_oh,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self.history = history.history
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final loss: {self.history['loss'][-1]:.6f}")
            print(f"Final accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self.history
    
    def predict(
        self,
        upscale_factor: int = 16,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Predict upscaled image using the trained model.
        
        Args:
            upscale_factor: Factor to upscale the image
            output_size: Optional specific output size (height, width)
            
        Returns:
            Upscaled RGB image array
        """
        if self.model is None or self.color_palette is None:
            raise ValueError("Model must be trained before prediction")
        
        # Determine output size
        if output_size:
            N1, N2 = output_size
        else:
            N1 = self.input_shape[0] * upscale_factor
            N2 = self.input_shape[1] * upscale_factor
        
        if self.verbose:
            print(f"Predicting upscaled image: {N1}x{N2}")
        
        # Create test coordinates
        S, T = self._create_coordinates(N1, N2)
        X_test = np.stack([S.ravel(), T.ravel()], axis=1).astype(np.float32)
        
        # Predict one-hot encoded colors
        Y_pred_oh = self.model.predict(X_test, verbose=0)
        
        # Convert one-hot back to colors
        predicted_indices = np.argmax(Y_pred_oh, axis=1)
        predicted_colors = self.color_palette[predicted_indices]
        
        # Reshape to image
        output_image = predicted_colors.reshape(N1, N2, 3)
        
        return output_image
    
    def predict_train(self) -> np.ndarray:
        """
        Predict on training data (reconstruction of input).
        
        Returns:
            Reconstructed image at original resolution
        """
        if self.model is None or self.color_palette is None:
            raise ValueError("Model must be trained before prediction")
        
        height, width, _ = self.input_shape
        
        # Create training coordinates
        S, T = self._create_coordinates(height, width)
        X_train = np.stack([S.ravel(), T.ravel()], axis=1).astype(np.float32)
        
        # Predict
        Y_pred_oh = self.model.predict(X_train, verbose=0)
        predicted_indices = np.argmax(Y_pred_oh, axis=1)
        predicted_colors = self.color_palette[predicted_indices]
        
        # Reshape
        output_image = predicted_colors.reshape(height, width, 3)
        
        return output_image
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'color_palette': self.color_palette,
            'input_shape': self.input_shape,
            'architecture': self.architecture,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        np.save(filepath.replace('.h5', '_metadata.npy'), metadata)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        # Load Keras model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.npy')
        metadata = np.load(metadata_path, allow_pickle=True).item()
        
        self.color_palette = metadata['color_palette']
        self.input_shape = metadata['input_shape']
        self.architecture = metadata['architecture']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.palette_size = len(self.color_palette)
        
        if self.verbose:
            print(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, list]:
        """Get training history (loss, accuracy, etc.)."""
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history:
            axes[1].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
