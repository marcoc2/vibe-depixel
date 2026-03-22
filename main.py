import sys
import os
import argparse
from PIL import Image
import numpy as np
from core.graph import SimilarityGraph


def run_kopf_lischinski_pipeline(img: Image.Image, img_array: np.ndarray, run_dir: str):
    """
    Run the Kopf-Lischinski 2011 vectorization pipeline (Phases 1-5).
    """
    print("Construindo grafo de similaridade (Fase 1)...")
    graph = SimilarityGraph(img_array)

    print(f"Grafo construído com {len(graph.edges)} nós.")

    print("Resolvendo ambiguidades (Fase 2: Planarização)...")
    graph.planarize()

    print("Remodelando células (Fase 3: Geometria)...")
    graph.reshape_cells()

    print("Extraindo contornos (Fase 4)...")
    segments = graph.extract_visible_contours()

    from core.spline import SplineOptimizer
    print("Otimizando splines (Fase 5: Spline Fitting & Optimization)...")
    optimizer = SplineOptimizer(segments)
    # Use lower tolerance to keep more control points for corner detection
    splines = optimizer.get_splines(simplify_tolerance=0.1, optimize_iterations=3)

    print(f"Extraídas {len(splines)} curvas B-spline cúbicas otimizadas.")

    # Count corners detected and show stats
    total_corners = sum(len(s.corners) for s in splines)
    total_control_points = sum(len(s.control_points) for s in splines)
    avg_control_points = total_control_points / len(splines) if splines else 0
    print(f"Total de cantos (corners) preservados: {total_corners}")
    print(f"Pontos de controle totais: {total_control_points} (média: {avg_control_points:.2f} por curva)")

    print("Exportando artefatos (Fase 5)...")
    from core.render import SVGExporter
    exporter = SVGExporter(splines, graph.width, graph.height)

    # Save the similarity graph (connections)
    graph_path = os.path.join(run_dir, "grafo_similaridade.svg")
    exporter.export_similarity_graph(graph_path, graph.edges)

    # Save the contours/splines (clean version)
    spline_path = os.path.join(run_dir, "contornos_splines.svg")
    exporter.save(spline_path)

    # Save debug version with control points and corners
    spline_debug_path = os.path.join(run_dir, "contornos_splines_debug.svg")
    exporter.save(spline_debug_path, show_control_points=True, show_corners=True)

    # Save the adaptive cells mesh
    cells_path = os.path.join(run_dir, "celulas_voronoi.svg")
    exporter.export_cells(cells_path, graph.cells, graph.pixels_yuv)


def run_deep_nn_pipeline(
    img: Image.Image,
    run_dir: str,
    upscale_factor: int = 16,
    epochs: int = 1000,
    save_model: bool = False
):
    """
    Run the Deep Neural Network depixelization pipeline.
    
    Based on Diego Inacio's approach:
    - Learns color palette from input image
    - Maps 2D coordinates to colors using one-hot encoding
    - Predicts upscaled image at arbitrary resolution
    """
    from core.deep_nn import DeepNNDepixelizer
    
    print(f"Upscale factor: {upscale_factor}x")
    print(f"Training epochs: {epochs}")
    
    # Initialize depixelizer
    depixelizer = DeepNNDepixelizer(verbose=True)
    
    # Train the model
    depixelizer.fit(
        image=img,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        early_stopping=True,
        patience=50
    )
    
    # Predict upscaled image
    print("\nGenerating upscaled prediction...")
    upscaled = depixelizer.predict(upscale_factor=upscale_factor)
    
    # Clip and convert to uint8
    upscaled = np.clip(upscaled * 255, 0, 255).astype(np.uint8)
    
    # Save upscaled image
    upscaled_path = os.path.join(run_dir, f"upscaled_{upscale_factor}x.png")
    upscaled_img = Image.fromarray(upscaled)
    upscaled_img.save(upscaled_path)
    print(f"Upscaled image saved to: {upscaled_path}")
    
    # Save reconstruction (training prediction)
    print("\nGenerating reconstruction at original resolution...")
    reconstruction = depixelizer.predict_train()
    reconstruction = np.clip(reconstruction * 255, 0, 255).astype(np.uint8)
    
    recon_path = os.path.join(run_dir, "reconstruction.png")
    recon_img = Image.fromarray(reconstruction)
    recon_img.save(recon_path)
    print(f"Reconstruction saved to: {recon_path}")
    
    # Save training history plot
    history_path = os.path.join(run_dir, "training_history.png")
    depixelizer.plot_training_history(save_path=history_path)
    
    # Save model if requested
    if save_model:
        model_path = os.path.join(run_dir, "model.h5")
        depixelizer.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    # Create comparison image (original vs reconstruction vs upscaled)
    img_array = np.array(img)
    original_resized = np.kron(img_array, np.ones((upscale_factor, upscale_factor, 1), dtype=np.uint8))
    
    # Pad original if needed
    orig_h, orig_w = original_resized.shape[:2]
    up_h, up_w = upscaled.shape[:2]
    
    if orig_h < up_h or orig_w < up_w:
        padded_original = np.zeros((up_h, up_w, 3), dtype=np.uint8)
        padded_original[:orig_h, :orig_w] = original_resized
        original_resized = padded_original
    
    comparison = np.hstack([original_resized, upscaled])
    comparison_path = os.path.join(run_dir, "comparison.png")
    comparison_img = Image.fromarray(comparison)
    comparison_img.save(comparison_path)
    print(f"Comparison image saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Depixelizing Pixel Art - Vectorization and Deep Learning Upscaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Kopf-Lischinski vectorization pipeline (default)
  python main.py input/megaman.png
  
  # Run Deep Neural Network upscaling
  python main.py input/megaman.png --nn --upscale 16 --epochs 1000
  
  # Run both pipelines
  python main.py input/megaman.png --both
        """
    )
    
    parser.add_argument(
        "image_path",
        nargs='?',
        default=None,
        help="Path to the input pixel art image"
    )
    
    parser.add_argument(
        "--nn", "--neural-network",
        action="store_true",
        dest="use_nn",
        help="Use Deep Neural Network for upscaling instead of vectorization"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both Kopf-Lischinski vectorization AND Deep NN upscaling"
    )
    
    parser.add_argument(
        "--upscale", "-u",
        type=int,
        default=16,
        help="Upscale factor for Deep NN (default: 16)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1000,
        help="Number of training epochs for Deep NN (default: 1000)"
    )
    
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained Deep NN model"
    )

    # Super-Resolution flags
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the SR model on paired LR/HR dataset"
    )
    parser.add_argument(
        "--lr-dir",
        help="Directory with low-resolution training images"
    )
    parser.add_argument(
        "--hr-dir",
        help="Directory with high-resolution training images"
    )
    parser.add_argument(
        "--val-lr-dir",
        help="Directory with low-resolution validation images (optional, auto-split if omitted)"
    )
    parser.add_argument(
        "--val-hr-dir",
        help="Directory with high-resolution validation images (optional)"
    )
    parser.add_argument(
        "--sr",
        action="store_true",
        help="Run SR model inference on the input image"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sr_model_best.pth",
        help="Path to SR model checkpoint (default: checkpoints/sr_model_best.pth)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="LR patch size for training (default: 64)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for SR training (default: 16)"
    )
    parser.add_argument(
        "--sr-epochs",
        type=int,
        default=200,
        help="Number of epochs for SR training (default: 200)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for SR training (default: 5e-4)"
    )
    parser.add_argument(
        "--preset",
        default="default",
        help="Model preset: 'default', 'gemini', or 'esrgan'"
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to pre-trained ESRGAN .pth file (required for --preset esrgan)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last saved state"
    )
    parser.add_argument(
        "--no-perceptual",
        action="store_true",
        help="Disable VGG19 perceptual loss during SR training (on by default)"
    )
    parser.add_argument(
        "--lum-loss",
        action="store_true",
        help="Use YCbCr-weighted loss: Y=1.0, Cb/Cr=0.1 (robust to color drift)"
    )

    args = parser.parse_args()
    
    # --- SR training mode (no image_path needed) ---
    if args.train:
        if not args.lr_dir or not args.hr_dir:
            parser.error("--train requires --lr-dir and --hr-dir")
        from core.sr_train import train as sr_train
        sr_train(
            lr_dir=args.lr_dir,
            hr_dir=args.hr_dir,
            val_lr_dir=args.val_lr_dir,
            val_hr_dir=args.val_hr_dir,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            epochs=args.sr_epochs,
            lr=args.learning_rate,
            use_perceptual=not args.no_perceptual,
            use_lum_loss=args.lum_loss,
            preset=args.preset,
            resume=args.resume,
            pretrained_path=args.pretrained,
        )
        return

    # --- From here, image_path is required ---
    if args.image_path is None:
        parser.error("image_path is required for inference pipelines")

    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Erro: Arquivo {args.image_path} não encontrado.")
        return

    # --- SR inference mode ---
    if args.sr:
        from core.sr_train import infer as sr_infer
        checkpoint = args.checkpoint
        if checkpoint == "checkpoints/sr_model_best.pth":
            checkpoint = f"checkpoints/{args.preset}/sr_model_best.pth"
        sr_infer(args.image_path, checkpoint, preset=args.preset, pretrained_path=args.pretrained)
        return

    # Load image
    print(f"Carregando imagem: {args.image_path}")
    img = Image.open(args.image_path).convert('RGB')
    img_array = np.array(img)
    print(f"Imagem carregada: {img_array.shape[1]}x{img_array.shape[0]} pixels")

    # Create timestamped output directory
    from datetime import datetime
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", f"{image_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {run_dir}\n")

    # Run selected pipeline(s)
    if args.use_nn and not args.both:
        # Deep NN only
        run_deep_nn_pipeline(
            img, run_dir,
            upscale_factor=args.upscale,
            epochs=args.epochs,
            save_model=args.save_model
        )
    elif args.both:
        # Run Kopf-Lischinski first
        print("=" * 60)
        print("PIPELINE 1: Kopf-Lischinski Vectorization")
        print("=" * 60 + "\n")
        run_kopf_lischinski_pipeline(img, img_array, run_dir)
        
        print("\n" + "=" * 60)
        print("PIPELINE 2: Deep Neural Network Upscaling")
        print("=" * 60 + "\n")
        run_deep_nn_pipeline(
            img, run_dir,
            upscale_factor=args.upscale,
            epochs=args.epochs,
            save_model=args.save_model
        )
    else:
        # Kopf-Lischinski only (default)
        run_kopf_lischinski_pipeline(img, img_array, run_dir)
    
    print(f"\n{'=' * 60}")
    print(f"Processo concluído com sucesso! Pasta de saída: {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
