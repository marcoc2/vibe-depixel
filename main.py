import sys
import os
from PIL import Image
import numpy as np
from core.graph import SimilarityGraph

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_da_imagem>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo {image_path} não encontrado.")
        return

    print(f"Carregando imagem: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Create timestamped output directory
    from datetime import datetime
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", f"{image_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {run_dir}")

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

    # NEW: Save the similarity graph (connections)
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
    
    print(f"Processo concluído com sucesso! Pasta de saída: {run_dir}")

if __name__ == "__main__":
    main()
