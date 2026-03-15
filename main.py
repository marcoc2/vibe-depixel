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

    print("Construindo grafo de similaridade (Fase 1)...")
    graph = SimilarityGraph(img_array)
    
    print(f"Grafo construído com {len(graph.edges)} nós.")
    
    print("Resolvendo ambiguidades (Fase 2: Planarização)...")
    graph.planarize()
    
    print("Extraindo contornos (Fase 4)...")
    segments = graph.extract_visible_contours()
    
    from core.spline import SplineOptimizer
    optimizer = SplineOptimizer(segments)
    splines = optimizer.get_splines()
    
    print(f"Extraídas {len(optimizer.paths)} curvas contínuas.")
    
    print("Exportando para SVG (Fase 5)...")
    from core.render import SVGExporter
    exporter = SVGExporter(splines, graph.width, graph.height)
    output_path = "resultado.svg"
    exporter.save(output_path)
    
    print("Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
