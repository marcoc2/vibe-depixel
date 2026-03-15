class SVGExporter:
    def __init__(self, splines, width, height):
        self.splines = splines
        self.width = width
        self.height = height

    def save(self, file_path):
        """Saves the splines as an SVG file."""
        with open(file_path, 'w') as f:
            f.write(f'<svg width="{self.width*10}" height="{self.height*10}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg">\n')
            
            # Simple path rendering
            for spline in self.splines:
                points = spline.points
                if len(points) < 2: continue
                
                path_str = f"M {points[0][1]} {points[0][0]} "
                for p in points[1:]:
                    path_str += f"L {p[1]} {p[0]} "
                
                f.write(f'  <path d="{path_str}" stroke="black" stroke-width="0.05" fill="none" />\n')
            
            f.write('</svg>')
        print(f"Resultado exportado para: {file_path}")

    def export_similarity_graph(self, file_path, edges):
        """Exports the similarity graph edges (connections between pixels)."""
        with open(file_path, 'w') as f:
            f.write(f'<svg width="{self.width*10}" height="{self.height*10}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg" style="background: #222">\n')
            
            seen_edges = set()
            for p1, neighbors in edges.items():
                for p2 in neighbors:
                    edge = tuple(sorted((p1, p2)))
                    if edge not in seen_edges:
                        # Draw line between pixel centers (y+0.5, x+0.5)
                        f.write(f'  <line x1="{p1[1]+0.5}" y1="{p1[0]+0.5}" x2="{p2[1]+0.5}" y2="{p2[0]+0.5}" stroke="#44ff44" stroke-width="0.02" opacity="0.5" />\n')
                        seen_edges.add(edge)
            
            f.write('</svg>')
        print(f"Grafo de similaridade exportado para: {file_path}")

    def export_cells(self, file_path, cells, image_yuv):
        """Exports the adaptive cells as a colored SVG."""
        from .color import yuv_to_rgb
        with open(file_path, 'w') as f:
            f.write(f'<svg width="{self.width*20}" height="{self.height*20}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg" style="background: #eee">\n')
            
            for (y, x), poly in cells.items():
                # Convert YUV back to RGB for fill
                rgb = yuv_to_rgb(image_yuv[y, x])
                color_hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                
                points_str = " ".join([f"{v[1]},{v[0]}" for v in poly])
                # Draw the polygon
                f.write(f'  <polygon points="{points_str}" fill="{color_hex}" stroke="white" stroke-width="0.01" />\n')
            
            f.write('</svg>')
        print(f"Células exportadas para: {file_path}")
