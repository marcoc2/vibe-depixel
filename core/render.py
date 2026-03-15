class SVGExporter:
    def __init__(self, splines, width, height):
        self.splines = splines
        self.width = width
        self.height = height

    def save(self, file_path):
        """Saves the splines as an SVG file."""
        with open(file_path, 'w') as f:
            f.write(f'<svg width="{self.width*20}" height="{self.height*20}" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg">\n')
            
            # Simple path rendering
            for spline in self.splines:
                points = spline.points
                if len(points) < 2: continue
                
                path_str = f"M {points[0][1]} {points[0][0]} "
                for p in points[1:]:
                    path_str += f"L {p[1]} {p[0]} "
                
                f.write(f'  <path d="{path_str}" stroke="black" stroke-width="0.1" fill="none" />\n')
            
            f.write('</svg>')
        print(f"Resultado exportado para: {file_path}")
