import numpy as np

class BSplineCurve:
    def __init__(self, control_points):
        """
        control_points: list of (y, x) tuples.
        """
        self.points = np.array(control_points)

    def evaluate(self, t):
        """
        Evaluates the quadratic B-spline at parameter t in [0, 1].
        This is a simplified version.
        """
        # Linear interpolation as placeholder
        return self.points[0] * (1-t) + self.points[-1] * t

class SplineOptimizer:
    def __init__(self, segments):
        self.segments = segments
        self.paths = self._chain_segments(segments)

    def _chain_segments(self, segments):
        """Chains individual segments into continuous paths."""
        paths = []
        if not segments: return paths
        
        # Simple greedy chaining
        current_path = list(segments[0])
        remaining = segments[1:]
        
        while remaining:
            last_point = current_path[-1]
            found = False
            for i, seg in enumerate(remaining):
                if seg[0] == last_point:
                    current_path.append(seg[1])
                    remaining.pop(i)
                    found = True
                    break
                elif seg[1] == last_point:
                    current_path.append(seg[0])
                    remaining.pop(i)
                    found = True
                    break
            
            if not found:
                paths.append(current_path)
                if remaining:
                    current_path = list(remaining.pop(0))
                else:
                    break
        
        if current_path: paths.append(current_path)
        return paths

    def get_splines(self):
        return [BSplineCurve(path) for path in self.paths]
