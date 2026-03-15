import numpy as np
from typing import List, Tuple, Optional

class CubicBSpline:
    """
    Cubic B-Spline curve implementation following Kopf-Lischinski 2011.
    
    Uses uniform cubic B-spline basis functions:
    B0(t) = (1-t)³/6
    B1(t) = (3t³ - 6t² + 4)/6
    B2(t) = (-3t³ + 3t² + 3t + 1)/6
    B3(t) = t³/6
    """
    
    # Corner detection threshold (angle in degrees)
    # Lower = more corners detected. 
    # 90° detects typical pixel-art corners, 120° detects smoother turns
    CORNER_ANGLE_THRESHOLD = 100.0
    
    def __init__(self, control_points: List[Tuple[float, float]], is_closed: bool = False):
        """
        control_points: list of (y, x) tuples defining the control polygon.
        is_closed: whether the curve should be closed (loop).
        """
        self.control_points = np.array(control_points, dtype=np.float64)
        self.is_closed = is_closed
        self.corners = set()  # Indices of corner control points
        self._detect_corners()
    
    def _basis_functions(self, t: float) -> np.ndarray:
        """
        Evaluates the 4 cubic B-spline basis functions at parameter t.
        Returns array [B0(t), B1(t), B2(t), B3(t)].
        """
        t2 = t * t
        t3 = t2 * t
        
        B0 = (1 - t) ** 3 / 6.0
        B1 = (3 * t3 - 6 * t2 + 4) / 6.0
        B2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
        B3 = t3 / 6.0
        
        return np.array([B0, B1, B2, B3])
    
    def _basis_derivative(self, t: float) -> np.ndarray:
        """
        Evaluates the derivative of basis functions at parameter t.
        Used for energy optimization.
        """
        t2 = t * t
        
        dB0 = -0.5 * (1 - t) ** 2
        dB1 = 0.5 * (3 * t2 - 4 * t)
        dB2 = 0.5 * (-3 * t2 + 2 * t + 1)
        dB3 = 0.5 * t2
        
        return np.array([dB0, dB1, dB2, dB3])
    
    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluates the B-spline curve at parameter t in [0, 1].
        For a curve with n control points, uses local support property.
        """
        n = len(self.control_points)
        if n < 2:
            return self.control_points[0] if n > 0 else np.array([0.0, 0.0])
        if n == 2:
            # Linear interpolation for 2 points
            return self.control_points[0] * (1 - t) + self.control_points[1] * t
        if n == 3:
            # Quadratic-like case
            return self._evaluate_segment(t, [0, 1, 2, 2])
        
        # For cubic: map t to the appropriate segment
        # Each segment uses 4 control points
        n_segments = n - 3 if not self.is_closed else n
        
        if self.is_closed:
            # Wrap around for closed curves
            t_scaled = t * n
            segment_idx = int(t_scaled) % n
            t_local = t_scaled - int(t_scaled)
            
            indices = [
                (segment_idx - 1) % n,
                segment_idx,
                (segment_idx + 1) % n,
                (segment_idx + 2) % n
            ]
        else:
            # Open curve
            t_scaled = t * (n - 3)
            segment_idx = max(0, min(int(t_scaled), n - 4))
            t_local = t_scaled - segment_idx
            
            indices = [segment_idx, segment_idx + 1, segment_idx + 2, segment_idx + 3]
        
        return self._evaluate_segment(t_local, indices)
    
    def _evaluate_segment(self, t: float, indices: List[int]) -> np.ndarray:
        """Evaluates a single B-spline segment using 4 control points."""
        B = self._basis_functions(t)
        points = self.control_points[indices]
        return B[0] * points[0] + B[1] * points[1] + B[2] * points[2] + B[3] * points[3]
    
    def evaluate_path(self, num_samples: int = 100) -> np.ndarray:
        """
        Samples the curve at num_samples points.
        Returns array of (y, x) coordinates.
        """
        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            samples.append(self.evaluate(t))
        return np.array(samples)
    
    def _detect_corners(self):
        """
        Detects corner points in the control polygon.
        Corners are vertices where the angle between adjacent segments
        exceeds the threshold. These points should be preserved during
        optimization.
        """
        n = len(self.control_points)
        if n < 3:
            return
        
        self.corners = set()
        
        for i in range(n):
            if self.is_closed:
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
            else:
                prev_idx = i - 1
                next_idx = i + 1
                # Skip endpoints for open curves (they don't have both neighbors)
                if prev_idx < 0 or next_idx >= n:
                    continue
            
            # Calculate angle at vertex i
            v1 = self.control_points[prev_idx] - self.control_points[i]
            v2 = self.control_points[next_idx] - self.control_points[i]
            
            # Normalize
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            
            v1 = v1 / len1
            v2 = v2 / len2
            
            # Angle between vectors (0 = sharp corner, 180 = straight line)
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            
            # If angle is sharp (less than threshold), mark as corner
            # A small angle means the path makes a sharp turn
            if angle < self.CORNER_ANGLE_THRESHOLD:
                self.corners.add(i)
    
    def is_corner(self, idx: int) -> bool:
        """Checks if a control point index is a corner."""
        return idx in self.corners
    
    def optimize(self, iterations: int = 10, smoothness_weight: float = 0.1):
        """
        Optimizes control points to minimize curve energy while preserving corners.
        
        Energy function:
        E = E_data + λ * E_smooth
        
        E_data: fidelity to original control points
        E_smooth: minimizes curvature (second derivative)
        """
        n = len(self.control_points)
        if n < 4:
            return  # Not enough points for optimization
        
        original_points = self.control_points.copy()
        
        for _ in range(iterations):
            new_points = self.control_points.copy()
            
            for i in range(n):
                # Skip corners - preserve sharp features
                if i in self.corners:
                    continue
                
                # Handle boundary conditions for open curves
                if not self.is_closed:
                    if i == 0 or i == n - 1:
                        continue
                
                # Compute forces from neighboring points
                prev_idx = (i - 1) % n if self.is_closed else i - 1
                next_idx = (i + 1) % n if self.is_closed else i + 1
                
                if not self.is_closed and (prev_idx < 0 or next_idx >= n):
                    continue
                
                # Smoothness force: Laplacian (move towards average of neighbors)
                laplacian = (self.control_points[prev_idx] + self.control_points[next_idx]) / 2.0
                laplacian -= self.control_points[i]
                
                # Data force: attraction to original position
                data_force = original_points[i] - self.control_points[i]
                
                # Combine forces
                displacement = smoothness_weight * laplacian + (1 - smoothness_weight) * data_force
                
                # Limit displacement to prevent oscillation
                max_displacement = 0.1
                disp_norm = np.linalg.norm(displacement)
                if disp_norm > max_displacement:
                    displacement = displacement * (max_displacement / disp_norm)
                
                new_points[i] = self.control_points[i] + displacement
            
            self.control_points = new_points
        
        # Re-detect corners after optimization
        self._detect_corners()
    
    def to_svg_path(self, num_samples: int = 50) -> str:
        """
        Converts the spline to an SVG path string.
        Uses cubic Bezier approximation for smoother output.
        """
        if len(self.control_points) < 2:
            return ""
        
        samples = self.evaluate_path(num_samples)
        
        # Build SVG path
        path = f"M {samples[0][1]:.3f} {samples[0][0]:.3f}"
        
        for i in range(1, len(samples)):
            path += f" L {samples[i][1]:.3f} {samples[i][0]:.3f}"
        
        if self.is_closed:
            path += " Z"
        
        return path


class SplineOptimizer:
    """
    Optimizes and chains contour segments into smooth B-spline curves.
    """
    
    def __init__(self, segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        self.segments = segments
        self.paths = self._chain_segments(segments)
        self.splines: List[CubicBSpline] = []
    
    def _chain_segments(self, segments: List) -> List[List[Tuple[float, float]]]:
        """
        Chains individual segments into continuous paths.
        Uses a greedy algorithm to connect segments end-to-end.
        """
        paths = []
        if not segments:
            return paths
        
        # Convert to list of mutable segments
        remaining = [list(seg) for seg in segments]
        
        while remaining:
            current_path = list(remaining[0])
            remaining.pop(0)
            
            if not remaining:
                paths.append(current_path)
                break
            
            # Try to extend the path
            extended = True
            while extended and remaining:
                extended = False
                last_point = current_path[-1]
                
                for i, seg in enumerate(remaining):
                    # Check if segment start matches path end
                    if self._points_match(seg[0], last_point):
                        current_path.append(seg[1])
                        remaining.pop(i)
                        extended = True
                        break
                    # Check if segment end matches path end (reverse)
                    elif self._points_match(seg[1], last_point):
                        current_path.append(seg[0])
                        remaining.pop(i)
                        extended = True
                        break
                
                if not extended:
                    break
            
            paths.append(current_path)
        
        return paths
    
    def _points_match(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                      tolerance: float = 0.01) -> bool:
        """Checks if two points are approximately equal."""
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
    
    def _simplify_path(self, path: List[Tuple[float, float]], 
                       tolerance: float = 0.5) -> List[Tuple[float, float]]:
        """
        Simplifies a path using Ramer-Douglas-Peucker algorithm.
        Reduces number of control points while preserving shape.
        """
        if len(path) < 3:
            return path
        
        points = np.array(path)
        
        # Find perpendicular distance from each point to line (start-end)
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return [path[0], path[-1]]
        
        line_unit = line_vec / line_len
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            point_vec = points[i] - start
            proj_len = np.dot(point_vec, line_unit)
            proj = start + proj_len * line_unit
            dist = np.linalg.norm(points[i] - proj)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # Recursively simplify if max distance exceeds tolerance
        if max_dist > tolerance:
            left = self._simplify_path(path[:max_idx + 1], tolerance)
            right = self._simplify_path(path[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]
    
    def get_splines(self, simplify_tolerance: float = 0.3, 
                    optimize_iterations: int = 5) -> List[CubicBSpline]:
        """
        Converts chained paths into optimized cubic B-splines.
        
        simplify_tolerance: threshold for Ramer-Douglas-Peucker simplification
        optimize_iterations: number of energy minimization iterations
        """
        self.splines = []
        
        for path in self.paths:
            # Simplify path to reduce control points
            simplified = self._simplify_path(path, simplify_tolerance)
            
            # Need at least 2 points for a curve
            if len(simplified) < 2:
                continue
            
            # Create cubic B-spline
            spline = CubicBSpline(simplified, is_closed=False)
            
            # Optimize if enough control points
            if len(simplified) >= 4:
                spline.optimize(iterations=optimize_iterations, smoothness_weight=0.15)
            
            self.splines.append(spline)
        
        return self.splines
    
    def get_spline_paths(self) -> List[np.ndarray]:
        """Returns sampled paths from all splines."""
        return [spline.evaluate_path(100) for spline in self.splines]
