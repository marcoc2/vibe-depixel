import numpy as np
from .color import are_colors_similar, rgb_to_yuv

class SimilarityGraph:
    def __init__(self, image_data):
        """
        image_data: NumPy array of shape (height, width, 3)
        """
        self.height, self.width, _ = image_data.shape
        self.pixels_yuv = np.zeros((self.height, self.width, 3))
        
        # Precompute YUV and initialize graph
        for y in range(self.height):
            for x in range(self.width):
                self.pixels_yuv[y, x] = rgb_to_yuv(image_data[y, x])
        
        # Connections: dict mapping (y, x) to list of (y2, x2)
        self.edges = {}
        self._build_initial_graph()

    def _build_initial_graph(self):
        """
        Builds the initial 8-connected grid and prunes based on color similarity.
        """
        for y in range(self.height):
            for x in range(self.width):
                self.edges[(y, x)] = []
                self._add_connections(y, x)

    def _add_connections(self, y, x):
        # 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if are_colors_similar(self.pixels_yuv[y, x], self.pixels_yuv[ny, nx]):
                        self.edges[(y, x)].append((ny, nx))

    def get_neighbors(self, y, x):
        return self.edges.get((y, x), [])

    def planarize(self):
        """
        Resolves crossing diagonals using heuristics: Curves, Sparse Pixels, and Islands.
        """
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                self._resolve_2x2_block(y, x)

    def _resolve_2x2_block(self, y, x):
        """
        Analyzes a 2x2 block to find crossing diagonals.
        Pixels in block:
        (y, x)     (y, x+1)
        (y+1, x)   (y+1, x+1)
        """
        # Diagonals
        d1 = ((y, x), (y+1, x+1))
        d2 = ((y, x+1), (y+1, x))
        
        has_d1 = (d1[1] in self.edges[d1[0]])
        has_d2 = (d2[1] in self.edges[d2[0]])
        
        if has_d1 and has_d2:
            # Both diagonals exist -> Crossing!
            # If all 4 edges are connected and colors are same, it's a "Fully Connected" block.
            # The paper says connectivity doesn't matter here, but let's check heuristics.
            self._apply_heuristics(y, x, d1, d2)

    def _apply_heuristics(self, y, x, d1, d2):
        # Weight for each diagonal
        w1 = 0
        w2 = 0
        
        # 1. Island Heuristic (Weight 5)
        if self._is_island(d1[0], d1[1]): w2 += 5
        if self._is_island(d2[0], d2[1]): w1 += 5

        # 2. Curve Heuristic
        len1 = self._get_path_length(d1[0], d1[1])
        len2 = self._get_path_length(d2[0], d2[1])
        if len1 > len2: w1 += (len1 - len2)
        elif len2 > len1: w2 += (len2 - len1)

        # 3. Sparse Heuristic
        # Weight is the difference in component size in 8x8 window
        s1 = self._get_sparse_count(y, x, d1[0])
        s2 = self._get_sparse_count(y, x, d2[0])
        if s1 < s2: w1 += (s2 - s1)
        elif s2 < s1: w2 += (s1 - s2)
        
        # Resolve
        if w1 > w2:
            self._remove_edge(d2[0], d2[1])
        elif w2 > w1:
            self._remove_edge(d1[0], d1[1])
        else:
            self._remove_edge(d1[0], d1[1])
            self._remove_edge(d2[0], d2[1])

    def _get_path_length(self, p1, p2):
        """Calculates the length of the connected component (same color) that this edge belongs to."""
        # Simple BFS/DFS to count nodes in the connected component
        visited = {p1, p2}
        stack = [p1, p2]
        count = 0
        color = self.pixels_yuv[p1[0], p1[1]]
        
        while stack:
            curr = stack.pop()
            count += 1
            for neighbor in self.edges.get(curr, []):
                if neighbor not in visited:
                    # Check if neighbor has same color (already pruned in early stage, but good to be safe)
                    visited.add(neighbor)
                    stack.append(neighbor)
        return count

    def _get_sparse_count(self, y_center, x_center, p):
        """Counts how many pixels of the same color as p exist in an 8x8 window."""
        count = 0
        color = self.pixels_yuv[p[0], p[1]]
        
        for dy in range(-4, 4):
            for dx in range(-4, 4):
                ny, nx = y_center + dy, x_center + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if are_colors_similar(color, self.pixels_yuv[ny, nx]):
                        count += 1
        return count

    def _is_island(self, p1, p2):
        """Checks if deleting edge (p1, p2) leaves p1 or p2 with only 1 connection."""
        # Note: neighbors include the other point of the diagonal currently.
        return len(self.edges[p1]) <= 1 or len(self.edges[p2]) <= 1

    def reshape_cells(self):
        """
        Phase 3: Reshapes pixel squares into adaptive cells.
        Calculates the polygon vertices for each pixel cell.
        """
        self.cells = {} # (y, x) -> list of (vy, vx)
        
        # We process each 2x2 block's center point.
        # There are (height+1) x (width+1) potential corner points.
        for y in range(self.height + 1):
            for x in range(self.width + 1):
                self._generate_vertices_at_corner(y, x)

    def _generate_vertices_at_corner(self, cy, cx):
        """
        At each corner (cy, cx), we decide how the 4 surrounding pixels meet.
        """
        # Pixel indices around corner (cy, cx)
        # p00: (cy-1, cx-1), p10: (cy, cx-1), p01: (cy-1, cx), p11: (cy, cx)
        p00 = (cy-1, cx-1)
        p10 = (cy, cx-1)
        p01 = (cy-1, cx)
        p11 = (cy, cx)
        
        # Connectivity
        # Check if p00 is connected to p11
        conn_diag1 = self._are_pixels_connected(p00, p11)
        # Check if p10 is connected to p01
        conn_diag2 = self._are_pixels_connected(p10, p01)
        
        # Based on Kopf-Lischinski, we might split the vertex.
        # If diag1 is connected, the corner is split to separate p10 and p01.
        # For simplicity, we assign the vertex (cy, cx) to the correct polygons.
        for p in [p00, p10, p01, p11]:
            if 0 <= p[0] < self.height and 0 <= p[1] < self.width:
                if p not in self.cells: self.cells[p] = []
                # In basic grid, every pixel gets the corner vertex.
                self.cells[p].append((cy, cx))

    def extract_visible_contours(self):
        """
        Phase 4: Extracts boundaries between disconnected components.
        Returns a list of segments [(v1, v2), ...]
        """
        visible_segments = []
        
        # We iterate over all adjacent pixel pairs (horizontal and vertical)
        for y in range(self.height):
            for x in range(self.width):
                # Check right neighbor
                if x + 1 < self.width:
                    if (y, x+1) not in self.edges.get((y, x), []):
                        # Shared edge is visible!
                        v1, v2 = (y, x+1), (y+1, x+1)
                        visible_segments.append((v1, v2))
                
                # Check bottom neighbor
                if y + 1 < self.height:
                    if (y+1, x) not in self.edges.get((y, x), []):
                        # Shared edge is visible!
                        v1, v2 = (y+1, x), (y+1, x+1)
                        visible_segments.append((v1, v2))
                        
        return visible_segments

    def _are_pixels_connected(self, p1, p2):
        if 0 <= p1[0] < self.height and 0 <= p1[1] < self.width and \
           0 <= p2[0] < self.height and 0 <= p2[1] < self.width:
            return p2 in self.edges.get(p1, [])
        return False

    def _get_cell_segments(self, y, x):
        """
        Identifies the segments that form the boundary of the cell for pixel (y, x).
        Segments are defined on a grid where corners are (y, x), (y+1, x), etc.
        """
        segments = []
        # Relative positions of the 8 neighbors
        neighbors_pos = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # This is the "Voronoi" reshaping part. 
        # For simplicity, we start with the 4 corners of the pixel square.
        # corners = [(y, x), (y, x+1), (y+1, x+1), (y+1, x)]
        # We need to adapt these based on connections.
        
        # A more robust way to implement Fase 3 is to detect which 
        # grid-edges (the lines between pixels) are "active".
        return segments

    def _remove_edge(self, p1, p2):
        if p2 in self.edges[p1]: self.edges[p1].remove(p2)
        if p1 in self.edges[p2]: self.edges[p2].remove(p1)
