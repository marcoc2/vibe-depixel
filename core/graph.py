import numpy as np
from .color import are_colors_similar, rgb_to_yuv

class SimilarityGraph:
    def __init__(self, image_data):
        self.height, self.width, _ = image_data.shape
        self.pixels_yuv = np.zeros((self.height, self.width, 3))
        
        for y in range(self.height):
            for x in range(self.width):
                self.pixels_yuv[y, x] = rgb_to_yuv(image_data[y, x])
        
        self.edges = {}
        self.component_sizes = {}
        self.cells = {}
        self._build_initial_graph()

    def _build_initial_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                self.edges[(y, x)] = []
                self._add_connections(y, x)

    def _add_connections(self, y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if are_colors_similar(self.pixels_yuv[y, x], self.pixels_yuv[ny, nx]):
                        self.edges[(y, x)].append((ny, nx))

    def planarize(self):
        """
        Resolves crossing diagonals. 
        Optimized version: uses component size cache and sequential loop to avoid IPC overhead.
        """
        # Step 1: Cache component sizes (O(N))
        self.component_sizes = self._precompute_component_sizes()
        
        # Step 2: Resolve blocks
        crossings_count = 0
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                d1 = ((y, x), (y+1, x+1))
                d2 = ((y, x+1), (y+1, x))
                
                # Check for crossing
                if (d1[1] in self.edges[d1[0]]) and (d2[1] in self.edges[d2[0]]):
                    self._resolve_2x2_block(y, x, d1, d2)
                    crossings_count += 1
        
        if crossings_count > 0:
            print(f"Resolvidos {crossings_count} cruzamentos diagonais.")

    def _resolve_2x2_block(self, y, x, d1, d2):
        w1, w2 = 0, 0
        
        # 1. Island Heuristic
        if len(self.edges[d1[0]]) <= 1 or len(self.edges[d1[1]]) <= 1: w2 += 10
        if len(self.edges[d2[0]]) <= 1 or len(self.edges[d2[1]]) <= 1: w1 += 10

        # 2. Curve Heuristic
        len1 = self.component_sizes.get(d1[0], 0)
        len2 = self.component_sizes.get(d2[0], 0)
        if len1 > len2: w1 += (len1 - len2)
        elif len2 > len1: w2 += (len2 - len1)

        # 3. Sparse Heuristic
        s1 = self._get_sparse_count(y, x, d1[0])
        s2 = self._get_sparse_count(y, x, d2[0])
        if s1 < s2: w1 += (s2 - s1)
        elif s2 < s1: w2 += (s1 - s2)
        
        # Resolve
        if w1 > w2:
            self._remove_edge(d2[0], d2[1])
        else:
            self._remove_edge(d1[0], d1[1])

    def _precompute_component_sizes(self):
        visited = set()
        component_map = {}
        for y in range(self.height):
            for x in range(self.width):
                p = (y, x)
                if p not in visited:
                    comp_nodes = []
                    stack = [p]
                    visited.add(p)
                    while stack:
                        curr = stack.pop()
                        comp_nodes.append(curr)
                        for neighbor in self.edges.get(curr, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                stack.append(neighbor)
                    size = len(comp_nodes)
                    for node in comp_nodes:
                        component_map[node] = size
        return component_map

    def _get_sparse_count(self, y_h, x_h, p):
        count = 0
        color = self.pixels_yuv[p[0], p[1]]
        for dy in range(-4, 4):
            for dx in range(-4, 4):
                ny, nx = y_h + dy, x_h + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if are_colors_similar(color, self.pixels_yuv[ny, nx]):
                        count += 1
        return count

    def _remove_edge(self, p1, p2):
        if p2 in self.edges[p1]: self.edges[p1].remove(p2)
        if p1 in self.edges[p2]: self.edges[p2].remove(p1)

    def extract_visible_contours(self):
        visible_segments = []
        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    if (y, x+1) not in self.edges.get((y, x), []):
                        visible_segments.append(((y, x+1), (y+1, x+1)))
                if y + 1 < self.height:
                    if (y+1, x) not in self.edges.get((y, x), []):
                        visible_segments.append(((y+1, x), (y+1, x+1)))
        return visible_segments

    def reshape_cells(self):
        """
        Phase 3: Reshapes pixel squares into adaptive cells using vertex splitting.
        """
        self.cells = {} # (y, x) -> list of vertices in clockwise order
        
        # We'll store vertices by corner and type. 
        # corner_vertices[(cy, cx)] = list of (y, x) vertex positions
        corner_vertices = {}
        
        for cy in range(self.height + 1):
            for cx in range(self.width + 1):
                 corner_vertices[(cy, cx)] = self._get_vertices_for_corner(cy, cx)

        # Build polygons for each pixel
        for y in range(self.height):
            for x in range(self.width):
                # A pixel (y, x) is surrounded by 4 corners:
                # Top-Left: (y, x), Top-Right: (y, x+1), Bottom-Right: (y+1, x+1), Bottom-Left: (y+1, x)
                poly = []
                
                # Top-Left corner (y, x)
                v_tl = corner_vertices[(y, x)]
                poly.extend(self._pick_vertices_for_pixel(y, x, y, x, v_tl))
                
                # Top-Right corner (y, x+1)
                v_tr = corner_vertices[(y, x+1)]
                poly.extend(self._pick_vertices_for_pixel(y, x, y, x+1, v_tr))
                
                # Bottom-Right corner (y+1, x+1)
                v_br = corner_vertices[(y+1, x+1)]
                poly.extend(self._pick_vertices_for_pixel(y, x, y+1, x+1, v_br))
                
                # Bottom-Left corner (y+1, x)
                v_bl = corner_vertices[(y+1, x)]
                poly.extend(self._pick_vertices_for_pixel(y, x, y+1, x, v_bl))
                
                self.cells[(y, x)] = poly

    def _get_vertices_for_corner(self, cy, cx):
        """Returns 1 or 2 vertices for a given grid corner."""
        p00, p01, p10, p11 = (cy-1, cx-1), (cy-1, cx), (cy, cx-1), (cy, cx)
        
        # Check diagonal connections
        conn1 = self._are_pixels_connected(p00, p11)
        conn2 = self._are_pixels_connected(p01, p10)
        
        if conn1:
            # Diagonal 1 is connected. Split perpendicular to it.
            # Vertices moved 0.25 towards connected pixels.
            return [(cy - 0.25, cx + 0.25), (cy + 0.25, cx - 0.25)]
        elif conn2:
            # Diagonal 2 is connected.
            return [(cy - 0.25, cx - 0.25), (cy + 0.25, cx + 0.25)]
        else:
            return [(float(cy), float(cx))]

    def _pick_vertices_for_pixel(self, py, px, cy, cx, corner_verts):
        """Picks and orders the vertices from a corner that belong to a specific pixel."""
        if len(corner_verts) == 1:
            return corner_verts
        
        # Split case. We have 2 vertices per corner.
        # We need to determine which ones are 'inside' or 'on boundary' for pixel (py, px).
        # Actually, each pixel gets 1 or 2 vertices from the split.
        # If the pixel is part of the connection that caused the split, it gets BOTH.
        # If it's one of the "squeezed" pixels, it gets ONE (the one closer to it?).
        # No, the paper says the new edge separates the squeezed pixels.
        
        # Corner (cy, cx) pixels:
        p00, p01, p10, p11 = (cy-1, cx-1), (cy-1, cx), (cy, cx-1), (cy, cx)
        conn1 = self._are_pixels_connected(p00, p11)
        conn2 = self._are_pixels_connected(p01, p10)
        
        v1, v2 = corner_verts
        # v1 is (cy-0.25, cx+0.25) or (cy-0.25, cx-0.25)
        # v2 is (cy+0.25, cx-0.25) or (cy+0.25, cx+0.25)
        
        if conn1: # p00 and p11 are connected
            if (py, px) == p00: return [v1, v2] # v1 is top-right, v2 is bottom-left relative to corner
            if (py, px) == p11: return [v2, v1] # reverse order for clockwise?
            if (py, px) == p01: return [v1]
            if (py, px) == p10: return [v2]
        elif conn2: # p01 and p10 are connected
            if (py, px) == p01: return [v1, v2]
            if (py, px) == p10: return [v2, v1]
            if (py, px) == p00: return [v1]
            if (py, px) == p11: return [v2]
            
        return corner_verts

    def extract_visible_contours(self):
        """
        Phase 4: Extracts boundaries between disconnected components using cell geometry.
        Returns a list of segments [((y1, x1), (y2, x2)), ...]
        """
        if not hasattr(self, 'cells'):
            self.reshape_cells()
            
        visible_segments = []
        for y in range(self.height):
            for x in range(self.width):
                poly = self.cells[(y, x)]
                n = len(poly)
                for i in range(n):
                    v1 = poly[i]
                    v2 = poly[(i + 1) % n]
                    
                    # We need to check if this segment is a boundary.
                    # A segment is a boundary if it's shared with a pixel of a DIFFERENT color
                    # OR if it's on the image border.
                    # Simplified: for now, if it's not internal to a same-color component.
                    # But the cells are already built based on connectivity!
                    # Only segments between UNCONNECTED pixels are "visible".
                    
                    # This is complex to check perfectly here.
                    # Alternative: iterate over all cell edges and check neighbors.
                    pass
        
        # Falling back to the previous logic but using cell vertices for better geometry
        # is hard without a full topological mesh. 
        # Let's use the pixel-based connectivity to decide which pixel-pixel boundaries are visible.
        # Then, use the split vertices to define the exact segment.
        
        visible_segments = []
        for y in range(self.height):
            for x in range(self.width):
                # Right boundary
                if x + 1 < self.width:
                    if (y, x+1) not in self.edges.get((y, x), []):
                        # Boundary between (y, x) and (y, x+1)
                        # This boundary goes from corner (y, x+1) to (y+1, x+1)
                        c1 = (y, x+1)
                        c2 = (y+1, x+1)
                        v_start = self._get_boundary_vertex(y, x, y, x+1, c1)
                        v_end = self._get_boundary_vertex(y, x, y, x+1, c2)
                        visible_segments.append((v_start, v_end))
                
                # Bottom boundary
                if y + 1 < self.height:
                    if (y+1, x) not in self.edges.get((y, x), []):
                        c1 = (y+1, x)
                        c2 = (y+1, x+1)
                        v_start = self._get_boundary_vertex(y, x, y+1, x, c1)
                        v_end = self._get_boundary_vertex(y, x, y+1, x, c2)
                        visible_segments.append((v_start, v_end))
                        
        return visible_segments

    def _get_boundary_vertex(self, p1y, p1x, p2y, p2x, corner):
        """Returns the specific vertex at a corner that separates two pixels."""
        cy, cx = corner
        verts = self._get_vertices_for_corner(cy, cx)
        if len(verts) == 1: return verts[0]
        
        # Split case.
        # If the split separates p1 and p2, we need the vertex that is part of that split edge.
        # Actually, the split edge IS the boundary.
        # If diagonal split happens at this corner:
        p00, p01, p10, p11 = (cy-1, cx-1), (cy-1, cx), (cy, cx-1), (cy, cx)
        conn1 = self._are_pixels_connected(p00, p11)
        conn2 = self._are_pixels_connected(p01, p10)
        
        v1, v2 = verts
        if conn1:
            # Edge is between v1 and v2.
            # If we are looking for boundary between p01 and p10, it's the segment (v1, v2).
            # But which one is 'start' and 'end'? 
            # This depends on which corner we are at.
            # Let's just return the midpoint for now to avoid complexity, 
            # or better, return the mean if it's the "squeezed" boundary.
            if { (p1y, p1x), (p2y, p2x) } == { p01, p10 }:
                 # This is the "squeezed" boundary.
                 # Actually, we should return BOTH vertices as a mini-segment?
                 # No, the caller expects a single vertex for the path.
                 return ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2)
            else:
                 # Boundary is between p00 and p01? No, p00 and p11 are connected.
                 # Boundary between p00 and p01 goes from corner to split vertex.
                 if (p1y, p1x) == p00 or (p2y, p2x) == p00:
                      return v1 if (p1y, p1x) == p01 or (p2y, p2x) == p01 else v2
        
        return verts[0]

    def _are_pixels_connected(self, p1, p2):
        return p2 in self.edges.get(p1, [])
