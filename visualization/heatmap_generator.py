"""
Heatmap generator for footwork visualization
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.stats import gaussian_kde
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import VIZ_CONFIG


class HeatmapGenerator:
    """Generate footwork heatmaps"""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        bandwidth: float = 0.02,
    ):
        self.grid_size = grid_size
        self.bandwidth = bandwidth

    def generate_heatmap(
        self,
        positions: List[Tuple[float, float]],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate heatmap from position data

        Args:
            positions: List of (x, y) normalized positions
            weights: Optional weights for each position
            normalize: Whether to normalize to [0, 1]

        Returns:
            2D heatmap array
        """
        if len(positions) < 2:
            return np.zeros(self.grid_size)

        # Convert to numpy arrays
        points = np.array(positions).T  # Shape: (2, N)

        # Create grid
        x_grid = np.linspace(0, 1, self.grid_size[1])
        y_grid = np.linspace(0, 1, self.grid_size[0])
        xx, yy = np.meshgrid(x_grid, y_grid)

        try:
            # Use Gaussian KDE for density estimation
            if len(positions) > 1:
                kde = gaussian_kde(points, bw_method=self.bandwidth)
                positions_grid = np.vstack([xx.ravel(), yy.ravel()])
                heatmap = kde(positions_grid).reshape(self.grid_size)
            else:
                heatmap = np.zeros(self.grid_size)
        except Exception:
            # Fallback: simple histogram
            heatmap, _, _ = np.histogram2d(
                points[1], points[0],
                bins=self.grid_size,
                range=[[0, 1], [0, 1]],
                weights=weights,
            )

        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def apply_to_frame(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Apply heatmap overlay to video frame

        Args:
            frame: Original BGR frame
            heatmap: 2D heatmap array
            alpha: Transparency (0-1)
            colormap: OpenCV colormap

        Returns:
            Frame with heatmap overlay
        """
        h, w = frame.shape[:2]

        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend with original frame
        blended = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

        return blended

    def generate_trajectory_heatmap(
        self,
        positions: List[Tuple[float, float]],
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate heatmap with trajectory emphasis

        Args:
            positions: List of positions over time
            window_size: Optional window for temporal emphasis

        Returns:
            2D heatmap array
        """
        if len(positions) < 2:
            return np.zeros(self.grid_size)

        # Create weighted positions (more recent = higher weight)
        if window_size:
            weights = np.exp(np.linspace(-1, 0, min(len(positions), window_size)))
            weights = np.pad(weights, (len(positions) - len(weights), 0), 'constant')
        else:
            # Use velocity as weight (faster movement = brighter)
            weights = [0]
            for i in range(1, len(positions)):
                dist = np.linalg.norm(
                    np.array(positions[i]) - np.array(positions[i-1])
                )
                weights.append(dist)
            weights = np.array(weights)
            if weights.max() > 0:
                weights = weights / weights.max()

        return self.generate_heatmap(positions, weights.tolist())

    def get_hotspots(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.7,
        min_region_size: int = 5,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect hotspots in heatmap

        Returns:
            List of (x, y, intensity) for each hotspot
        """
        # Threshold heatmap
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8)
        )

        hotspots = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_region_size:
                cx, cy = centroids[i]
                # Normalize to [0, 1]
                x = cx / heatmap.shape[1]
                y = cy / heatmap.shape[0]
                intensity = heatmap[int(cy), int(cx)]
                hotspots.append((x, y, intensity))

        return hotspots

    def create_court_heatmap(
        self,
        positions: List[Tuple[float, float]],
        court_image: Optional[np.ndarray] = None,
        court_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Create heatmap overlaid on badminton court

        Args:
            positions: Player positions (normalized to court)
            court_image: Optional court background image
            court_bounds: Court corner points for calibration

        Returns:
            Heatmap visualization
        """
        heatmap = self.generate_heatmap(positions)

        if court_image is not None:
            return self.apply_to_frame(court_image, heatmap)

        # Generate default court visualization
        court_viz = self._create_default_court()
        return self.apply_to_frame(court_viz, heatmap)

    def _create_default_court(
        self,
        size: Tuple[int, int] = (600, 1000),
    ) -> np.ndarray:
        """Create a default badminton court visualization"""
        court = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        # Badminton court dimensions (singles)
        court_w, court_h = size
        margin = 50

        # Court color (green)
        cv2.rectangle(
            court,
            (margin, margin),
            (court_w - margin, court_h - margin),
            (34, 139, 34),
            -1,
        )

        # Lines (white)
        line_color = (255, 255, 255)
        line_thickness = 3

        # Outer boundary
        cv2.rectangle(
            court,
            (margin, margin),
            (court_w - margin, court_h - margin),
            line_color,
            line_thickness,
        )

        # Center line
        center_x = court_w // 2
        cv2.line(
            court,
            (center_x, margin),
            (center_x, court_h - margin),
            line_color,
            line_thickness,
        )

        # Service lines
        service_y1 = margin + int((court_h - 2 * margin) * 0.3)
        service_y2 = court_h - margin - int((court_h - 2 * margin) * 0.3)

        cv2.line(
            court,
            (margin, service_y1),
            (court_w - margin, service_y1),
            line_color,
            line_thickness,
        )
        cv2.line(
            court,
            (margin, service_y2),
            (court_w - margin, service_y2),
            line_color,
            line_thickness,
        )

        return court


class TemporalHeatmap:
    """Heatmap that evolves over time"""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        decay_factor: float = 0.95,
    ):
        self.grid_size = grid_size
        self.decay_factor = decay_factor
        self.heatmap = np.zeros(grid_size)
        self.generator = HeatmapGenerator(grid_size)

    def update(
        self,
        position: Tuple[float, float],
        weight: float = 1.0,
    ) -> np.ndarray:
        """Update heatmap with new position"""
        # Apply decay
        self.heatmap *= self.decay_factor

        # Add new position
        x_idx = int(position[0] * (self.grid_size[1] - 1))
        y_idx = int(position[1] * (self.grid_size[0] - 1))

        x_idx = np.clip(x_idx, 0, self.grid_size[1] - 1)
        y_idx = np.clip(y_idx, 0, self.grid_size[0] - 1)

        # Gaussian spread around position
        sigma = 2.0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = y_idx + dy, x_idx + dx
                if 0 <= ny < self.grid_size[0] and 0 <= nx < self.grid_size[1]:
                    dist_sq = dx * dx + dy * dy
                    self.heatmap[ny, nx] += weight * np.exp(-dist_sq / (2 * sigma * sigma))

        return self.heatmap.copy()

    def reset(self) -> None:
        """Reset heatmap"""
        self.heatmap = np.zeros(self.grid_size)
