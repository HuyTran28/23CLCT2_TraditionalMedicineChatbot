# Image Extractor Module
# Extracts images, figures, and charts from PDF documents

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)

class ImageExtractor:
    """
    Extract images, figures, charts, and diagrams from PDF documents.
    Assigns unique IDs to each extracted image for referencing in text.
    """
    
    def __init__(self, output_dir: str | Path = "./images", min_width: int = 100, min_height: int = 100):
        """
        Initialize Image Extractor
        
        Args:
            output_dir: Directory to save extracted images
            min_width: Minimum width threshold for extracted images (pixels)
            min_height: Minimum height threshold for extracted images (pixels)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width
        self.min_height = min_height
        self.image_counter = 0
        
    def extract_images_from_pdf(
        self, 
        pdf_path: str | Path,
        prefix: str = "img",
        max_page_coverage: float = 0.85,
        detect_page_images: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract all images from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            prefix: Prefix for image filenames
            max_page_coverage: Max ratio of page area an image can cover (0.85 = 85%)
            detect_page_images: If True, uses smart detection to identify actual page scans vs content images
            
        Returns:
            List of dictionaries containing image metadata:
            - image_id: Unique identifier (e.g., "img_001")
            - page_num: Page number where image was found
            - file_path: Path to saved image file
            - bbox: Bounding box [x0, y0, x1, y1]
            - width: Image width in pixels
            - height: Image height in pixels
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        extracted_images = []
        extracted_xrefs = set()  # Track extracted images to avoid duplicates
        
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Extracting images from {pdf_path.name} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                page_area = page_width * page_height
                
                # Get list of images on the page
                image_list = page.get_images(full=True)
                
                logger.debug(f"  Page {page_num + 1}: Found {len(image_list)} image references (page size: {int(page_width)}x{int(page_height)})")
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # xref is the image reference number
                        
                        # Skip if already extracted (same image used multiple times)
                        if xref in extracted_xrefs:
                            logger.debug(f"    Skipping duplicate xref {xref}")
                            continue
                        
                        # Get image position on page first to check if it's a full-page image
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            logger.debug(f"    Skipping image with no position data")
                            continue
                        
                        rect = img_rects[0]  # Get first occurrence
                        # Calculate image area on page
                        img_width_on_page = rect.x1 - rect.x0
                        img_height_on_page = rect.y1 - rect.y0
                        img_area_on_page = img_width_on_page * img_height_on_page
                        
                        # Calculate coverage ratio
                        coverage_ratio = img_area_on_page / page_area if page_area > 0 else 0
                        
                        # Smart filtering for page-sized images
                        if detect_page_images and coverage_ratio > max_page_coverage:
                            # Check if this is a true page scan vs a large content image
                            # Page scans typically:
                            # 1. Start at or near page origin (0,0)
                            # 2. Cover nearly the entire page (>85%)
                            # 3. Have aspect ratio similar to page
                            
                            is_at_origin = rect.x0 < 10 and rect.y0 < 10
                            page_aspect = page_width / page_height if page_height > 0 else 1
                            img_aspect = img_width_on_page / img_height_on_page if img_height_on_page > 0 else 1
                            aspect_similar = abs(page_aspect - img_aspect) < 0.1
                            
                            if is_at_origin and coverage_ratio > 0.9 and aspect_similar:
                                logger.debug(f"    Skipping page scan/background (covers {coverage_ratio*100:.1f}%, at origin, matching aspect)")
                                continue
                            else:
                                # Large image but not a page scan - likely a diagram/chart
                                logger.debug(f"    Keeping large content image (covers {coverage_ratio*100:.1f}% but not a page scan)")
                            logger.debug(f"    Skipping full-page image (covers {coverage_ratio*100:.1f}% of page)")
                            continue
                        
                        # Extract image
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Get image dimensions
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        # Filter small images (likely decorative elements)
                        if width < self.min_width or height < self.min_height:
                            logger.debug(f"    Skipping small image: {width}x{height}")
                            continue
                        
                        # Mark this xref as extracted
                        extracted_xrefs.add(xref)
                        
                        # Generate unique image ID
                        self.image_counter += 1
                        image_id = f"{prefix}_{self.image_counter:03d}"
                        
                        # Save image
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = self.output_dir / image_filename
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Define bbox from rect
                        bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                        
                        extracted_images.append({
                            "image_id": image_id,
                            "page_num": page_num + 1,
                            "file_path": str(image_path),
                            "filename": image_filename,
                            "bbox": bbox,
                            "width": width,
                            "height": height,
                            "format": image_ext
                        })
                        
                        logger.debug(f"    ✓ Extracted {image_id}: {width}x{height} ({image_ext}) - covers {coverage_ratio*100:.1f}% of page at {[int(x) for x in bbox]}")
                        
                    except Exception as e:
                        logger.warning(f"    Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
                
                # Log summary for this page
                page_img_count = sum(1 for img in extracted_images if img['page_num'] == page_num + 1)
                if len(image_list) > 0:
                    logger.info(f"  Page {page_num + 1}: Extracted {page_img_count}/{len(image_list)} images")
                elif page_img_count > 0:
                    logger.info(f"  Page {page_num + 1}: Extracted {page_img_count} images")
            
            doc.close()
            if len(extracted_images) > 0:
                logger.info(f"✓ Total images extracted: {len(extracted_images)}")
            else:
                logger.info("No embedded images found in PDF (this is normal for scanned documents)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract images from PDF: {e}")
        
        return extracted_images
    
    def is_illustration(self, figure_img, check_text: bool = True) -> bool:
        """
        Enhanced heuristic to check if a cropped box is likely an illustration (not text).
        Uses multiple computer vision techniques:
        - Pixel density analysis
        - Edge density (illustrations have more edges)
        - Connected component analysis
        - Optional OCR text detection
        
        Returns True if likely illustration, False if likely text.
        """
        import cv2
        import numpy as np

        # Convert to grayscale if needed
        if len(figure_img.shape) == 3:
            gray = cv2.cvtColor(figure_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = figure_img
        
        h, w = gray.shape
        total_pixels = h * w
        
        # 1. Pixel density check
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        nonwhite_ratio = np.count_nonzero(binary) / total_pixels
        
        # If less than 2% nonwhite, it's probably empty
        if nonwhite_ratio < 0.02:
            return False
        
        # 2. Edge density check (illustrations typically have more edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / total_pixels
        
        # High edge density suggests illustration/diagram
        if edge_ratio > 0.15:
            return True
        
        # 3. Connected components analysis
        # Text has many small components, illustrations have fewer larger ones
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 2:  # Background + at least 1 component
            # Get component sizes (excluding background)
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            avg_component_size = np.mean(component_areas) if len(component_areas) > 0 else 0
            
            # Large average component size suggests illustration
            if avg_component_size > total_pixels * 0.05:
                return True
            
            # Few large components suggest illustration, many small ones suggest text
            if num_labels < 20 and nonwhite_ratio > 0.1:
                return True
        
        # 4. Aspect ratio and density combined
        # Dense images with reasonable aspect ratio are likely illustrations
        if nonwhite_ratio > 0.4:
            return True
        
        # 5. Optional OCR check (can be slow, so make it optional)
        if check_text and nonwhite_ratio > 0.05:
            try:
                import pytesseract
                # Quick text detection
                text = pytesseract.image_to_string(gray, config='--psm 6 --oem 1').strip()
                # If significant text found, not an illustration
                if len(text) > 50:
                    return False
            except Exception:
                # If OCR fails, rely on other heuristics
                pass
        
        # Default: if we got here, use edge and density as final criteria
        return edge_ratio > 0.05 or nonwhite_ratio > 0.2

    def extract_figures_from_page_image(
        self,
        page_image_path: str | Path,
        page_num: int,
        prefix: str = "fig",
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.50,
        use_multiple_strategies: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Advanced figure/chart detection from scanned page images using multiple CV strategies.
        Combines contour detection, morphological operations, and connected components.
        
        Args:
            page_image_path: Path to page image
            page_num: Page number
            prefix: Prefix for figure filenames
            min_area_ratio: Minimum area as ratio of page (default 2%)
            max_area_ratio: Maximum area as ratio of page (default 50%)
            use_multiple_strategies: Use multiple detection methods for better recall
            
        Returns:
            List of dictionaries containing figure metadata
        """
        import cv2
        import numpy as np
        
        page_image_path = Path(page_image_path)
        if not page_image_path.exists():
            raise FileNotFoundError(f"Page image not found: {page_image_path}")
        
        extracted_figures = []
        
        try:
            # Read the image
            img = cv2.imread(str(page_image_path))
            if img is None:
                raise ValueError(f"Could not read image: {page_image_path}")
            
            height, width = img.shape[:2]
            page_area = width * height
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate area thresholds
            min_area = page_area * min_area_ratio
            max_area = page_area * max_area_ratio
            
            # Collect candidate boxes from multiple strategies
            all_candidates = []
            
            # Strategy 1: Edge-based contour detection (works for bordered boxes)
            candidates_1 = self._detect_by_edges(gray, min_area, max_area, width, height)
            all_candidates.extend([(box, 'edge') for box in candidates_1])
            logger.debug(f"    Strategy 1 (edges): Found {len(candidates_1)} candidates")
            
            if use_multiple_strategies:
                # Strategy 2: Morphological operations (works for dense regions)
                candidates_2 = self._detect_by_morphology(gray, min_area, max_area, width, height)
                all_candidates.extend([(box, 'morph') for box in candidates_2])
                logger.debug(f"    Strategy 2 (morphology): Found {len(candidates_2)} candidates")
                
                # Strategy 3: Connected components (works for isolated illustrations)
                candidates_3 = self._detect_by_components(gray, min_area, max_area, width, height)
                all_candidates.extend([(box, 'component') for box in candidates_3])
                logger.debug(f"    Strategy 3 (components): Found {len(candidates_3)} candidates")
            
            # Merge and filter overlapping candidates
            unique_boxes = self._merge_overlapping_boxes(all_candidates, overlap_threshold=0.7)
            logger.debug(f"    After merging: {len(unique_boxes)} unique candidates")
            
            # Process each candidate
            for box_info in unique_boxes:
                x, y, w, h = box_info['bbox']
                
                # Add padding
                padding = 10
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(width - x_pad, w + 2 * padding)
                h_pad = min(height - y_pad, h + 2 * padding)
                
                # Crop the figure
                figure_img = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # Verify it's an illustration (skip OCR for speed)
                if not self.is_illustration(figure_img, check_text=False):
                    logger.debug(f"    Rejected candidate at ({x},{y}): not an illustration")
                    continue
                
                # Generate unique figure ID
                self.image_counter += 1
                figure_id = f"{prefix}_{self.image_counter:03d}"
                
                # Save figure
                figure_filename = f"{figure_id}.png"
                figure_path = self.output_dir / figure_filename
                cv2.imwrite(str(figure_path), figure_img)
                
                extracted_figures.append({
                    "image_id": figure_id,
                    "page_num": page_num,
                    "file_path": str(figure_path),
                    "filename": figure_filename,
                    "bbox": [x_pad, y_pad, x_pad+w_pad, y_pad+h_pad],
                    "width": w_pad,
                    "height": h_pad,
                    "format": "png",
                    "detection_method": box_info.get('method', 'unknown')
                })
                
                area_pct = (w_pad * h_pad) / page_area * 100
                logger.debug(f"    ✓ Extracted {figure_id}: {w_pad}x{h_pad} at ({x_pad},{y_pad}), "
                           f"area={area_pct:.1f}%, method={box_info.get('method', 'unknown')}")
            
            if len(extracted_figures) > 0:
                logger.info(f"  Page {page_num}: Extracted {len(extracted_figures)} figures from scanned page")
            else:
                logger.debug(f"  Page {page_num}: No figures detected")
            
        except Exception as e:
            logger.warning(f"Failed to extract figures from page {page_num}: {e}")
        
        return extracted_figures
    
    def _detect_by_edges(self, gray, min_area, max_area, page_width, page_height) -> List[Dict[str, Any]]:
        """
        Strategy 1: Detect figures using edge detection and contour finding.
        Works well for figures with clear borders/boundaries.
        """
        import cv2
        import numpy as np
        
        candidates = []
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with Canny
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic filtering
                if w >= self.min_width and h >= self.min_height:
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'method': 'edge'
                        })
        
        return candidates
    
    def _detect_by_morphology(self, gray, min_area, max_area, page_width, page_height) -> List[Dict[str, Any]]:
        """
        Strategy 2: Detect figures using morphological operations.
        Works well for dense diagrams and illustrations.
        """
        import cv2
        import numpy as np
        
        candidates = []
        
        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Morphological closing to connect nearby elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Opening to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w >= self.min_width and h >= self.min_height:
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'method': 'morphology'
                        })
        
        return candidates
    
    def _detect_by_components(self, gray, min_area, max_area, page_width, page_height) -> List[Dict[str, Any]]:
        """
        Strategy 3: Detect figures using connected component analysis.
        Works well for isolated illustrations and diagrams.
        """
        import cv2
        import numpy as np
        
        candidates = []
        
        # Otsu's thresholding for better binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Skip background (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if min_area <= area <= max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                if w >= self.min_width and h >= self.min_height:
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 <= aspect_ratio <= 5.0:
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'method': 'component'
                        })
        
        return candidates
    
    def _merge_overlapping_boxes(
        self, 
        candidates: List[tuple], 
        overlap_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Merge overlapping bounding boxes from different detection strategies.
        
        Args:
            candidates: List of (box_dict, method) tuples
            overlap_threshold: IoU threshold for considering boxes as duplicates
            
        Returns:
            List of unique box dictionaries
        """
        import cv2
        import numpy as np
        
        if not candidates:
            return []
        
        # Extract boxes
        boxes = []
        for box_dict, method in candidates:
            if isinstance(box_dict, dict) and 'bbox' in box_dict:
                x, y, w, h = box_dict['bbox']
                boxes.append({
                    'bbox': (x, y, w, h),
                    'area': box_dict.get('area', w * h),
                    'method': method
                })
            else:
                boxes.append(box_dict)
        
        if not boxes:
            return []
        
        # Sort by area (largest first)
        boxes.sort(key=lambda b: b.get('area', 0), reverse=True)
        
        # Non-maximum suppression
        keep = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = box1['bbox']
            keep.append(box1)
            used[i] = True
            
            # Check for overlaps with remaining boxes
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2 = boxes[j]['bbox']
                
                # Calculate IoU (Intersection over Union)
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersect_area = (ix2 - ix1) * (iy2 - iy1)
                    union_area = w1 * h1 + w2 * h2 - intersect_area
                    iou = intersect_area / union_area if union_area > 0 else 0
                    
                    # Mark as duplicate if high overlap
                    if iou > overlap_threshold:
                        used[j] = True
        
        return keep
    
    def reset_counter(self):
        """Reset the image counter"""
        self.image_counter = 0
