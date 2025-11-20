"""
Streamlit app for drawing segmentation masks over uploaded images.

Features:
- Upload an image
- Freehand draw on a canvas with a selectable color and class id
- Register color <-> class mappings
- Generate a per-pixel class mask (shape = image H x W)
- Download mask as .npy and .png and preview combined overlay
- Run Random Forest pixel segmentation using drawn mask as training labels

Requires: streamlit, streamlit-drawable-canvas, pillow, numpy, scikit-image, scikit-learn
Install: pip install streamlit streamlit-drawable-canvas pillow numpy scikit-image scikit-learn
Run: streamlit run app.py
"""

import io
from typing import Tuple
from functools import partial

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
	return "#%02x%02x%02x" % tuple(int(x) for x in rgb)


def hex_to_rgb(hexcol: str) -> Tuple[int, int, int]:
	h = hexcol.lstrip('#')
	return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


st.set_page_config(page_title="Segmentation Mask Drawer", layout="wide")

st.title("Segmentation mask drawing tool")

if 'color2class' not in st.session_state:
	st.session_state.color2class = {}  # hex -> int
if 'class2color' not in st.session_state:
	st.session_state.class2color = {}  # int -> hex
if 'rf_result' not in st.session_state:
	st.session_state.rf_result = None


with st.sidebar:
	st.header("Brush / class settings")
	st.write("This app uses 3 fixed classes. Select one to draw:")
	# Fixed 3-class palette
	fixed_palette = {1: "#ff0000", 2: "#00ff00", 3: "#0000ff"}
	sel_class = st.radio("Select class", options=[1, 2, 3], index=0)
	brush_color = fixed_palette[int(sel_class)]
	stroke_width = st.slider("Stroke width (px)", 1, 200, 15)

	st.markdown("---")
	st.subheader("Legend")
	for cid, hexc in fixed_palette.items():
		st.write(f"Class {cid}: {hexc}")

	st.markdown("---")
	st.subheader("Random Forest Segmentation")
	st.write("Use your drawn mask as training labels to segment the entire image.")
	
	sigma_min = st.slider("Sigma min", 1, 10, 1, help="Minimum scale for feature extraction")
	sigma_max = st.slider("Sigma max", 2, 32, 16, help="Maximum scale for feature extraction")
	n_estimators = st.slider("Number of trees", 10, 200, 50, help="Number of trees in Random Forest")
	max_depth = st.slider("Max depth", 5, 30, 10, help="Maximum tree depth")
	
	run_rf = st.button("🚀 Run Random Forest Segmentation", use_container_width=True)
	
	st.markdown("---")
	st.write("Export")


uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded is None:
	st.info("Upload an image to start drawing a mask.")
	st.stop()

# Load image
image = Image.open(uploaded).convert("RGB")
orig_w, orig_h = image.size

# limit display size for usability but keep original for saving
MAX_DIM = 1024
scale = min(1.0, MAX_DIM / max(orig_w, orig_h))
display_w, display_h = int(orig_w * scale), int(orig_h * scale)
display_image = image.resize((display_w, display_h), resample=Image.LANCZOS)

col1, col2 = st.columns([2, 1])

with col1:
	st.subheader("Canvas")
	# pass PIL Image as background to st_canvas (avoids numpy truth-value error)
	canvas_result = st_canvas(
		stroke_width=stroke_width,
		stroke_color=brush_color,
		background_image=display_image,
		height=display_h,
		width=display_w,
		drawing_mode="freedraw",
		key="canvas",
	)

	st.caption("Draw freehand strokes on the image. Switch brush color/class in the sidebar to label different classes.")

with col2:
	st.subheader("Preview / Actions")
	# Process canvas image to mask
	mask = None
	if canvas_result.image_data is not None:
		canvas_img = (canvas_result.image_data).astype(np.uint8)
		canvas_rgb = canvas_img[..., :3]

		# background as numpy
		bg_rgb = np.array(display_image).astype(np.uint8)

		# detect drawn pixels by RGB distance from background
		diff = np.linalg.norm(canvas_rgb.astype(int) - bg_rgb.astype(int), axis=2)
		tol = 15
		drawn = diff > tol

		# small mask for display size
		mask_small = np.zeros((display_h, display_w), dtype=np.uint8)

		ys, xs = np.nonzero(drawn)
		if ys.size > 0:
			# precompute palette RGBs
			palette_rgb = {cid: np.array([int(v) for v in hex_to_rgb(col)], dtype=int) for cid, col in fixed_palette.items()}
			# for each drawn pixel, find nearest palette color (handles anti-aliasing/blending)
			for y, x in zip(ys, xs):
				pix = canvas_rgb[y, x].astype(int)
				best_cid = 0
				best_dist = 1e9
				for cid, prgb in palette_rgb.items():
					d = np.linalg.norm(pix - prgb)
					if d < best_dist:
						best_dist = d
						best_cid = cid
				# optional threshold to avoid assigning background-like colors
				if best_dist < 200:
					mask_small[y, x] = int(best_cid)

		# resize back to original image size
		if scale < 1.0:
			pil_mask_small = Image.fromarray(mask_small.astype('uint8'))
			pil_mask = pil_mask_small.resize((orig_w, orig_h), resample=Image.NEAREST)
			mask = np.array(pil_mask, dtype=np.uint8)
		else:
			mask = mask_small

		st.write(f"Mask shape: {mask.shape}")

		# Build an overlay image (RGB) from mask and fixed palette
		overlay = np.array(image).astype(float)
		color_overlay = np.zeros_like(overlay)
		for cid, hexc in fixed_palette.items():
			rgb = np.array(hex_to_rgb(hexc), dtype=np.uint8)
			color_overlay[mask == int(cid)] = rgb

		alpha = 0.5
		blended = overlay.copy()
		mask_bool = mask > 0
		blended[mask_bool] = (alpha * color_overlay[mask_bool] + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)

		st.image([image, color_overlay.astype(np.uint8), blended.astype(np.uint8)], caption=["Original", "Mask colors", "Overlay"], use_column_width=True)

		# Random Forest Segmentation
		if run_rf:
			if np.sum(mask > 0) == 0:
				st.error("Please draw some training labels first before running segmentation!")
			else:
				with st.spinner("Running Random Forest segmentation..."):
					try:
						# Extract features from the original image
						img_array = np.array(image)
						
						features_func = partial(
							feature.multiscale_basic_features,
							intensity=True,
							edges=False,
							texture=True,
							sigma_min=sigma_min,
							sigma_max=sigma_max,
							channel_axis=-1,
						)
						
						features = features_func(img_array)
						
						# Train classifier
						clf = RandomForestClassifier(
							n_estimators=n_estimators,
							n_jobs=-1,
							max_depth=max_depth,
							max_samples=0.05,
							random_state=42
						)
						clf = future.fit_segmenter(mask, features, clf)
						
						# Predict on entire image
						result = future.predict_segmenter(features, clf)
						
						st.session_state.rf_result = result
						st.success("Segmentation complete!")
						
					except Exception as e:
						st.error(f"Error during segmentation: {str(e)}")
		
		# Display RF results if available
		if st.session_state.rf_result is not None:
			st.markdown("---")
			st.subheader("Random Forest Segmentation Result")
			
			rf_result = st.session_state.rf_result
			
			# Build overlay for RF result
			rf_overlay = np.array(image).astype(float)
			rf_color_overlay = np.zeros_like(rf_overlay)
			for cid, hexc in fixed_palette.items():
				rgb = np.array(hex_to_rgb(hexc), dtype=np.uint8)
				rf_color_overlay[rf_result == int(cid)] = rgb
			
			rf_blended = rf_overlay.copy()
			rf_mask_bool = rf_result > 0
			rf_blended[rf_mask_bool] = (alpha * rf_color_overlay[rf_mask_bool] + (1 - alpha) * rf_overlay[rf_mask_bool]).astype(np.uint8)
			
			st.image([rf_color_overlay.astype(np.uint8), rf_blended.astype(np.uint8)], 
					 caption=["RF Segmentation", "RF Overlay"], 
					 use_column_width=True)
			
			# Download RF result
			buf_rf_npy = io.BytesIO()
			np.save(buf_rf_npy, rf_result)
			buf_rf_npy.seek(0)
			
			buf_rf_png = io.BytesIO()
			Image.fromarray(rf_result.astype('uint8')).save(buf_rf_png, format='PNG')
			buf_rf_png.seek(0)
			
			st.download_button("Download RF result (.npy)", data=buf_rf_npy, file_name="rf_segmentation.npy", mime="application/octet-stream")
			st.download_button("Download RF result (.png)", data=buf_rf_png, file_name="rf_segmentation.png", mime="image/png")
			
			if st.button("Clear RF Result"):
				st.session_state.rf_result = None
				st.rerun()

		# downloads for manual mask
		st.markdown("---")
		st.subheader("Manual Mask Downloads")
		buf_npy = io.BytesIO()
		np.save(buf_npy, mask)
		buf_npy.seek(0)

		buf_png = io.BytesIO()
		Image.fromarray(mask.astype('uint8')).save(buf_png, format='PNG')
		buf_png.seek(0)

		st.download_button("Download mask (.npy)", data=buf_npy, file_name="mask.npy", mime="application/octet-stream")
		st.download_button("Download mask (.png) - pixel values are class ids", data=buf_png, file_name="mask.png", mime="image/png")

	else:
		st.write("Draw on the canvas to create a mask.")