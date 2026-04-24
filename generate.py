import cv2
import numpy as np
import os

SRC_FN = 'data/ground.webp'
DIFFERENCE_FN = 'data/difference.png'
OUT_FN = 'out/recursive_hq.png'
SCALE = 4

# Load difference image for mending writing
diff_img = cv2.imread(DIFFERENCE_FN, cv2.IMREAD_UNCHANGED)

# This function uses an "Inside-Out" Iterative Feedback loop to generate the Droste effect.
# Instead of rendering all 2^N nested copies individually (which would be 2^iterations work),
# we reuse the result of the previous iteration as the source for the next.
#
# Work: 2 * iterations (Linear)
# Result: 2^iterations total nested copies (Exponential)
#
# This "caching" strategy makes the recursion extremely efficient while allowing the 
# "mending" patch to propagate through all depth levels naturally.
def create_high_quality_meta(template_path, diff_patch=None, iterations=5):
  img = cv2.imread(template_path)
  # Upscale initial image for higher resolution "meta" effect
  img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)
  
  # Convert to float32 for better precision during recursion
  img = img.astype(np.float32) / 255.0
  
  h, w = img.shape[:2]
  src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

  # Coordinates for the papers (scaled)
  left_paper = np.float32([[55-2, -6-1], [370, 61], [321, 286], [6, 217]]) * SCALE
  right_paper = np.float32([[406-2, 51-1], [725+2, 121-1], [676+2, 353+1], [356-2, 282+1]]) * SCALE
  

  for i in range(iterations):
    # To emulate mipmapping and prevent aliasing artifacts (shimmering/noise) when 
    # downsampling, we pre-filter the source image with a Gaussian blur.
    # This is essential for high-ratio perspective downscales.
    sigma = 0.8  # Slight blur to keep copies smooth across recursive layers
    img_src = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    # Use INTER_LANCZOS4 from the pre-filtered source for maximum smoothness
    warped_l = cv2.warpPerspective(img_src, cv2.getPerspectiveTransform(src_pts, left_paper), 
                                  (w, h), flags=cv2.INTER_LANCZOS4)
    warped_r = cv2.warpPerspective(img_src, cv2.getPerspectiveTransform(src_pts, right_paper), 
                                  (w, h), flags=cv2.INTER_LANCZOS4)

    # Create feathered masks to prevent harsh edges
    mask_l = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask_l, left_paper.astype(np.int32), 1.0)
    # Optional: slight blur on the mask to anti-alias the edges of the paper
    mask_l = cv2.GaussianBlur(mask_l, (3, 3), 0)
    
    mask_r = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask_r, right_paper.astype(np.int32), 1.0)
    mask_r = cv2.GaussianBlur(mask_r, (3, 3), 0)

    # Blend using the masks
    for c in range(3):
      img[:, :, c] = img[:, :, c] * (1 - mask_l) + warped_l[:, :, c] * mask_l
      img[:, :, c] = img[:, :, c] * (1 - mask_r) + warped_r[:, :, c] * mask_r

    # Apply "mending" patch in each iteration to preserve writing
    if diff_patch is not None:
      x, y = 515 * SCALE, 323 * SCALE
      h_f, w_f = diff_patch.shape[:2]
      h_bound = min(h_f, img.shape[0] - y)
      w_bound = min(w_f, img.shape[1] - x)
      
      if h_bound > 0 and w_bound > 0:
        patch_roi = diff_patch[:h_bound, :w_bound]
        alpha = patch_roi[:, :, 3:4]
        color = patch_roi[:, :, :3]
        img[y:y+h_bound, x:x+w_bound] = img[y:y+h_bound, x:x+w_bound] * (1 - alpha) + color * alpha

  # Convert back to uint8 with clipping to prevent overflow noise from Lanczos filtering
  final_img = np.clip(img * 255, 0, 255).astype(np.uint8)
  return final_img

def view_image(fn):

  # Load the result you just saved
  result = cv2.imread(fn)

  if result is not None:
    cv2.imshow('Recursive Result', result)
    
    # Wait for any key press to close the window
    # This prevents the window from freezing or closing instantly
    print("Press any key on the image window to exit.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
  else:
    print("Error: Could not load the output image.")

if not os.path.exists('out'):
  os.makedirs('out')

# Prepare scaled difference patch (float32 for recursive blending)
diff_patch = None
if diff_img is not None:
  diff_patch = cv2.resize(diff_img, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
  diff_patch[:, :, :3] /= 255.0
  diff_patch[:, :, 3] /= 255.0
  # Ensure alpha is 3D for broadcasting: (H, W, 1)
  if len(diff_patch.shape) == 3 and diff_patch.shape[2] == 4:
    alpha = diff_patch[:, :, 3:4]
    color = diff_patch[:, :, :3]
    diff_patch = np.concatenate([color, alpha], axis=2)

final_img = create_high_quality_meta(SRC_FN, diff_patch=diff_patch, iterations=8)

cv2.imwrite(OUT_FN, final_img)
view_image(OUT_FN)

