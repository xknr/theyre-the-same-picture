import cv2
import numpy as np

SRC_FN = 'ground.webp'
OUT_FN = 'recursive_hq.png'

def create_high_quality_meta(template_path, OUT_FN, iterations=5):
  img = cv2.imread(template_path)
  # Convert to float32 for better precision during recursion
  img = img.astype(np.float32) / 255.0
  
  h, w = img.shape[:2]
  src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

  # Coordinates for the papers
  left_paper = np.float32([[45, 120], [210, 115], [215, 280], [35, 290]])
  right_paper = np.float32([[260, 110], [430, 112], [440, 275], [265, 285]])

  for i in range(iterations):
    # Use INTER_LANCZOS4 for the perspective warp to handle the mapping better
    warped_l = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src_pts, left_paper), 
                                  (w, h), flags=cv2.INTER_LANCZOS4)
    warped_r = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src_pts, right_paper), 
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

  # Convert back to uint8
  final_img = (img * 255).astype(np.uint8)
  cv2.imwrite(OUT_FN, final_img)

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

create_high_quality_meta(SRC_FN, OUT_FN)
view_image(OUT_FN)

