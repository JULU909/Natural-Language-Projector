import cv2
import numpy as np
import os 
import json
# List to store selected points
points = []
PERSPECTIVE_JSON_PATH = "temp_storage/perspective_data.json"
def select_points(event, x, y, flags, param):
    """Mouse callback function to capture four points for perspective transform."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)} selected: {x}, {y}")

def get_perspective_transform(frame, src_points):
    """Performs perspective transformation given 4 source points and returns the matrix."""
    width_a = np.linalg.norm(src_points[0] - src_points[1])
    width_b = np.linalg.norm(src_points[2] - src_points[3])
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(src_points[0] - src_points[3])
    height_b = np.linalg.norm(src_points[1] - src_points[2])
    max_height = max(int(height_a), int(height_b))
    max_width = 256*8
    max_height  = 64*8
    # Destination points for the perspective transform
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # dst pt size k256 x k64
    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply perspective transformation
    transformed = cv2.warpPerspective(frame, matrix, (max_width, max_height))

    return transformed, matrix

def save_perspective_matrix(image_path, matrix):
    """Saves the perspective matrix for an image in a JSON file."""
    if os.path.exists(PERSPECTIVE_JSON_PATH):
        with open(PERSPECTIVE_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Store the perspective matrix
    data[os.path.basename(image_path)] = matrix.tolist()

    # Save back to JSON
    with open(PERSPECTIVE_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Perspective matrix saved for {image_path}")



def calibration(save_path):
    points = camera_transform(save_path)
    return points


def camera_transform(save_path):
    """Opens the camera, selects points, applies perspective transform, and saves the image."""
    global points
    points = []  # Reset points

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Mark selected points

        cv2.imshow("Select Points", frame)

        if len(points) == 4:
            src_points = np.array(points, dtype=np.float32)
            cropped, matrix = get_perspective_transform(frame, src_points)

            cv2.imwrite(save_path, cropped)  # Save the transformed image


            # Save the perspective matrix

            break  # Exit loop after saving

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Reset points
            points = []
        elif key == ord('q'):  # Quit
            break
    # **Ensure proper cleanup**
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    return src_points


def capture_input(src_points, save_path):
    """Captures an image from the camera, applies perspective transform, and saves the transformed image."""
    
    cap = cv2.VideoCapture(0)  # Open webcam
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Capture a single frame
    ret, frame = cap.read()
    cap.release()  # Release the camera immediately after capturing
    
    if not ret:
        print("Error: Could not capture an image.")
        return None

    # Ensure src_points is a valid numpy array
    src_points = np.array(src_points, dtype=np.float32)
    if src_points.shape != (4, 2):
        print("Error: src_points must be a list of four (x, y) coordinates.")
        return None

    # Perform perspective transformation
    transformed, matrix = get_perspective_transform(frame, src_points)

    # Save the transformed image
    cv2.imwrite(save_path, transformed)
    print(f"Transformed image saved at: {save_path}")

    # Save the perspective matrix
    save_perspective_matrix(save_path, matrix)

    return transformed


if __name__ == "__main__":
    # Example usage:
    camera_transform("/home/jazz/Harish_ws/Demo/microled/agents/temp_storage/camera_input.png")
    