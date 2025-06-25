import cv2
import os

def extract_frames_every_second(video_path, output_folder):
    """
    Extracts one frame every second from a video file and saves it to a specified folder.

    Args:
        video_path (str): The path to the input MP4 video file.
        output_folder (str): The path to the folder where frames will be saved.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Video duration: {duration_seconds:.2f} seconds")

    frame_count = 0
    frames_saved = 0
    last_saved_time_in_seconds = -1  # Initialize to ensure the first frame is saved

    print(f"Starting frame extraction from: {video_path}")

    while True:
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Calculate the current time in seconds
        current_time_in_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Check if it's time to save a frame (every full second mark)
        # We use a small epsilon to account for floating point inaccuracies
        if int(current_time_in_seconds) > last_saved_time_in_seconds:
            frame_filename = os.path.join(output_folder, f"frame_{int(current_time_in_seconds):04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames_saved += 1
            last_saved_time_in_seconds = int(current_time_in_seconds)
            print(f"Saved {frame_filename} at {current_time_in_seconds:.2f} seconds")

        frame_count += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {frames_saved}")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your MP4 video file
    video_file_path = "C:\\Users\\Sam\\Documents\\pool-tracker\\footage\\game_20250624_163320_cut.mp4"

    # IMPORTANT: Replace with the desired folder to save the frames
    output_frames_directory = "output_frames"
    extract_frames_every_second(video_file_path, output_frames_directory)