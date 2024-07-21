import cv2


def save_first_20_seconds(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # Codec para formato WEBM
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height),
                          True)  # Certifique-se de definir o contêiner corretamente

    frame_count = 0
    max_frames = int(fps * 20)  # 20 segundos de frames

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved the first 20 seconds of video to {output_path}")


if __name__ == "__main__":
    input_video_path = "../../data/processed/video.mp4"
    output_video_path = "../../data/processed/video_20s.mp4"
    save_first_20_seconds(input_video_path, output_video_path)
