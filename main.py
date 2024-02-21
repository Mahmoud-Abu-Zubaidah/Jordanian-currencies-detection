import cv2
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient

API_KEY= "ieQ1KiYMQlkOqP9GKP8G"
PROJECT = "money-classifier-diff-spaces"

def count_coins_from_image(image_path):
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT)
    model = project.version("3").model

    results = model.predict(image_path, confidence=50, overlap=30).json()

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image")
        return

    color_map = {
        "20-dinar": (255, 0, 0),   # Red
        "10-dinar": (0, 255, 0),   # Green
        "1-dinar": (0, 0, 255),    # Blue
        "5-coins": (255, 255, 0),  # Yellow
        "50-dinar": (255, 0, 255), # Purple
        "10-coins": (0, 255, 255), # Cyan
        "50-coins": (255, 255, 255), # White
        "5-dinar": (128, 0, 128),  # Purple
        "25-coins": (0, 128, 128)  # Teal
    }

    total_sum = 0
    
    for res in results["predictions"]:
        coin_class = res["class"]
        if coin_class in color_map:
            color = color_map[coin_class]
        else:
            color = (0, 0, 0)  # Default color

        coin_value = get_coin_value(coin_class)
        if coin_value is not None:
            total_sum += coin_value

        draw_bbox_and_label(image, res, color)
    num_objects = len(results["predictions"])
    display_sum(image, num_objects, total_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_coin_value(coin_class):
    coin_values = {
        "20-dinar": 20.0,
        "10-dinar": 10.0,
        "1-dinar": 1.0,
        "5-coins": 0.05,
        "50-dinar": 50.0,
        "10-coins": 0.10,
        "50-coins": 0.50,
        "5-dinar": 5.0,
        "25-coins": 0.25
    }
    return coin_values.get(coin_class)

def draw_bbox_and_label(image, res, color):
    x, y, width, height = res["x"], res["y"], res["width"], res["height"]
    x1, y1 = int(x - width / 2), int(y - height / 2)
    x2, y2 = int(x + width / 2), int(y + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    confidence = round(res["confidence"] * 100, 2)
    label = f"{res['class']} - {confidence}%"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def display_sum(image, num_objects, total_sum):
    height, width, _ = image.shape
    text = f"{num_objects} objects - Sum: {round(total_sum, 3)}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)
    text_x = width - text_size[0] - 10
    text_y = height - 20
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5)
    cv2.imshow("Detected Objects", image)
   
def count_coins_from_camera():
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Camera not detected.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Show the camera window
        cv2.imshow("Camera", frame)

        # Infer on the frame
        count_coins_from_frame(frame)  # Pass the frame directly

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def count_coins_from_frame(frame):
    # Initialize the client
    CLIENT = InferenceHTTPClient(
        api_url="http://detect.roboflow.com",
        api_key=API_KEY
    )

    # Infer on the provided image
    results = CLIENT.infer(frame, model_id=f"{PROJECT}/3")

    print("Detected coins:")
    for res in results["predictions"]:
        print(f"- {res['class']}")

    # Sum up the values of the detected coins
    total_sum = 0
    for res in results["predictions"]:
            coin_class = res["class"]
            coin_value = get_coin_value(coin_class)
            if coin_value is not None:
                total_sum += coin_value
                text = f"sum = {total_sum}"
                print(text)
   

def count_coins_from_video(video_path):
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT)
    model = project.version("3").model

    job_id, signed_url, expire_time = model.predict_video(video_path,
        fps=5,
        prediction_type="batch-video",
    )
    color_map = {
        "20-dinar": (255, 0, 0),   # Red
        "10-dinar": (0, 255, 0),   # Green
        "1-dinar": (0, 0, 255),    # Blue
        "5-coins": (255, 255, 0),  # Yellow
        "50-dinar": (255, 0, 255), # Purple
        "10-coins": (0, 255, 255), # Cyan
        "50-coins": (255, 255, 255), # White
        "5-dinar": (128, 0, 128),  # Purple
        "25-coins": (0, 128, 128)  # Teal
    }
    results = model.poll_until_video_results(job_id)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_sum = 0
        result = results[PROJECT][0]
        for res in result["predictions"]:
            coin_class = res["class"]
            if coin_class in color_map:
                color = color_map[coin_class]
            else:
                color = (0, 0, 0)  # Default color

            coin_value = get_coin_value(coin_class)
            if coin_value is not None:
                total_sum += coin_value

            draw_bbox_and_label(frame, res, color)
        num_objects = len(result["predictions"])
        display_sum(frame, num_objects, total_sum)
   

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    input_source = input("Enter 'image', 'video', or 'camera' to choose input source: ")
    if input_source == "image":
        image_path = input("Enter the path to the image: ")
        count_coins_from_image(image_path)
    elif input_source == "video":
        video_path = input("Enter the path to the video: ")
        count_coins_from_video(video_path)
    elif input_source == "camera":
        count_coins_from_camera()
    else:
        print("Invalid input source.")

if __name__ == "__main__":
    main()
