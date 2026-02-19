import json, os
import cv2

CONFIG_PATH = os.path.join("assets", "mockup_config.json")

points = []
img = None

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, str(len(points)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calibrate", img)

def main(mockup_path: str):
    global img, points
    name = os.path.basename(mockup_path)

    img = cv2.imread(mockup_path)
    if img is None:
        raise FileNotFoundError(mockup_path)

    cv2.imshow("calibrate", img)
    cv2.setMouseCallback("calibrate", click_event)

    print("Click 4 points in order: TL, TR, BR, BL. Press 's' to save, 'r' reset, 'q' quit.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            points = []
            img = cv2.imread(mockup_path)
            cv2.imshow("calibrate", img)
        if k == ord("s"):
            if len(points) != 4:
                print("Need exactly 4 points.")
                continue

            cfg = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)

            cfg[name] = {
                "quad": points,
                "shadow": {"blur": 10, "opacity": 55, "offset": [3, 4]}
            }

            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)

            print(f"Saved quad for {name} -> {CONFIG_PATH}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
