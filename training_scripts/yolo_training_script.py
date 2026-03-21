from ultralytics import YOLO
import os


import matplotlib.pyplot as plt
import cv2
import os


def show_training_report(results):
    """
    Prints a summary and displays the training results plot.
    """
    # 1. Print the key metrics to the console
    print("\n" + "=" * 30)
    print("      TRAINING REPORT")
    print("=" * 30)

    # Accessing the metrics from the results object
    # mAP50-95 is the standard 'fitness' score
    metrics = results.results_dict

    print(f"Precision: {metrics['metrics/precision(B)']:.4f}")
    print(f"Recall:    {metrics['metrics/recall(B)']:.4f}")
    print(f"mAP50:     {metrics['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95:  {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 30)

    # 2. Visually show the 'results.png' plot
    plot_path = os.path.join(results.save_dir, "results.png")
    if os.path.exists(plot_path):
        img = cv2.imread(plot_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Training Curves (Loss, Precision, Recall, mAP)")
        plt.show()
    else:
        print("Warning: results.png not found in the save directory.")
    F1_path = os.path.join(results.save_dir, "F1_curve.png")
    if os.path.exists(F1_path):
        img = cv2.imread(F1_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("F1 Curve")
        plt.show()
    else:
        print("Warning: F1_curve.png not found in the save directory.")


# --- Usage ---
# results = model.train(...)
# show_training_report(results)
if __name__ == "__main__":
    data_path = "C:/Users/medbe/OneDrive/Bureau/PFA2026/yolo_dataset"
    yaml_file = os.path.join(data_path, "dataset.yaml")
    model = YOLO("yolo26m.pt")
    # if you have i5 14th generation or more 6 workers is good for training,# if you have less than that you can set it to 0 or 2
    results = model.train(
        data=yaml_file,
        epochs=100,
        batch=8,
        imgsz=640,
        device="cuda",
        workers=6,
        scale=0.9,
        mixup=0.2,
        mosaic=0.3,
        copy_paste=0.2,
        flipud=0.5,
        fliplr=0.5,
        degree=20,
    )
    show_training_report(results)
