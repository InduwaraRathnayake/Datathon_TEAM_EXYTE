from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    # Clear GPU cache first
    torch.cuda.empty_cache()
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {gpu_memory:.1f}GB")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ===== Phase 1: Train detection head only =====
    print("🚀 Phase 1: Training detection head only (freeze backbone)...")
    model = YOLO("yolo11m.pt")

    results_phase1 = model.train(
        data="data.yaml",
        epochs=10,
        batch=4,                    # REDUCED from 16 to 4
        imgsz=640,
        device=0,
        workers=2,                  # REDUCED from 4 to 2
        project="pii_detection",
        name="face_license_detection_phase1",
        save=True,
        save_period=5,
        patience=5,
        verbose=True,
        pretrained=True,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        close_mosaic=5,
        freeze=[0],
        lr0=0.003,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,                  # DISABLED mixup to save memory
        auto_augment="randaugment",
        cache=False,                # DISABLED cache to save memory
        amp=True                    # Enable mixed precision for memory savings
    )

    # Clear cache after phase 1
    torch.cuda.empty_cache()

    # Evaluate after Phase 1
    print("📊 Evaluating after Phase 1...")
    metrics1 = model.val()
    print(f"[Phase 1] mAP50: {metrics1.box.map50:.4f}")
    print(f"[Phase 1] mAP50-95: {metrics1.box.map:.4f}")

    # Clear cache before phase 2
    torch.cuda.empty_cache()

    # ===== Phase 2: Fine-tune entire model =====
    print("🚀 Phase 2: Fine-tuning entire model (unfreeze backbone)...")
    model = YOLO("pii_detection/face_license_detection_phase1/weights/best.pt")

    results_phase2 = model.train(
        data="data.yaml",
        epochs=50,
        batch=4,                    # REDUCED from 16 to 4
        imgsz=640,
        device=0,
        workers=2,                  # REDUCED from 4 to 2
        project="pii_detection",
        name="face_license_detection_phase2",
        save=True,
        save_period=5,
        patience=10,
        verbose=True,
        pretrained=True,
        optimizer="SGD",
        seed=0,
        deterministic=True,
        close_mosaic=5,
        freeze=0,
        lr0=0.001,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.85,
        warmup_bias_lr=0.08,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.4,
        mixup=0.0,                  # DISABLED mixup to save memory
        auto_augment="randaugment",
        cache=False,                # DISABLED cache to save memory
        amp=True                    # Enable mixed precision for memory savings
    )

    # Evaluate after Phase 2
    print("📊 Evaluating after Phase 2...")
    metrics2 = model.val()
    print(f"[Phase 2] mAP50: {metrics2.box.map50:.4f}")
    print(f"[Phase 2] mAP50-95: {metrics2.box.map:.4f}")
    
    # Export the best model
    print("Exporting best model...")
    model.export(format="onnx")
    
    # Test on a sample image
    if os.path.exists("stage1_face_detection/dataset/infer/test_image1.jpg"):
        print("Testing on sample image...")
        results = model("stage1_face_detection/dataset/infer/test_image1.jpg")
        results[0].show()
        results[0].save("stage1_face_detection/dataset/infer/detection/detection_result.jpg")
    else:
        print("No test image found. Place a test image as 'test_image1.jpg' to test.")