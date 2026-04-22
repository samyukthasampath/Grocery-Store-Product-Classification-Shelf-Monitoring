# Grocery-Store-Product-Classification-Shelf-Monitoring
🛒 Grocery Store Product Classification & Shelf Monitoring
Leeds School of Business – MSBC5190 Modern AI | Spring 2026

Built an end-to-end computer vision pipeline to automate grocery shelf compliance monitoring using YOLOv8.

🔍 What we built:
• Fine-tuned YOLOv8 to detect and classify 19 fresh produce categories from shelf images
• Developed a pixel-based shelf space estimator to measure planogram compliance (±15% threshold)
• Generated 400+ synthetic shelf images with automatic YOLO annotations to overcome the lack of public bounding box datasets
• Manually annotated 11 real-world grocery shelf images in Roboflow for ground-truth evaluation

📊 Results:
• mAP50 of 0.784 on synthetic test data with 100% product detection rate
• Near-perfect shelf space estimation for visually distinct classes (Apple: -0.2%, Banana: -0.1% error)
• Identified a 77-point synthetic-to-real domain gap (mAP50: 0.784 → 0.014), diagnosing three specific failure modes: class hallucination, fallback misclassification, and non-detection across visually complex real scenes

💡 Key takeaway: The system architecture is proven under controlled conditions. The primary path forward is fine-tuning on annotated real-world images to bridge the domain gap.

Tech stack: Python · YOLOv8 · Roboflow · AdamW · COCO pretraining · Synthetic data generation

#ComputerVision #MachineLearning #RetailTech #YOLOv8 #DeepLearning #MsBC
