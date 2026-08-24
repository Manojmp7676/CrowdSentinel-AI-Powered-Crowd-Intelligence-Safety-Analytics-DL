import yaml
from src.inference.predictor import CrowdDensityPredictor
from src.video.processor import VideoProcessor
from src.analysis.risk import CrowdRiskAnalyzer

config = yaml.safe_load(open("config_train.yaml", "r", encoding="utf-8"))
print("Config loaded")

predictor = CrowdDensityPredictor(config["model"]["best_model_path"])
print("Predictor initialized")

processor = VideoProcessor(predictor, config)
print("VideoProcessor ready")
print("Processing test video...")

results = processor.process_video("outputs/test_crowd.mp4", output_dir="outputs")
print("Video processed")

analyzer = CrowdRiskAnalyzer(config)
analysis = analyzer.analyze_video_results(results)
analyzer.save_analysis(analysis, "outputs/analysis_report.json")

print()
print("=== Results ===")
print("Avg count:", analysis["crowd_metrics"]["avg_estimated_count"])
print("Peak count:", analysis["crowd_metrics"]["peak_estimated_count"])
print("Avg density:", analysis["crowd_metrics"]["avg_density"])
print("Risk score:", analysis["risk_assessment"]["risk_score"])
print("Risk level:", analysis["risk_assessment"]["risk_level"])
print("Crowd status:", analysis["risk_assessment"]["crowd_status"])
print("Output video:", results["output_video"])
