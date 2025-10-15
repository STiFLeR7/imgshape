# plugins/sample_plugin.py
from imgshape.plugins import AnalyzerPlugin, RecommenderPlugin

class SimpleEntropyAnalyzer(AnalyzerPlugin):
    NAME = "simple_entropy"
    def analyze(self, dataset_path: str):
        import os
        count = 0
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    count += 1
        return {"plugin": self.NAME, "image_count": count, "mean_entropy": 0.0}


class SimpleRecommender(RecommenderPlugin):
    NAME = "simple_recommender"
    def recommend(self, analysis: dict):
        steps = {"preprocessing": [], "augmentations": [], "meta": {"plugin": self.NAME}}
        if analysis.get("image_count", 0) > 100:
            steps["preprocessing"].append({"name": "resize", "spec": {"resize": [256, 256]}})
        else:
            steps["preprocessing"].append({"name": "resize", "spec": {"resize": [320, 320]}})
        return steps
