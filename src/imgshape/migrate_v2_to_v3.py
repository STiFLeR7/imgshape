# migrate_v2_to_v3.py
"""
Convert v2-style plan dicts -> v3 RecommendationPipeline JSON.
Usage:
    python migrate_v2_to_v3.py old_plan.json new_pipeline.json
"""
import sys, json
from imgshape.pipeline import RecommendationPipeline, PipelineStep

def migrate(old_path, out_path):
    with open(old_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = RecommendationPipeline()
    for p in data.get("preprocessing", []):
        rec.add_step(PipelineStep(p.get("name", "pre"), spec=p.get("spec", p)))
    for a in data.get("augmentations", []):
        rec.add_step(PipelineStep(a.get("name", "aug"), spec=a.get("spec", a)))
    rec.meta = data.get("meta", {})
    rec.save(out_path)
    print("Migrated ->", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python migrate_v2_to_v3.py old_plan.json new_pipeline.json")
    else:
        migrate(sys.argv[1], sys.argv[2])
