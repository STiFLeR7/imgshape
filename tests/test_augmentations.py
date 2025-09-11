from imgshape.augmentations import AugmentationRecommender
def test_recommend_low_entropy():
    stats = {"entropy_mean": 3.0, "colorfulness_mean": 8.0, "shape_distribution":{"square":0.5}, "edge_density":0.2, "class_balance":{"a":100,"b":100}}
    rec = AugmentationRecommender(seed=42)
    plan = rec.recommend_for_dataset(stats)
    names = [a.name for a in plan.augmentations]
    assert "ColorJitter" in names
