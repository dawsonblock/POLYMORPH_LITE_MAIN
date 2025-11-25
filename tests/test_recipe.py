from retrofitkit.core.recipe import Recipe
def test_load():
    r = Recipe.from_yaml("recipes/demo_gate.yaml")
    assert r.name == "demo_raman_gate"
    assert len(r.steps) >= 1
