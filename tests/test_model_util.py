import pytest
from models.utils import Model

@pytest.fixture
def model_name():
    return "Qwen/Qwen2-0.5B"

# This test takes a significant amount of time, so try not to repeatedly run it
# def test_model_forward(model_name):
#     model = Model(model_name)
#     print(model("This is a statement, isnt it?"))