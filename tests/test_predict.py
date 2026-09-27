import pytest
from predict import predict_command
from train_final_model import train_final_model

@pytest.fixture(scope="session", autouse=True)

def trained_model():
    train_final_model()

def test_move_fd():
    prediction = predict_command(front=1.2, left=0.7)
    assert prediction == "Move-Forward"

def test_sharp_right_turn():
    prediction = predict_command(front=0.6, left=0.8)
    assert prediction == "Sharp-Right-Turn"

@pytest.mark.parametrize("front, left", [(-1.0, 0.7), (1.0, -0.7), (-1.0, -0.7)])

def test_negative_dist_error(front, left):
    with pytest.raises(ValueError, match="Расстояние не может быть отрицательным"):
        predict_command(front=front, left=left)
