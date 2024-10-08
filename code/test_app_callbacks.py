import pytest
import numpy as np
import main 


model = main.get_latest_model()

sample = np.array([[ 0.92753623,  0.,  1.,  0. ,  1.,1., -0.41967976, -0.49691849]])

def test_model_output_shape():
    output = model.predict(sample)
    assert output.shape == (1,-1), f"Expecting the shape to be (1,1) but got {output.shape=}"

def test_model_coeff_shape():
    output = model.W
    assert output.shape == (8,4), f"Expecting the shape to be (1,2) but got {output.shape=}"
