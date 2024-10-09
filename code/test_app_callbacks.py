from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import pytest
import numpy as np
import main 


model = main.get_latest_model()

click = 1
year = 2014                        
fuel = 'Diesel'
seller_type = 'Individual'
transmission = 'Manual'   
owner = 'First Owner'       
engine = 1248            
max_power =  78               



sample = np.array([[ 0.92753623,  0.,  1.,  0. ,  1.,1., -0.41967976, -0.49691849]])

def test_model_output_shape():
    output = model.predict(sample)
    assert output.shape == (1,), f"Expecting the shape to be (1,) but got {output.shape=}"

def test_model_coeff_shape():
    output = model.W
    assert output.shape == (8,4), f"Expecting the shape to be (8,4) but got {output.shape=}"

def test_perdict():
    output = main.predict(click=click,f=fuel,s=seller_type,t=transmission,o=owner,y=year,e=engine,m=max_power)