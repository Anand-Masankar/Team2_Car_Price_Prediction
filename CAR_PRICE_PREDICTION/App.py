import sys
sys.path.append("D:\\Assignments_Assesment_UHG\\UHG_Assignment_2\\CAR_PRICE_PREDICTION")
from model_training import training_data

model_instance = training_data.ModelBuilding()
model_instance.get_all_methods()