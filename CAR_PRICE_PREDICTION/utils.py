import pickle
import json
import numpy as np


class CarPrice():
    def __init__(self,'name', 'year', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
       'seats'):
        self.name = name
        self.year = year
        self.km_driven=km_driven
        self.fuel = fuel
        self.transmission = transmission
        self.owner = owner
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.length=length
        self.width=width
        self.height=height
        self.curb_weight=curb_weight
        self.engine_type=engine_type
        self.num_of_cylinders=num_of_cylinders
        self.engine_size=engine_size
        self.fuel_system=fuel_system
        self.bore=bore
        self.stroke=stroke
        self.compression_ratio=compression_ratio
        self.horsepower=horsepower
        self.peak_rpm=peak_rpm
        self.city_mpg=city_mpg
        self.highway_mpg=highway_mpg

    def __load_model(self):
        with open(r'artifacts/linear_reg.pkl',"rb") as f:
            self.model=pickle.load(f)

        with open(r"artifacts/Column_Names.json","r") as f:
            self.project_=json.load(f)

        with open(r"artifacts/label_encoded_data.json","r") as f:
            self.label_encode_=json.load(f)
        
    def get_predicted_price(self):

        self.__load_model()
        test_array=np.zeros(self.model.n_features_in_)

        test_array[0] = self.symboling
        test_array[1] = self.normalized_losses
        test_array[2] = self.label_encode_['fuel_type'][self.fuel_type]
        test_array[3] = self.label_encode_['aspiration'][self.aspiration]
        test_array[4] = self.label_encode_['number_of_doors'][self.num_of_doors]
        test_array[5] = self.label_encode_['engine_location'][self.engine_location]
        test_array[6] = self.wheel_base
        test_array[7] = self.length
        test_array[8] = self.width
        test_array[9] = self.height
        test_array[10] = self.curb_weight
        test_array[11] = self.label_encode_['num_of_cylinders'][self.num_of_cylinders]
        test_array[12] = self.engine_size
        test_array[13] = self.bore
        test_array[14] = self.stroke
        test_array[15] = self.compression_ratio
        test_array[16] = self.horsepower
        test_array[17] = self.peak_rpm
        test_array[18] = self.city_mpg
        test_array[19] = self.highway_mpg
        test_array[self.project_['Column Names'].index('make_' + self.make)] = 1
        test_array[self.project_['Column Names'].index('body_style_' + self.body_style)] = 1
        test_array[self.project_['Column Names'].index('drive_wheels_' + self.drive_wheels)] = 1
        test_array[self.project_['Column Names'].index('engine_type_' + self.engine_type)] = 1
        test_array[self.project_['Column Names'].index('fuel_system_' + self.fuel_system)] = 1


        print("test array is ", test_array)
        predicted_price=self.model.predict([test_array])[0]
        print("predicted_price is :",predicted_price)


        return predicted_price 