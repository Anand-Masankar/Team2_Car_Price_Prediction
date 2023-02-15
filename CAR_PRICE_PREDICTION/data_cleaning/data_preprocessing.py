import datetime
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
print("library imported")

car_price_df = pd.read_csv(r'D:\Assignments_Assesment_UHG\UHG_Assignment_2\CAR_PRICE_PREDICTION\Car details.csv')

# D:\Persistent_Work\Assignment_2\Car details.csv

class DataCleaning():

    def __init__(self):

        self.onehot_encoder = None
        self.label_encoder = None
        self.features = None
        
        print("csv read")

    def feature_splitting(self, car_price_df):
        # df = pd.read_csv('D:\Persistent_Work\Assignment\Car details.csv')
        """
            Method Name/Function Name >> feature_splitting
            Description >> This method splits the "name" column and takes the "manufacturer" name from it. 
            For this we use split function from the pandas
            Parameters >> Dataframe
            Return >> Pandas dataframe with the manufacturer name
        """
        name_feature_split = car_price_df["name"].str.split(" ", expand = True, n = 1)
        car_price_df['manufacturer'] = name_feature_split[0]
        # print("feature_splitting")
        return car_price_df
        
        # print("unique values present inside the features :", name_df_1[0].nunique())
        # name_df_1[0].value_counts(normalize=True)*100

    def feature_creation_car_age(self,car_price_df):
        """
            Method Name/Function Name >> feature_creation_car_age
            Description >> This method creates the function car age from the buying year to current year.
            Parameters >> Dataframe
            Return >> Pandas dataframe appended with the specified age of car
        """
        today = datetime.date.today() 
        yr = today.year
        print("<<<<<<<<<<<")
        car_price_df['car_age'] = yr - car_price_df['year']
        print("feature_creation_car_age execute")
        return car_price_df
    
    def feature_extraction_mileage(self,car_price_df):
        """
            Method Name/Function Name >> feature_extraction_mileage
            Description >> This method modifies mileage to corrected mileage and converts all the km/kg to kmpl, 
            thus bringing all the values in one single unit.
            Parameters >> Dataframe
            Return >> Pandas dataframe with the specified correct mileage after conversion of km/kg with kmpl
        """
        corrected_mileage = []
        for i in car_price_df.mileage:
            if str(i).endswith('km/kg'):
                i = i[:-6]
                i = float(i) * 1.40    # 1.40 is unit conversion value for km/kg to kmpl
                corrected_mileage.append(float(i))

            elif str(i).endswith('kmpl'):
                i = i[:-6]
                corrected_mileage.append(float(i))
        car_price_df['corrected_mileage_kmpl'] = pd.Series(corrected_mileage) 
        print("feature_extraction_mileage executed")  
        return car_price_df

    def feature_extraction_engine_size(self,car_price_df):
        """
            Method Name/Function Name >> feature_extraction_engine_size
            Description >> With an objective to convert engine size from object data type to float data type 
            this function eliminates cc from the string and converts the numerical values to float data type.
            Parameters >> Dataframe
            Return >> Pandas dataframe with the modifies engine size after conversion into float datatype
        """
        car_price_df["engine_size_cc"] = car_price_df["engine"].str.split(" ", expand = True, n =1)[0]
        car_price_df["engine_size_cc"] = car_price_df["engine_size_cc"].astype("float")
        print("feature_extraction_engine_size executed") 
        return car_price_df
    
    def feature_extraction_maximum_power(self,car_price_df):
        """
            Method Name/Function Name >> feature_extraction_maximum_power
            Description >> With an objective to convert engine size from object data type to float data type 
            this function eliminates bhp from the string and converts the numerical values to float data type.
            Parameters >> Dataframe
            Return >> Pandas dataframe with the modifies max power after conversion into float datatype
        """
        car_price_df['max_power_bhp'] = car_price_df['max_power'].str.replace(' bhp','')
        car_price_df['max_power_bhp'] = pd.to_numeric(car_price_df['max_power_bhp'])
        print("feature_extraction_maximum_power executed") 
        return car_price_df

    def drop_irrelevent_feature(self, car_price_df, feature_list):
        """
            Method Name/Function Name >> drop_irrelevent_feature
            Description >> This method drops the unwanted columns
            Parameters >> Dataframe
            Column_names >> List of columns that is required to drop
            Return >> pandas dataframe with the specified list of columns removed
        """
        car_price_df = car_price_df.drop(feature_list, axis=1)     # feature_list = ["name", mileage, torque,engine, max_power]
        print("drop_irrelevent_feature executed")
        return car_price_df

    def rename_features(self,car_price_df):
        """
            Method Name/Function Name >> rename_features
            Description >> This method renames few column names
            Parameters >> Dataframe
            Return >> Pandas dataframe with new column names after renaming
        """
        # print(df.columns) #####################################################
        car_price_df.rename({"fuel" : "fuel_type","seats" : "no_seats", "owner" : "owner_type", 
        "transmission" : "transmission_type"}, inplace = True, axis = 1)
        print("rename_features executed")
        # print(df.columns) ####################################################
        return car_price_df
    
    def missing_values_imputation(self, car_price_df):
        """
            Method Name/Function Name >> impute_missing_values
            Description >> This method replaces all the missing values in the dataframe using median
            Parameters >> Dataframe
            Return >> A dataframe which has all the missing values imputed
        """
        # print(list(df.iloc[: , list(np.where(df.isna().sum() > 0)[0]) ].columns))
        missing_value_features = list(car_price_df.iloc[: , list(np.where(car_price_df.isna().sum() > 0)[0]) ].columns)
        for i in missing_value_features:
            if i == 'corrected_mileage_kmpl':
                car_price_df['corrected_mileage_kmpl'].fillna(car_price_df['corrected_mileage_kmpl'].mean(), inplace = True)
            elif i == 'engine_size_cc':
                car_price_df['engine_size_cc'].fillna(car_price_df['engine_size_cc'].median(), inplace = True)
            elif i == 'max_power_bhp':
                car_price_df['max_power_bhp'].fillna(car_price_df['max_power_bhp'].mean(), inplace = True)
            elif i == "no_seats":
                car_price_df['no_seats'].fillna(car_price_df['no_seats'].mode()[0], inplace = True)
        print("missing_values_imputation executed")
        
        return car_price_df 
        

    def feature_encoding(self, car_price_df):
        """
            Method Name/Function Name >> feature_encoding
            Description >> This method performs one hot encoding on fuel type, seller type and manufacture 
            and label encoding on transmission type and owner type
            Parameters >> Dataframe
            Return >> Pandas dataframe with numerical data from categorical data with the use of one hot encoding and label encoding
        """
        # get_dummies - OneHotEncoding >> 'fuel_type','seller_type'
        car_price_df = pd.get_dummies(car_price_df, columns = ['fuel_type','seller_type','manufacturer'],drop_first= True, prefix = None)

        #OneHotEncoding >> "manufacturer"

        # self.onehot_encoder = OneHotEncoder()
        # encode_manufacturer = self.onehot_encoder.fit_transform(car_price_df[['manufacturer']])
        # manfacturer_df = pd.DataFrame(encode_manufacturer.toarray(),columns = car_price_df['manufacturer'].unique().tolist())
        # manfacturer_df.drop("Tata",inplace = True, axis = 1)  ##############################
        # print("man_df",manfacturer_df.columns)
        # car_price_df = car_price_df.join(manfacturer_df, how='outer')
        
        
        # with open('onehot_encode_manfact.pkl', 'wb') as f:     # save in current directory onehot of manfacturer
        #     pickle.dump(encode_manufacturer, f)

        #LabelEncoding >>  "owner"
        self.label_encoder = LabelEncoder()
        label_encode_transmission = self.label_encoder.fit_transform(car_price_df["transmission_type"])
        car_price_df["transmission_type"] = label_encode_transmission

        with open('label_encode_transmission.pkl', 'wb') as f:
            pickle.dump(label_encode_transmission, f)

        #LabelEncoding >>  "owner"
        label_encode_owner = self.label_encoder.fit_transform(car_price_df["owner_type"])
        car_price_df["owner_type"] = label_encode_owner
        
        with open('label_encode_owner.pkl', 'wb') as f:
            pickle.dump(label_encode_owner, f)
        
        print("feature_encoding executed")
        return car_price_df


    
    # def get_all_methods(self,df):
    #     self.feature_extraction_engine_size(df)
    #     self.feature_extraction_maximum_power(df)
    #     self.drop_irrelevent_feature(df, ["name", 'mileage', 'torque','engine', 'max_power'])
    #     self.rename_features(df)
    #     self.missing_values_imputation(df)
    #     self.feature_encoding(df)



# model_instance = DataCleaning()
# model_instance.feature_splitting(df)
# model_instance.feature_creation_car_age(df)
# model_instance.feature_extraction_mileage(df)

# model_instance.get_all_methods(df)

