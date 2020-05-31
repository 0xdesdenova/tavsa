import json

with open('atm_data.txt') as json_file:
    atm_data_object = json.load(json_file)

print('Missing coordinates')
for atm in atm_data_object:
    if atm_data_object[atm]['longitude'] == 0.0 or atm_data_object[atm]['latitude'] == 0.0:
        print(atm)

print('Missing distance vector')
for atm in atm_data_object:
    if 'distance_vector' not in atm_data_object[atm]:
        print(atm)