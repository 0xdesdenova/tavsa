import json

with open('atm_data.txt') as json_file:
    atm_data_object = json.load(json_file)

route = ["TAV","1573","4239","4496","1388","3758","4498","3239","953","2405","1537","2330","3476","C272","C146","3667","1181","1844","2717","3367","4649","3604","4011","1842","2050","4485","2261","4642","C722","2474","4129","227","4567","3436","3285","4135","2377","1948","2774","723","931","1603","4097","2738","3554","1652","1594","149","318","C238","3686","3054","3151","3909","3925","C729","2852","CAY9","3899","C229","2997","4030","43","1330","4047","4548","C555","2950","3322","3993","1223","4365","3922","C672","C038","127","CAT1","4191","C844","630","1784","4240","1374","748","4349","3696","4061","4361","4516","2011","4504","1073","4280","3625","1712","3346","2327","3959","975","3429","4279","C701","2895","4390","4526","4334","4344","3980","4424","2924","4146","1829","3154","751","C518","2832","2401","3643","4696","1923","1523","3988","2748","3113","1044","3394","60","4690","4502","316","CAU2","4464","3345","2970","3629","3756","4264","C227","293","51","2914","133","2988","1744","C061","2118","3814","4121","81","4189"]

for atm in route:
    if atm not in atm_data_object:
        print(atm)