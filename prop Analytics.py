import pandas as pd



def extract_info(d, regex ):
    return d.str.extract(regex)


def extract_num(res):
    return

res_prop_summ = pd.read_csv ('G:/Machine Learning/Web Scraping/Property/99acres-property-list-9-24-2017 backup.csv')

res_prop_details = pd.read_csv('G:/Machine Learning/Web Scraping/Property/property-details-op/99acres-property-details-list-9-24-2017.csv')


"""********************************************************************************"""
"""***************Pre-Processing Residential summary data**************************"""
"""********************************************************************************"""

#extract the property type

regex = r"([a-zA-Z]+) ([a-zA-Z/]+)|([a-zA-Z/]+) in"

ps = extract_info(res_prop_summ['name'],regex)

res_prop_summ['prop_type'] = ps[0] +" "+ ps[1]


#extract the residential area

regex = r"in ([a-zA-Z]+) ([a-zA-Z]+)|in ([a-zA-Z]+)"

ps = extract_info(res_prop_summ['name'],regex)

res_prop_summ['residential_area'] = ps[0].fillna('') +" "+ ps[1].fillna('') + ps[2].fillna('')


#Exxtract the price of the apartment
regex = r"(\d+\.\d{1,2}) ([a-zA-Z]+)"

ps = extract_info(res_prop_summ['price'],regex)

res_prop_summ['price_ext'] = ps[0]
res_prop_summ['price_type'] = ps[1]

#extract the area of the plot
regex = r"([\d]+) ([a-zA-Z.]+)"
ps = extract_info(res_prop_summ['plot_area'],regex)

res_prop_summ['area_ext'] = ps[0]
res_prop_summ['area_type'] = ps[1]

#extract the price per sqft

regex = r"([\d]+)/([a-zA-Z.]+)"
ps = extract_info(res_prop_summ['plot_area_per_sq_sel'],regex)

res_prop_summ['price_per_area'] = ps[0]
res_prop_summ['price_per_area_type'] = ps[1]

#seller extract

regex = r"(Dealer |Builder|Owner): (.*)"

ps = extract_info(res_prop_summ['seller'],regex)

res_prop_summ['seller_type'] = ps[0]
res_prop_summ['seller_ext'] = ps[1]


"""********************************************************************************"""
"""***************Pre-Processing Residential details data**************************"""
"""********************************************************************************"""

