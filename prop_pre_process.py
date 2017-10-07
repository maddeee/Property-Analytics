import pandas as pd

def extract_info(d, regex ):
    return d.str.extract(regex)


def convert_to_y_n(res):
    for i in range(len(res.columns)):
        #print(i)
        res.iloc[:,i][res.iloc[:,i].notnull()==True] = 'Y'
        res.iloc[:,i][res.iloc[:,i].isnull()==True] = 'N'
    return res

def convert_to_whole(pExt,pType):
    rs_conv ={
              'Lac': 100000,
              'Crore':10000000,
              'Grounds': 2400,
              'Acres' :43560,
              'Ares':1076.39,
              'Cents':435.54
             }
    
    pExt.columns = range(pExt.shape[1])
    pType.columns = range(pType.shape[1])
    
    err=pd.DataFrame(columns=['Index','Num','Type'])
    j = 0 

    for i in range(len(pExt)):
      if (pd.isnull(pExt.loc[i,0])): 
        continue  
      elif (pType.loc[i,0] not in rs_conv.keys()):
           err.loc[j,'Index'] = i
           err.loc[j,'Num'] = pExt.loc[i,0]
           err.loc[j,'Type'] = pType.loc[i,0]     
           j = j+1
      else:   
        pExt.loc[i,0] = pExt.loc[i,0] * rs_conv[pType.loc[i,0]]
        
    return pExt,pType,err     



    def main():
        """********************************************************************************"""
        """******************************Input files***************************************"""
        """********************************************************************************"""
        res_prop_summ = pd.read_csv ('G:/Machine Learning/Web Scraping/Property/99acres-property-list-9-24-2017 backup.csv')
        res_prop_details = pd.read_csv('G:/Machine Learning/Web Scraping/Property/property-details-op/99acres-property-details-list-9-24-2017.csv')
        
        pre_process_prop_summ(res_prop_summ)
        pre_process_prop_details(res_prop_details)        
        
        print('^^^^^^^^^^^^^^^^^^^^^^^^end ^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    def pre_process_prop_summ(prop_summ):
                    
            prop_summ = pd.read_csv ('G:/Machine Learning/Web Scraping/Property/99acres-property-list-9-24-2017 backup.csv')
            """********************************************************************************"""
            """***************Pre-Processing Residential summary data**************************"""
            """********************************************************************************"""
            print('pre-processing for residential summary file')
            #extract the property type
            
            regex = r"([a-zA-Z]+) ([a-zA-Z/]+)|([a-zA-Z/]+) in"
            
            ps = extract_info(prop_summ['name'],regex)
            
            prop_summ['prop_type'] = ps[0] +" "+ ps[1]
            
            
            #extract the residential area
            print('*********extracting residential area *********')
            regex = r"in ([a-zA-Z]+) ([a-zA-Z]+)|in ([a-zA-Z]+)"
            
            ps = extract_info(prop_summ['name'],regex)
            prop_summ['residential_area'] = ps[0].fillna('') +" "+ ps[1].fillna('') + ps[2].fillna('')
            
            
            #Extract the price of the apartment
            print('*********extracting apartment price *********')
            regex = r"(\d+\.?\d{1,2}?) ([a-zA-Z]+)"
            
            ps = extract_info(prop_summ['price'],regex)
            type(ps[0])
            
            prop_summ['price_ext'] = pd.to_numeric(ps[0])
            prop_summ['price_type'] = ps[1]
           
            ## convert all price to rs 
            ## convert lac, crore to whole numbers
           
            p_ext,p_type,err = convert_to_whole(prop_summ.loc[:,['price_ext']],prop_summ.loc[:,['price_type']])
            
            prop_summ['price_in_rs_ext'] = p_ext
            
            #extract the area of the plot
            print('*********extracting area of plot *********')
            
            regex = r"([\d\.?\d{1,2}?]+) ([a-zA-Z.]+) ([\d\.?\d{1,2}?]+)? ?([ a-zA-Z.]+)?"
            ps = extract_info(prop_summ['plot_area'],regex)
            
            ps[3] = ps[3].fillna('')
            
            for i in range(len(ps)):#len(ps)):
                if ps.loc[i,1] == 'to':
                    ps.loc[i,4] = ps.loc[i,3].strip()
                else:
                    ps.loc[i,4] = str(ps.loc[i,1]).strip() + str(ps.loc[i,3]).strip()
            
            prop_summ['area_ext'] = pd.to_numeric(ps[0])
            prop_summ['area_type'] = ps[4]
          
            ## convert all area to square feet
            
            a_ext,a_type,err = convert_to_whole(prop_summ.loc[:,['area_ext']],prop_summ.loc[:,['area_type']])
            
            prop_summ['area_ext_sqft'] = a_ext
            
            #check = prop_summ.loc[:,['area_ext','area_ext_whole','area_type']]
            #check1 = check[check['area_type']!='Sq.Ft.']
          
            #extract the price per sqft
            print('*********extracting price per sqft *********')
            
            regex = r"([\d\.?\d{1,2}?]+)/([a-zA-Z.]+)"
            ps = extract_info(prop_summ['plot_area_per_sq_sel'],regex)
            
            prop_summ['price_per_area'] = pd.to_numeric(ps[0])
            prop_summ['price_per_area_type'] = ps[1]
       
        
            regex = r"</span>([\d\.?\d{1,2}?]+)/ ([a-zA-Z.]+)"
            ps = extract_info(prop_summ['price'],regex)
            
            for i in range(len(prop_summ['price_per_area'])):
                if pd.isnull(prop_summ['price_per_area'][i]):
                   prop_summ.loc[i,['price_per_area']] = pd.to_numeric(ps.loc[i,0])
                if pd.isnull(prop_summ['price_per_area_type'][i]):
                   prop_summ.loc[i,['price_per_area_type']] = ps.loc[i,1]

            prop_summ['price_check'] = prop_summ['price_in_rs_ext']/(prop_summ['area_ext_sqft'] * prop_summ['price_per_area'])
            
            print('*********price checking in progress*********')
            #comparing the price of the property with the multiplication of the property price per sqft and the area of the plot. 
            #only if the difference is greater then the price of the property is replaced by the multiplication of price per sqft
            #and area of the plot
           
            check4 = prop_summ[prop_summ['price_check'] < 0.99]
            
            
            
            for i in range(len(prop_summ['price_in_rs_ext'])):
                if prop_summ['price_check'][i] > 1:
                   print (i)                    
                   prop_summ['price_in_rs_ext'][i]= prop_summ['area_ext'][i] * prop_summ['price_per_area'][i]
            
            prop_summ['pr_drvd_'] = prop_summ['area_ext'] * prop_summ['price_per_area']
            check3 = prop_summ['pr_drvd_'] ==prop_summ['price_in_rs_ext']
            
            p1=prop_summ[prop_summ['area_type']!='Sq.Ft.']
            
            prop_summ['check3'] = abs(prop_summ['price_in_rs_ext'] - prop_summ['pr_drvd_'])
            prop_summ['check4'] = prop_summ['check3']/prop_summ['price_in_rs_ext']
            
            p=prop_summ[prop_summ['check4']>1]
            
            #seller extract
            
            print('*********extracting seller *********')
            
            regex = r"(Dealer |Builder|Owner): (.*)"
            
            ps = extract_info(prop_summ['seller'],regex)
            
            prop_summ['seller_type'] = ps[0]
            prop_summ['seller_ext'] = ps[1]
            
            """*******************filter data*****************************"""
            fltr = [
            'extract_date',
            #'name',
            #'link',
            'id',
            #'price',
            #'area',
            #'plot_area',
            #'plot_area_per_sq_sel',
            'bld_name',
            'bdrm',
            'lat-long',
            #'seller',
            'prop_type',
            'residential_area',
            'price_ext',
            #'price_type',
            'price_in_rs_ext',
            'area_ext_sqft',
            'price_per_area',
            'price_per_area_type',
            'seller_type',
            'seller_ext'
            ]
            

    def pre_process_prop_details(prop_details):
                        
            """********************************************************************************"""
            """***************Pre-Processing Residential details data**************************"""
            """********************************************************************************"""
            
            print('******************************************************')
            print('*********processing residential details data *********')
            print('******************************************************')
            
            #transforming id
            
            print('*********extracting id *********')
            
            regex = r"([a-zA-Z0-9]+)"
            ps = extract_info(prop_details['id'],regex)
            
            prop_details['id'] = ps
            
            #bedroom number extract
            
            print('*********extracting bedroom num *********')
            
            regex = r"([\d])"
            ps = extract_info(prop_details['bedroom_num'],regex)
            prop_details['bedroom_num'] = ps
            
            #bathroom number extract
            
            print('*********extracting bathroom num *********')
            
            
            ps = extract_info(prop_details['bathroom_num'],regex)
            prop_details['bathroom_num'] = ps
            
            #extract number of balconies
            
            
            print('*********extracting balcony num *********')
            
            
            ps = extract_info(prop_details['no_balcny'],regex)
            prop_details['no_balcny'] = ps
            
            #preprocess the floornum to convert Ground to 0
            
            print('*********preprocess floor num *********')
            
            
            for i in range(len(prop_details['floor_num'])):
                
                prop_details.loc[i,'floor_num'] = str(prop_details.loc[i,'floor_num']).replace("Ground","0")
                prop_details.loc[i,'floor_num'] = str(prop_details.loc[i,'floor_num']).replace("Basement","99")
                prop_details.loc[i,'floor_num'] = str(prop_details.loc[i,'floor_num']).replace("Lower Ground","0")
            
            
            #extract floor number
            
            print('*********extracting floor num *********')
            
            regex = r"([\d]+)|([a-zA-Z]) ([\d]+)"
            
            ps = extract_info(prop_details['floor_num'],regex)
            
            
            prop_details['flr_num'] = ps[0].fillna('') + ps[2].fillna('')
            
            
            print('*********convert boolean lists to Y/N *********')
            
            es = prop_details.loc[:,'lift':'club_community_center']
            es = convert_to_y_n(es)
            
            ps = pd.concat([prop_details,es],axis=1)

