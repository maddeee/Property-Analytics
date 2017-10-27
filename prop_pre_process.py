import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
%matplotlib inline

    def main():
        """********************************************************************************"""
        """******************************Input files***************************************"""
        """********************************************************************************"""
        res_prop_summ = pd.read_csv ('G:/Machine Learning/Web Scraping/Property/99acres-property-list-9-24-2017 backup.csv')
        res_prop_details = pd.read_csv('G:/Machine Learning/Web Scraping/Property/property-details-op/99acres-property-details-list-9-24-2017.csv')
        
        prop_summ = pre_process_prop_summ(res_prop_summ)
        prop_det= pre_process_prop_details(res_prop_details)        
        
        prop_data = pd.merge(prop_summ, prop_det, how='inner', left_on= 'id',right_on='property_code' )
        
        print('^^^^^^^^^^^^^^^^^^^^^^^^end ^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    def pre_process_prop_summ(prop_summ):
                    
            #prop_summ = pd.read_csv ('G:/Machine Learning/Web Scraping/Property/99acres-property-list-9-24-2017 backup.csv')
            """********************************************************************************"""
            """***************Pre-Processing Residential summary data**************************"""
            """********************************************************************************"""
            
            prop_summ.set_index('id')
            
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
            regex = r"(\d+?\.?\d{1,2}?) ([a-zA-Z]+)"
            
            ps = extract_info(prop_summ['price'],regex)
            type(ps[0])
            
            prop_summ['price_ext'] = pd.to_numeric(ps[0])
            prop_summ['price_type'] = ps[1]
           
            ## convert all price to rs 
            ## convert lac, crore to whole numbers
           
            p_ext,p_type,err = convert_to_whole(prop_summ.loc[:,['price_ext']],prop_summ.loc[:,['price_type']],'m')
            
            prop_summ['price_in_rs_ext'] = p_ext
            
            #extract the area of the plot
            print('*********extracting area of plot *********')
            
            regex = r"([\d\.?\d{1,2}?]+) ([a-zA-Z.]+) ([\d\.?\d{1,2}?]+)? ?([ a-zA-Z.]+)?"
            ps = extract_info(prop_summ['plot_area'],regex)
            
            ps[3] = ps[3].fillna('')
            
            for i in range(len(ps)):
                if ps.loc[i,1] == 'to':
                    ps.loc[i,4] = ps.loc[i,3].strip()
                else:
                    ps.loc[i,4] = str(ps.loc[i,1]).strip() + str(ps.loc[i,3]).strip()
            
            prop_summ['area_ext'] = pd.to_numeric(ps[0])
            prop_summ['area_type'] = ps[4]
          
            ## convert all area to square feet
            
            a_ext,a_type,err = convert_to_whole(prop_summ.loc[:,['area_ext']],prop_summ.loc[:,['area_type']],'m')
            
            prop_summ['area_ext_sqft'] = a_ext
            
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

            ## Convert the price per ground, acre, ares etc to sqft price.
            
            p_a_ext,p_a_type,err = convert_to_whole(prop_summ.loc[:,['price_per_area']],prop_summ.loc[:,['price_per_area_type']],'d')
            
            prop_summ['price_per_area_sqft'] = p_a_ext
            
            print('*********price checking in progress*********')
            #comparing the price of the property with the multiplication of the property price per sqft and the area of the plot. 
            #if the price is not within 0.9 to 1.1 times the price mentioned in the ad, then mark the record as an error 
            #and leave exclude it from the model. 
           
            prop_summ['price_check'] = prop_summ['price_in_rs_ext']/(prop_summ['area_ext_sqft'] * prop_summ['price_per_area_sqft'])
            
            prop_summ['data_stat'] = list(' ' * len(prop_summ['price_check']) )
                        
            for i in range(len(prop_summ['price_check'])):
                if (prop_summ['price_check'][i] > 1.1 or prop_summ['price_check'][i] < 0.9):
                    print(i)
                    print('Err')
                    prop_summ.loc[i,'data_stat'] = 'Err'
                else:
                    prop_summ.loc[i,'data_stat'] = 'Good'
                    
        
            #seller extract
            
            print('*********extracting seller *********')
            
            regex = r"(Dealer |Builder|Owner): (.*)"
            
            ps = extract_info(prop_summ['seller'],regex)
            
            prop_summ['seller_type'] = ps[0]
            prop_summ['seller_ext'] = ps[1]
            
            prop_summ.columns
            
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
            #'price_ext',
            #'price_type',
            'price_in_rs_ext',
            #'area_ext',
            #'area_type',
            'area_ext_sqft',
            #'price_per_area',
            #'price_per_area_type',
            'price_per_area_sqft',
            #'price_check',
            'data_stat',
            'seller_type',
            'seller_ext'
            ]
            
            return prop_summ[fltr]

    

    def pre_process_prop_details(prop_details):
                        
            """********************************************************************************"""
            """***************Pre-Processing Residential details data**************************"""
            """********************************************************************************"""
            prop_details = pd.read_csv('G:/Machine Learning/Web Scraping/Property/property-details-op/99acres-property-details-list-9-24-2017.csv')
            
            prop_details.set_index('id')
            
            print('******************************************************')
            print('*********processing residential details data *********')
            print('******************************************************')
            
            #transforming id
            
            print('*********extracting id *********')
            
            regex = r"prop_([a-zA-Z0-9]+)"
            ps = extract_info(prop_details['id'],regex)
            
            prop_details['id_ext'] = ps
            
            #bedroom number extract
            
            print('*********extracting bedroom num *********')
            
            regex = r"([\d]+)([+])? ([a-zA-Z]+)"
            ps = extract_info(prop_details['bedroom_num'],regex)
            prop_details['bdrm_ext'] = pd.to_numeric(ps[0])
           
            #bathroom number extract
            
            print('*********extracting bathroom num *********')
            
            ps = extract_info(prop_details['bathroom_num'],regex)
            prop_details['bathrm_num_ext'] = pd.to_numeric(ps[0])
            
            #extract number of balconies
            
            print('*********extracting balcony num *********')
            
            ps = extract_info(prop_details['no_balcny'],regex)
            prop_details['no_balcny_ext'] = pd.to_numeric(ps[0])
            
            #preprocess the floornum to convert Ground to 0
            
            print('*********preprocess floor num *********')
            
            prop_details['floor_num_trfm'] = prop_details['floor_num']
            
            for i in range(len(prop_details['floor_num'])):
                
                prop_details.loc[i,'floor_num_trfm'] = str(prop_details.loc[i,'floor_num_trfm']).replace("Ground","0")
                prop_details.loc[i,'floor_num_trfm'] = str(prop_details.loc[i,'floor_num_trfm']).replace("Basement","99")
                prop_details.loc[i,'floor_num_trfm'] = str(prop_details.loc[i,'floor_num_trfm']).replace("Lower Ground","0")
            
            #extract floor number
            
            print('*********extracting floor num *********')
            
            #regex = r"([\d]+)|([\d]+) ([a-zA-Z]+) ([\d]+)?"
            
            regex = r"([\d]+) ?([a-zA-Z]+)? ?([\d])?"
            
            ps = extract_info(prop_details['floor_num_trfm'],regex)
            
            
            prop_details['flr_num_ext'] = ps[0]
            
            #p=pd.unique(prop_details.loc[:,['flr_num_ext','floor_num_trfm']].values)
             
            
            print('*********extract number of reserved parking *********')
            
            regex = r"([""])?([\d]+) ([a-zA-Z]+)([,])? ?([\d]+)? ?([a-zA-Z]+)?"
                        
            ps = extract_info(prop_details['reserved_parking'],regex)
            
            ps['orig_col'] = prop_details['reserved_parking'] 
            
            
            ps[0]=ps[0].fillna(0)
            ps[3]=ps[3].fillna(0)
            
            
            ps['cvrd_parking'] = pd.to_numeric(ps[0].tolist())
            ps['open_parking'] = pd.to_numeric(ps[0].tolist())
            
            ps['cvrd_parking'] = 0
            ps['open_parking'] = 0 
            
            for i in range(len(ps[0])):
                if ps[1][i] == 'Covered':
                    ps.loc[i,'cvrd_parking'] = ps.loc[i,'cvrd_parking'] + pd.to_numeric(ps[0][i])
                elif ps[4][i] == 'Covered':
                    ps.loc[i,'cvrd_parking'] = ps.loc[i,'cvrd_parking'] + pd.to_numeric(ps[3][i])
                    
            for i in range(len(ps[0])):
                if ps[1][i] == 'Open':
                    ps.loc[i,'open_parking'] = ps.loc[i,'open_parking'] + pd.to_numeric(ps[0][i])
                elif ps[4][i] == 'Open':
                    ps.loc[i,'open_parking'] = ps.loc[i,'open_parking'] + pd.to_numeric(ps[3][i])
                
            prop_details['cvrd_parking'] =  ps['cvrd_parking']
            prop_details['open_parking'] =  ps['open_parking']
            
            
            print('*********convert boolean lists to Y/N *********')
            
            es = prop_details.loc[:,'lift':'club_community_center']
            es = convert_to_y_n(es)
            es_col = es.columns.tolist()
            
            type(es_col)
            
            for i in range(len(es_col)):
                print(es_col[i])
                es_col[i] = str(es_col[i]) + '_trfm'
        
            es.columns = es_col
            
            prop_det_proc = pd.concat([prop_details,es],axis=1)
            
            g = copy.deepcopy(prop_details['gated_cmty'])
            
            prop_det_proc.loc[:,'gated_cmty_trfm'] = convert_to_y_n(g)
        
            """
            trfm_check = [
                            'club_community_center',
                            'club_community_center_trfm',
                            'feng_shui_vastu',
                            'feng_shui_vastu_trfm',
                            'garden',
                            'garden_trfm',
                            'gym',
                            'gym_trfm',
                            'id_ext',
                            'intercom',
                            'intercom_trfm',
                            'lift',
                            'lift_trfm',
                            'maintenance_staff',
                            'maintenance_staff_trfm',
                            'park',
                            'park_trfm',
                            'piped_gas',
                            'piped_gas_trfm',
                            'security_personal',
                            'security_personal_trfm',
                            'swimming_pool',
                            'swimming_pool_trfm',
                            'visitor_parking',
                            'visitor_parking_trfm',
                            'waste_disposal',
                            'waste_disposal_trfm',
                            'water_purifier',
                            'water_purifier_trfm',
                            'water_softner',
                            'water_softner_trfm',
                            'water_storage',
                            'water_storage_trfm',
                            'gated_cmty',
                            'gated_cmty_trfm'
                            ]
            """
             
            #p = prop_det_proc[trfm_check]
            #prop_det_proc.columns.tolist()
            
            fltr = [
                    'extract_date',
                    'property_code',
                    'id_ext',
                     #'built_up_area',
                     #'id',
                     'Carpet_area',
                     #'bedroom_num',
                     #'bathroom_num',
                     #'no_balcny',
                     #'floor_num',
                     'facing_direction',
                     'overlooking',
                     #'property_age',
                     'transaction_type',
                     'property_owner',
                     'flooring_type',
                     'furnish_type',
                     #'gated_cmty',
                     #'reserved_parking',
                     #'water_source',
                     'power_backup',
                     #'lift',
                     #'intercom',
                     #'garden',
                     #'water_purifier',
                     #'park',
                     #'maintenance_staff',
                     #'visitor_parking',
                     #'water_storage',
                     #'swimming_pool',
                     #'security_personal',
                     #'water_softner',
                     #'gym',
                     #'waste_disposal',
                     #'feng_shui_vastu',
                     #'piped_gas',
                     #'club_community_center',
                     'bdrm_ext',
                     #'bathrm_num',
                     'bathrm_num_ext',
                     'no_balcny_ext',
                     #'floor_num_ext',
                     #'floor_num_trfm',
                     'cvrd_parking',
                     'open_parking',
                     'flr_num_ext',
                     'lift_trfm',
                     'intercom_trfm',
                     'garden_trfm',
                     'water_purifier_trfm',
                     'park_trfm',
                     'maintenance_staff_trfm',
                     'visitor_parking_trfm',
                     'water_storage_trfm',
                     'swimming_pool_trfm',
                     'security_personal_trfm',
                     'water_softner_trfm',
                     'gym_trfm',
                     'waste_disposal_trfm',
                     'feng_shui_vastu_trfm',
                     'piped_gas_trfm',
                     'club_community_center_trfm',
                     'gated_cmty_trfm'
                     ]

            return prop_det_proc[fltr]


            
        
def prop_analytics(prop_data):        
    
    
    sns.set_style("whitegrid")
    sns.set_context("paper")

    prop = prop_data[prop_data['data_stat']=='Good']
    
    prop.describe()
    
    
    
    prop['bdrm'][prop['bdrm'].isnull()] = round(prop['bdrm'].mean(),0)

    prop_price_na=prop[prop['price_in_rs_ext'].isnull()]
    
    = round(prop['bdrm'].mean(),0)

"""
    number_cols = ['bdrm','price_in_rs_ext','area_ext_sqft','price_per_area_sqft','Carpet_area','bdrm_ext','bathrm_num_ext','no_balcny_ext','cvrd_parking','open_parking']

    plot_var(prop[number_cols])
    
    IQR_out_range(prop.loc[:,'bdrm'].dropna(),25,75)
    
    sd_outlier(prop.loc[:,'bdrm'].dropna(),3)
    
    df_w_outliers=outlier_IQR (prop[number_cols],25,75)
    
    df_sd_outliers = sd_outlier(prop[number_cols],3)
    
    df_w_outliers['area_ext_sqft_outliers'].unique()
    
    
    
    number_cols_out = ['bdrm_outliers',
                   'price_in_rs_ext_outliers',
                   'area_ext_sqft_outliers',
                   'price_per_area_sqft_outliers',
                   'Carpet_area_outliers',
                   'bdrm_ext_outliers',
                   'bathrm_num_ext_outliers',
                   'no_balcny_ext_outliers',
                   'cvrd_parking_outliers',
                   'open_parking_outliers']
    
    prop_w_out=pd.concat([prop,df_w_outliers[number_cols_out]],axis=1)
    
    prop_w_out['price_in_rs_ext_outliers'].unique()
    
    prs=prop_w_out.loc[prop_w_out['price_in_rs_ext_outliers']==1,:]
   """ 
    
        
        
    return d
  
1      
def extract_info(d, regex ):
    return d.str.extract(regex)


def convert_to_y_n(res):

    typ = type(res)
    
    if typ == 'pandas.core.frame.DataFrame':
        ln = len(res.columns)
        for i in range(ln):
          res.iloc[:,i][res.iloc[:,i].notnull()==True] = 'Y'
          res.iloc[:,i][res.iloc[:,i].isnull()==True] = 'N'
    else:
        res[res.notnull()==True] = 'Y'
        res[res.isnull()==True] = 'N'

    return res

def convert_to_whole(pExt,pType,o):
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
      elif (o == 'm'):   
        pExt.loc[i,0] = pExt.loc[i,0] * rs_conv[pType.loc[i,0]]
      elif (o == 'd'):
        pExt.loc[i,0] = pExt.loc[i,0] / rs_conv[pType.loc[i,0]]
        
    return pExt,pType,err     


def remove_outliers(df):
    typ = type(df)
    b=210
    #if typ == 'pandas.core.frame.DataFrame':
    print(b)
    for col in df:
        print("Kernel Density for"+col)
        plt.figure(figsize=(15,8))
        ax=plt.subplot(211)
        ax.set_title(col)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        df[col].plot(kind='kde')
        print("boxplot for "+col)
        ax1=plt.subplot(212)
        ax1.set_title(b)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        sns.boxplot(x=df[col])
      
        
        #remove any 0's in the column
        #prop.loc[prop.area_ext_sqft == 0, 'area_ext_sqft'] = np.nan
    return b

        

    """
        
        
        q75, q25 = np.percentile(prop.Log_area_ext_sqft.dropna(), [75 ,25])
        iqr = q75 - q25
         
        min = q25 - (iqr*1.5)
        max = q75 + (iqr*1.5)
        
                
        i = 'Log_area_ext_sqft'
         
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.xlim(prop[i].min(), prop[i].max()*1.1)
        plt.axvline(x=min)
        plt.axvline(x=max)
         
        ax = prop[i].plot(kind='kde')
         
        plt.subplot(212)
        plt.xlim(prop[i].min(), prop[i].max()*1.1)
        sns.boxplot(x=prop[i])
        plt.axvline(x=min)
        plt.axvline(x=max)
        
        
        
        
        prop['area_sqft_outliers'] = 0
        
        prop.loc[prop[i] < min, 'area_sqft_outliers'] = 1
        prop.loc[prop[i] > max, 'area_sqft_outliers'] = 1
        
        i='area_ext_sqft'
        
        prop_wo_out = prop.loc[prop['area_sqft_outliers']==0,:]
        
        
        plt.figure(figsize=(15,8))
        plt.subplot(211)
        plt.xlim(prop_wo_out[i].min(), prop_wo_out[i].max()*1.1)
     
        prop[i].plot(kind='kde')
     
        plt.subplot(212)
        plt.xlim(prop_wo_out[i].min(), prop_wo_out[i].max()*1.1)
        sns.boxplot(x=prop_wo_out[i])
    """
    


def substitute_nan(df,):
    
    return null

def IQR_out_range(df,qmin,qmax):
    
    qmin = np.percentile(df,qmin)
    qmax = np.percentile(df,qmax)
    
    iqr = qmax - qmin
    min_val = qmin - (iqr*1.5)
    max_val = qmax + (iqr*1.5)
    
    return min_val, max_val

def sd_outlier(df,n):
    cf = copy.deepcopy(df)
    
    for col in cf:
        new_col = col + '_outliers'
        print(new_col)
        avg = cf[col].dropna().values.mean()
        sd = cf[col].dropna().values.std()
        sd_min = avg - n*sd
        sd_max = avg + n*sd
        print(avg)
        print(sd)
        cf[new_col] = 0
        cf.loc[cf[col]<sd_min,new_col] = 1
        cf.loc[cf[col]>sd_max,new_col] = 1
        print(cf[new_col].unique()) 
        
    return cf    

def outlier_IQR(df,qmin,qmax):
    
    cf = copy.deepcopy(df)
    
    for col in cf:
       print(col)
       new_col = col+'_outliers'
       min_v, max_v = IQR_out_range(cf[col].dropna(),qmin,qmax)
       print(min_v)
       print(max_v)
       
       cf[new_col] = 0
       cf.loc[cf[col] < min_v, new_col] = 1
       cf.loc[cf[col] > max_v, new_col] = 1
       print(cf[new_col].unique())

    return cf

def plot_var(df):
    #typ = type(df)
    #print(typ)
    #if typ == '<class \'pandas.core.frame.DataFrame\'>':
    for col in df:
        print("Kernel Density for"+col)
        plt.figure(figsize=(15,8))
        ax=plt.subplot(6,1,1)
        ax.set_title(col)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        df[col].plot(kind='kde')
        print("boxplot for "+col)
        ax1=plt.subplot(6,1,2)
        ax1.set_title(col)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        sns.boxplot(x=df[col])
        
        # Remove any zeros (otherwise we get (-inf)
        df.loc[df[col] == 0, col] = np.nan 
        # Drop NA
        df.dropna(inplace=True)
        #applying Log Transform to each of the columns to view their 
        df['Log_' + col] = np.log(df[col])
        
        log_col = 'Log_' + col
 
        #plotting the log transform kernel density function
        plt.figure(figsize=(10,8))
        ax2=plt.subplot(6,1,3)
        plt.xlim(df[log_col].min(), df[log_col].max()*1.1)
        ax = df[log_col].plot(kind='kde')
        #plotting the log transform boxplot
        ax3=plt.subplot(6,1,4)
        plt.xlim(df[log_col].min(), df[log_col].max()*1.1)
        sns.boxplot(x=df[log_col])    
        
        #calculating the interquartile range of the log transformed data to determine the cutoff to
        #remove outliers
        
        min_val,max_val=IQR_out_range(df.loc[:,log_col].dropna(),25,75)
        
        #q75 = np.percentile(df.loc[:,log_col].dropna(),75)
        #q25 = np.percentile(df.loc[:,log_col].dropna(),25)
        #iqr = q75 - q25
        
        
        #min_val = q25 - (iqr*1.5)
        #max_val = q75 + (iqr*1.5)
        
        #drawing the line of min value that will be included and max value that will be included 
        #in the plot
        plt.figure(figsize=(10,8))
        plt.subplot(6,1,5)
        plt.xlim(df[log_col].min(), df[log_col].max()*1.1)
        plt.axvline(x=min_val)
        plt.axvline(x=max_val)
         
        ax = df[log_col].plot(kind='kde')
         
        plt.subplot(6,1,6)
        plt.xlim(df[log_col].min(), df[log_col].max()*1.1)
        sns.boxplot(x=df[log_col])
        plt.axvline(x=min_val)
        plt.axvline(x=max_val)
        
        outlier_IQR (df,25,75)
        
        new_col = col+'outliers'
        
        df[new_col] = 0
        
        df.loc[df[new_col] < min_val, new_col] = 1
        df.loc[df[new_col] > max_val, new_col] = 1
       
    return None