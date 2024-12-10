!pip install sqlalchemy==1.3.9
!pip install ipython-sql
!pip install ipython-sql prettytable
%load_ext sql
import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

!pip install -q pandas
%sql sqlite:///my_data1.db
import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

#DROP THE TABLE IF EXISTS

%sql DROP TABLE IF EXISTS SPACEXTABLE;
%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null   %sql select DISTINCT "Launch_Site" from SPACEXTABLE

%sql select "Launch_Site" from SPACEXTABLE where "Launch_Site" like "CCA%" LIMIT 5;
%sql select SUM("PAYLOAD_MASS__KG_") from SPACEXTABLE where "Customer" is "NASA (CRS)";
%sql select AVG("PAYLOAD_MASS__KG_") from SPACEXTABLE where "Booster_Version" is "F9 v1.1";
%sql select min("Date") from SPACEXTABLE where "Landing_Outcome" is "Success (ground pad)"
%sql select Booster_Version from SPACEXTABLE where Landing_Outcome = "Success (drone ship)" and "PAYLOAD_MASS__KG_" BETWEEN 4000 AND 6000;
%sql SELECT COUNT(CASE WHEN Mission_Outcome LIKE 'Success%' THEN 1 END) AS Total_Success, COUNT(CASE WHEN Mission_Outcome LIKE 'Failure%' THEN 1 END) AS Total_Failure FROM SPACEXTABLE;
%sql SELECT Booster_Version FROM SPACEXTABLE WHERE Payload_Mass__KG_ = (SELECT MAX(Payload_Mass__KG_) FROM SPACEXTABLE);
%sql SELECT Landing_Outcome, substr(Date,6,2) AS Month_Number, substr(Date,0,5)='2015' AS Year_2015, Booster_Version, Launch_Site FROM SPACEXTABLE WHERE Landing_Outcome = 'Failure (drone ship)';
%sql SELECT Landing_Outcome, COUNT(*) AS Total from SPACEXTABLE WHERE Date BETWEEN '2010-06-04' and '2017-03-20' GROUP BY Landing_Outcome ORDER BY Total DESC;
