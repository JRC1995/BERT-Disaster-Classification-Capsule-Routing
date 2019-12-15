import subscripts.BigCrisisData
import subscripts.CrisisLexT6
import subscripts.CrisisLexT26
import subscripts.CrisisMMD
import subscripts.CrisisNLP_Crowdflower
import subscripts.CrisisNLP_Volunteers
import subscripts.ICWSM_2018
import subscripts.ISCRAM_2013
import subscripts.SWDM_2013
import os

data_dir = "../Processed_Data/Processed_Data_Intermediate.json"

if os.path.exists(data_dir):
    os.remove(data_dir)

subscripts.BigCrisisData.process()
subscripts.CrisisLexT6.process()
subscripts.CrisisLexT26.process()
subscripts.CrisisMMD.process()
subscripts.CrisisNLP_Crowdflower.process()
subscripts.CrisisNLP_Volunteers.process()
subscripts.ICWSM_2018.process()
subscripts.ISCRAM_2013.process()
subscripts.SWDM_2013.process()
